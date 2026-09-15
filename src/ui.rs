//! ccti TUI: image left (80), chat right (20), status bar.
//! Single visible image, Left/Right walks the render history.

use std::io::stdout;
use std::path::PathBuf;
use std::time::Duration;

use anyhow::Result;
use crossterm::{
    event::{Event, EventStream, KeyCode, KeyModifiers},
    execute,
    terminal::{EnterAlternateScreen, LeaveAlternateScreen, disable_raw_mode, enable_raw_mode},
};
use futures::StreamExt as _;
use ratatui::{
    Terminal,
    backend::CrosstermBackend,
    layout::{Constraint, Direction, Layout, Rect, Size},
    style::{Color, Style},
    text::{Line as RLine, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use ratatui_image::{FontSize, Image, Resize, picker::Picker, protocol::Protocol};
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::comfy::{self, Client};
use crate::prov::{self, Generation};

/// Messages into the UI loop (agent task, render tasks, watcher).
/// `ws: None` means "currently active workspace" (local commands, watcher);
/// agent turns carry their originating workspace.
#[derive(Debug)]
pub enum UiMsg {
    Chat {
        ws: Option<usize>,
        role: String,
        text: String,
    },
    Status(String),
    Image {
        ws: Option<usize>,
        label: String,
        bytes: Vec<u8>,
        generation: Option<Generation>,
    },
    Checkpoints(Vec<String>),
    ChatModel {
        ws: usize,
        model: String,
    },
    AgentBusy(bool),
    AgentDone,
    Error(String),
}

struct GalleryItem {
    label: String,
    bytes: Vec<u8>,
    generation: Option<Generation>,
    proto: Option<Protocol>,
}

/// One workspace: its own pictures, chat, render settings and chat model.
/// The agent backend session is shared; turns are tagged so late answers
/// still land where they were asked.
struct Workspace {
    name: String,
    chat: Vec<(String, String)>,
    gallery: Vec<GalleryItem>,
    gidx: usize,
    settings: RenderSettings,
    chat_model: Option<String>,
}

impl Workspace {
    fn new(name: String) -> Self {
        Self {
            name,
            chat: Vec::new(),
            gallery: Vec::new(),
            gidx: 0,
            settings: RenderSettings::default(),
            chat_model: None,
        }
    }
}

struct App {
    workspaces: Vec<Workspace>,
    active: usize,
    input: String,
    status: String,
    picker: Picker,
    busy: bool,
    agent_offline: bool,
    cells: Size,
    menu: Option<Menu>,
    checkpoints: Option<Vec<String>>,
}

impl App {
    fn ws(&self) -> &Workspace {
        &self.workspaces[self.active]
    }

    fn ws_mut(&mut self) -> &mut Workspace {
        &mut self.workspaces[self.active]
    }

    /// Resolve a workspace index for delivery: explicit tag wins, detached
    /// producers (watcher, local commands) land on the active workspace.
    fn deliver_ws(&self, ws: Option<usize>) -> usize {
        ws.filter(|&i| i < self.workspaces.len())
            .unwrap_or(self.active)
    }

    fn switch_workspace(&mut self, dir: i32) {
        if self.workspaces.is_empty() {
            return;
        }
        let n = self.workspaces.len() as i32;
        self.active = (self.active as i32 + dir).rem_euclid(n) as usize;
        self.rebuild_proto();
        self.status = format!("workspace → {}", self.workspaces[self.active].name);
    }

    fn new_workspace(&mut self) {
        let n = self.workspaces.len() + 1;
        self.workspaces.push(Workspace::new(format!("ws{n}")));
        self.active = self.workspaces.len() - 1;
        self.status = format!("workspace → ws{n}");
    }
}

/// Popup submenu opened from the F-key bar.
#[derive(Debug, Clone, Copy, PartialEq)]
enum MenuKind {
    ImageModel,
    SizePreset,
    Steps,
    Count,
}

#[derive(Debug, Clone)]
struct Menu {
    kind: MenuKind,
    title: String,
    items: Vec<String>,
    selected: usize,
}

impl Menu {
    fn up(&mut self) {
        self.selected = self.selected.saturating_sub(1);
    }

    fn down(&mut self) {
        if self.selected + 1 < self.items.len() {
            self.selected += 1;
        }
    }
}

/// Push an image into a specific workspace (detached producers pass the
/// index resolved by the caller).
fn push_image_to(
    app: &mut App,
    target: usize,
    label: String,
    bytes: Vec<u8>,
    generation: Option<Generation>,
) {
    if target >= app.workspaces.len() {
        return;
    }
    let ws = &mut app.workspaces[target];
    ws.gallery.push(GalleryItem {
        label,
        bytes,
        generation,
        proto: None,
    });
    ws.gidx = ws.gallery.len() - 1;
    // Only the visible workspace pays for protocol encoding right away;
    // background ones encode lazily when first shown.
    let gidx = ws.gidx;
    if target == app.active {
        let cells = app.cells;
        let proto = app.workspaces[target].gallery.get(gidx).and_then(|item| {
            decode(&item.bytes)
                .ok()
                .and_then(|img| app.picker.new_protocol(img, cells, Resize::Fit(None)).ok())
        });
        if let Some(item) = app.workspaces[target].gallery.get_mut(gidx) {
            item.proto = proto;
        }
    }
}

/// Rebuild the F2 model menu items from the cached checkpoint list,
/// preselecting the active workspace's checkpoint.
fn refresh_model_menu(app: &mut App) {
    let ckpts = app.checkpoints.clone().unwrap_or_default();
    let current = app.ws().settings.ckpt.clone();
    let mut items = ckpts;
    if items.is_empty() {
        items.push("loading…".to_string());
    }
    let selected = items.iter().position(|c| c == &current).unwrap_or(0);
    app.menu = Some(Menu {
        kind: MenuKind::ImageModel,
        title: "image model".to_string(),
        items,
        selected,
    });
}

fn send_help(tx: &mpsc::UnboundedSender<UiMsg>) {
    let _ = tx.send(UiMsg::Chat {
        ws: None,
        role: "sys".into(),
        text: "/render TEXT [--w N --h N --steps N --n 1-4] · /models · /chat-model <id> · /cancel · ←/→ images · F1 help · F2 model · F3 size · F4 steps · F5 count · F6/F7 workspace · F8 new · /quit".into(),
    });
}

/// Open the checkpoint menu, fetching the list in the background on first use.
fn open_model_menu(app: &mut App, comfy: &Client, tx: &mpsc::UnboundedSender<UiMsg>) {
    if app.checkpoints.is_none() {
        let tx2 = tx.clone();
        let c = comfy.clone();
        tokio::spawn(async move {
            let _ = tx2.send(UiMsg::Status("loading models…".into()));
            if c.wait_ready(Duration::from_secs(420), |_| {})
                .await
                .is_err()
            {
                let _ = tx2.send(UiMsg::Error("ComfyUI did not wake up".into()));
                return;
            }
            match c.checkpoints().await {
                Ok(list) => {
                    let _ = tx2.send(UiMsg::Checkpoints(list));
                }
                Err(e) => {
                    let _ = tx2.send(UiMsg::Error(format!("models: {e:#}")));
                }
            }
        });
    }
    refresh_model_menu(app);
}

fn open_size_menu(app: &mut App) {
    let names: Vec<String> = PRESETS
        .iter()
        .map(|(name, w, h, steps)| format!("{name} — {w}x{h}, {steps} steps"))
        .collect();
    app.menu = Some(Menu {
        kind: MenuKind::SizePreset,
        title: "size preset".to_string(),
        items: names,
        selected: app.ws().settings.preset.min(PRESETS.len() - 1),
    });
}

fn open_steps_menu(app: &mut App) {
    let items = [4u32, 8, 14, 20, 30]
        .iter()
        .map(|s| format!("{s} steps"))
        .collect::<Vec<_>>();
    let current = app.ws().settings.steps;
    let selected = [4u32, 8, 14, 20, 30]
        .iter()
        .position(|&s| s == current)
        .unwrap_or(1);
    app.menu = Some(Menu {
        kind: MenuKind::Steps,
        title: "quality (steps)".to_string(),
        items,
        selected,
    });
}

fn open_count_menu(app: &mut App) {
    let items = ["1 image", "2 images", "3 images", "4 images"]
        .iter()
        .map(|s| s.to_string())
        .collect::<Vec<_>>();
    app.menu = Some(Menu {
        kind: MenuKind::Count,
        title: "batch count".to_string(),
        items,
        selected: (app.ws().settings.n.saturating_sub(1) as usize).min(3),
    });
}

/// Apply the highlighted menu entry to the active workspace and close.
fn apply_menu(app: &mut App) {
    let Some(menu) = app.menu.take() else {
        return;
    };
    let item = menu.items.get(menu.selected).cloned().unwrap_or_default();
    if item == "loading…" {
        app.menu = Some(menu);
        return;
    }
    let ws = app.ws_mut();
    let msg = match menu.kind {
        MenuKind::ImageModel => {
            ws.settings.ckpt = item.clone();
            format!("image model → {}", short_ckpt(&item))
        }
        MenuKind::SizePreset => {
            ws.settings.set_preset(menu.selected);
            format!("preset → {}", ws.settings.describe())
        }
        MenuKind::Steps => {
            let steps = [4u32, 8, 14, 20, 30][menu.selected.min(4)];
            ws.settings.set_steps(steps as i32);
            format!("steps → {}", ws.settings.steps)
        }
        MenuKind::Count => {
            ws.settings.set_count(menu.selected as u32 + 1);
            format!("count → n={}", ws.settings.n)
        }
    };
    app.status = msg;
}

pub async fn run() -> Result<()> {
    enable_raw_mode().map_err(|_| anyhow::anyhow!("ccti needs a real terminal"))?;
    execute!(stdout(), EnterAlternateScreen)?;
    let res = run_inner().await;
    let _ = disable_raw_mode();
    let _ = execute!(stdout(), LeaveAlternateScreen);
    res
}

/// Wide (side-by-side) layout threshold. MUST match `split` below.
fn is_wide(area: Rect) -> bool {
    area.width >= 100
}

/// Terminal cells are taller than wide (~1:2), so one gap column is only
/// half a gap row in pixels. This returns how many columns equal one row,
/// keeping horizontal and vertical whitespace the same size on screen.
fn hgap_cols(font: FontSize) -> u16 {
    let (w, h) = (font.width.max(1) as u32, font.height.max(1) as u32);
    ((h + w / 2) / w).clamp(1, 8) as u16
}

/// Split main area into (image, chat) with pixel-matched breathing room.
fn split(area: Rect, hgap: u16) -> (Rect, Rect) {
    if is_wide(area) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(80), Constraint::Percentage(20)])
            .spacing(hgap)
            .split(area);
        (cols[0], cols[1])
    } else {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .spacing(1)
            .split(area);
        (rows[0], rows[1])
    }
}

/// All pane rectangles, computed once per frame from the same math so the
/// image protocol size always matches what is actually drawn.
struct Areas {
    header: Rect,
    pic: Rect,
    prov: Rect,
    settings: Rect,
    chat_msgs: Rect,
    chat_input: Rect,
    fkeys: Rect,
    status: Rect,
}

fn layout_areas(area: Rect, font: FontSize) -> Areas {
    let hgap = hgap_cols(font);
    // No outer margin: panes breathe edge to edge, separated only by
    // pixel-matched gaps (one row vertically, hgap columns horizontally).
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Length(1),
            Constraint::Min(0),
            Constraint::Length(1),
            Constraint::Length(1),
        ])
        .spacing(1)
        .split(area);
    let (img_col, chat_col) = split(rows[1], hgap);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(7),
            Constraint::Length(7),
        ])
        .spacing(1)
        .split(img_col);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .spacing(1)
        .split(chat_col);
    Areas {
        header: rows[0],
        pic: left[0],
        prov: left[1],
        settings: left[2],
        chat_msgs: right[0],
        chat_input: right[1],
        fkeys: rows[2],
        status: rows[3],
    }
}

/// Every pane keeps its full frame and top title; air between panes comes
/// from layout gaps, never from dropped borders.
fn pic_block(title: String) -> Block<'static> {
    Block::default().borders(Borders::ALL).title(title)
}

fn prov_block() -> Block<'static> {
    Block::default().borders(Borders::ALL).title("provenance")
}

fn settings_block() -> Block<'static> {
    Block::default().borders(Borders::ALL).title("render")
}

fn chat_block() -> Block<'static> {
    Block::default().borders(Borders::ALL).title("chat")
}

fn input_block() -> Block<'static> {
    Block::default().borders(Borders::ALL)
}

/// Render presets: fast iteration first, quality on demand.
const PRESETS: [(&str, u32, u32, u32); 3] = [
    ("Fast", 512, 512, 8),
    ("Balanced", 768, 768, 14),
    ("Quality", 1024, 1024, 20),
];

/// Defaults behind every `/render` without explicit flags. Adjusted live
/// with single keys while the input line is empty.
#[derive(Debug, Clone, PartialEq)]
struct RenderSettings {
    preset: usize,
    steps: u32,
    n: u32,
    ckpt: String,
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            preset: 0,
            steps: PRESETS[0].3,
            n: 1,
            ckpt: comfy::DEFAULT_CKPT.to_string(),
        }
    }
}

impl RenderSettings {
    fn dims(&self) -> (u32, u32) {
        (PRESETS[self.preset].1, PRESETS[self.preset].2)
    }

    fn name(&self) -> &'static str {
        PRESETS[self.preset].0
    }

    fn set_preset(&mut self, idx: usize) {
        self.preset = idx.min(PRESETS.len() - 1);
        self.steps = PRESETS[self.preset].3;
    }

    fn set_steps(&mut self, steps: i32) {
        self.steps = steps.clamp(4, 50) as u32;
    }

    fn set_count(&mut self, n: u32) {
        self.n = n.clamp(1, 4);
    }

    fn describe(&self) -> String {
        let (w, h) = self.dims();
        format!(
            "{} {w}x{h} · {} steps · n={}",
            self.name(),
            self.steps,
            self.n
        )
    }

    fn lines(&self) -> Vec<String> {
        let (w, h) = self.dims();
        vec![
            "F2 model · F3 size · F4 steps · F5 count".to_string(),
            format!("preset: {} · {w}x{h}", self.name()),
            format!("steps: {} · count: n={}", self.steps, self.n),
            format!("model: {}", short_ckpt(&self.ckpt)),
        ]
    }
}

fn gallery_cells(area: Rect, font: FontSize) -> Size {
    let a = layout_areas(area, font);
    Size::new(
        a.pic.width.saturating_sub(2),
        a.pic.height.saturating_sub(2),
    )
}

fn decode(bytes: &[u8]) -> Result<image::DynamicImage> {
    Ok(image::load_from_memory(bytes)?)
}

impl App {
    fn rebuild_proto(&mut self) {
        let gidx = self.ws().gidx;
        if let Some(item) = self.ws_mut().gallery.get_mut(gidx) {
            item.proto = None;
        }
        // Rebuild in two steps to satisfy the borrow checker.
        let cells = self.cells;
        let proto = self
            .ws()
            .gallery
            .get(gidx)
            .and_then(|item| decode(&item.bytes).ok())
            .and_then(|img| self.picker.new_protocol(img, cells, Resize::Fit(None)).ok());
        if let Some(item) = self.ws_mut().gallery.get_mut(gidx) {
            item.proto = proto;
        }
    }

    fn step_history(&mut self, dir: i32) {
        if self.ws().gallery.is_empty() {
            return;
        }
        let n = self.ws().gallery.len() as i32;
        {
            let ws = self.ws_mut();
            ws.gidx = (ws.gidx as i32 + dir).rem_euclid(n) as usize;
        }
        let cells = self.cells;
        let gidx = self.ws().gidx;
        let needs_encode = self
            .ws()
            .gallery
            .get(gidx)
            .is_some_and(|item| item.proto.is_none());
        if needs_encode {
            let proto = self
                .ws()
                .gallery
                .get(gidx)
                .and_then(|item| decode(&item.bytes).ok())
                .and_then(|img| self.picker.new_protocol(img, cells, Resize::Fit(None)).ok());
            if let Some(item) = self.ws_mut().gallery.get_mut(gidx) {
                item.proto = proto;
            }
        }
    }
}

async fn run_inner() -> Result<()> {
    let backend = CrosstermBackend::new(stdout());
    let mut term = Terminal::new(backend)?;
    let size = term.size()?;
    let full = Rect::new(0, 0, size.width, size.height);

    let picker = Picker::from_query_stdio().unwrap_or_else(|_| Picker::halfblocks());
    let font = picker.font_size();
    let (ui_tx, mut ui_rx) = mpsc::unbounded_channel::<UiMsg>();
    let comfy = Client::new(
        &std::env::var("CCTI_COMFY_URL").unwrap_or_else(|_| comfy::DEFAULT_BASE.into()),
    )?;

    // Agent (ccht + `opencode acp` + bundled MCP). Failure is non-fatal:
    // direct /render keeps working.
    let exe = std::env::current_exe()?;
    let cwd: PathBuf = std::env::var("CCTI_WORKDIR")
        .unwrap_or_else(|_| {
            exe.parent()
                .and_then(|p| p.to_str())
                .unwrap_or("/tmp")
                .to_string()
        })
        .into();
    let agent = Agent::spawn(exe, cwd, ui_tx.clone()).await?;

    // Watcher: agent-side renders land as files; surface them in the gallery.
    {
        let tx = ui_tx.clone();
        tokio::spawn(async move { watch_renders(tx).await });
    }

    let mut app = App {
        workspaces: vec![Workspace::new("main".to_string())],
        active: 0,
        input: String::new(),
        status: "starting…".into(),
        picker,
        busy: false,
        agent_offline: false,
        cells: gallery_cells(full, font),
        menu: None,
        checkpoints: None,
    };
    app.ws_mut().chat.push((
        "sys".into(),
        "ccti ready. Type text for the agent, /render <prompt> to render, F1 for keys.".into(),
    ));
    let mut events = EventStream::new();

    loop {
        term.draw(|f| draw(f, &mut app))?;
        tokio::select! {
            msg = ui_rx.recv() => {
                let Some(msg) = msg else { break };
                match msg {
                    UiMsg::Chat { ws, role, text } => {
                        let target = app.deliver_ws(ws);
                        for chunk in split_long(&text) {
                            app.workspaces[target].chat.push((role.clone(), chunk));
                        }
                    }
                    UiMsg::Status(s) => app.status = s,
                    UiMsg::Image {
                        ws,
                        label,
                        bytes,
                        generation,
                    } => {
                        let target = app.deliver_ws(ws);
                        let label2 = label.clone();
                        push_image_to(&mut app, target, label, bytes, generation);
                        app.workspaces[target]
                            .chat
                            .push(("sys".into(), format!("image: {label2}")));
                    }
                    UiMsg::Checkpoints(list) => {
                        app.checkpoints = Some(list);
                        if let Some(menu) = app.menu.as_mut()
                            && menu.kind == MenuKind::ImageModel
                        {
                            refresh_model_menu(&mut app);
                        }
                        app.status = "models loaded".into();
                    }
                    UiMsg::ChatModel { ws, model } => {
                        if let Some(w) = app.workspaces.get_mut(ws) {
                            w.chat_model = Some(model.clone());
                            w.chat.push((
                                "sys".into(),
                                format!("chat model → {model}"),
                            ));
                        }
                    }
                    UiMsg::AgentBusy(b) => {
                        app.busy = b;
                        if !b { app.status = "ready".into(); }
                    }
                    UiMsg::AgentDone => {
                        app.status = "turn done".into();
                    }
                    UiMsg::Error(e) => {
                        if e.contains("agent offline") { app.agent_offline = true; }
                        app.ws_mut().chat.push(("err".into(), e.clone()));
                        app.status = e;
                    }
                }
            }
            ev = events.next() => {
                let Some(Ok(ev)) = ev else { continue };
                match ev {
                    Event::Key(k) => {
                        if k.modifiers.contains(KeyModifiers::CONTROL) && matches!(k.code, KeyCode::Char('c') | KeyCode::Char('d')) {
                            break;
                        }
                        // Menu modal first: arrows/enter/esc/F-keys drive it,
                        // typing dismisses it and falls through to input.
                        if app.menu.is_some() {
                            match k.code {
                                KeyCode::Up => app.menu.as_mut().unwrap().up(),
                                KeyCode::Down => app.menu.as_mut().unwrap().down(),
                                KeyCode::Enter => apply_menu(&mut app),
                                KeyCode::Esc => {
                                    app.menu = None;
                                    app.status = "ready".into();
                                }
                                KeyCode::F(1) => send_help(&ui_tx),
                                KeyCode::F(2) => open_model_menu(&mut app, &comfy, &ui_tx),
                                KeyCode::F(3) => open_size_menu(&mut app),
                                KeyCode::F(4) => open_steps_menu(&mut app),
                                KeyCode::F(5) => open_count_menu(&mut app),
                                KeyCode::Char(_) => {
                                    app.menu = None;
                                }
                                _ => {}
                            }
                            // Chars fall through to normal typing below.
                            if !matches!(k.code, KeyCode::Char(_)) {
                                continue;
                            }
                        }
                        match k.code {
                            KeyCode::Enter => {
                                let line = std::mem::take(&mut app.input);
                                if line.trim().is_empty() { continue; }
                                if !handle_command(&line, &app, &agent, &comfy, &ui_tx) {
                                    break;
                                }
                            }
                            KeyCode::Char(c) => {
                                // All typing goes to the input line; render
                                // settings change only via F-key submenus.
                                app.input.push(c);
                            }
                            KeyCode::Backspace => { app.input.pop(); }
                            KeyCode::Esc => {
                                if app.busy { agent.cancel(); }
                                else { app.input.clear(); }
                            }
                            KeyCode::Left if app.input.is_empty() => app.step_history(-1),
                            KeyCode::Right if app.input.is_empty() => app.step_history(1),
                            KeyCode::F(1) => send_help(&ui_tx),
                            KeyCode::F(2) => open_model_menu(&mut app, &comfy, &ui_tx),
                            KeyCode::F(3) => open_size_menu(&mut app),
                            KeyCode::F(4) => open_steps_menu(&mut app),
                            KeyCode::F(5) => open_count_menu(&mut app),
                            KeyCode::F(6) => app.switch_workspace(-1),
                            KeyCode::F(7) => app.switch_workspace(1),
                            KeyCode::F(8) => app.new_workspace(),
                            _ => {}
                        }
                    }
                    Event::Resize(_, _) => {
                        let s = term.size()?;
                        let font = app.picker.font_size();
                        app.cells = gallery_cells(Rect::new(0, 0, s.width, s.height), font);
                        app.rebuild_proto();
                    }
                    _ => {}
                }
            }
        }
    }
    Ok(())
}

/// Bottom-left provenance panel: what produced the visible picture.
fn prov_lines(generation: Option<&Generation>) -> Vec<String> {
    let Some(g) = generation else {
        return vec!["no provenance recorded (pre-0.1.2 render)".to_string()];
    };
    let mut prompt = g.prompt.replace('\n', " ");
    if prompt.chars().count() > 160 {
        prompt = format!("{}…", prompt.chars().take(159).collect::<String>());
    }
    vec![
        format!("prompt: {prompt}"),
        format!("model: {}", short_ckpt(&g.ckpt)),
        format!(
            "{}x{} · {} steps · seed {} · {}/{} · {}",
            g.width,
            g.height,
            g.steps,
            g.seed,
            g.index + 1,
            g.n,
            g.software,
        ),
    ]
}

fn short_ckpt(ckpt: &str) -> String {
    ckpt.rsplit('/').next().unwrap_or(ckpt).to_string()
}

fn split_long(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| {
            if l.chars().count() <= 2000 {
                l.to_string()
            } else {
                format!("{}…", l.chars().take(1999).collect::<String>())
            }
        })
        .collect()
}

/// Word-wrap by display-cell width (CJK counts double, umlauts single).
/// Never panics on boundaries; overlong words are hard-split by cells.
fn wrap_text(text: &str, width: usize) -> Vec<String> {
    use unicode_width::{UnicodeWidthChar, UnicodeWidthStr};
    let width = width.max(1);
    let mut out = Vec::new();
    for line in text.lines() {
        if line.trim().is_empty() {
            out.push(String::new());
            continue;
        }
        let mut cur = String::new();
        let mut cur_w = 0;
        for word in line.split_whitespace() {
            let ww = word.width();
            if ww > width {
                if !cur.is_empty() {
                    out.push(std::mem::take(&mut cur));
                    cur_w = 0;
                }
                let mut chunk = String::new();
                let mut cw = 0;
                for ch in word.chars() {
                    let chw = UnicodeWidthChar::width(ch).unwrap_or(0);
                    if cw + chw > width && !chunk.is_empty() {
                        out.push(std::mem::take(&mut chunk));
                        cw = 0;
                    }
                    chunk.push(ch);
                    cw += chw;
                }
                if !chunk.is_empty() {
                    out.push(chunk);
                }
                continue;
            }
            if cur.is_empty() {
                cur.push_str(word);
                cur_w = ww;
            } else if cur_w + 1 + ww <= width {
                cur.push(' ');
                cur.push_str(word);
                cur_w += 1 + ww;
            } else {
                out.push(std::mem::take(&mut cur));
                cur.push_str(word);
                cur_w = ww;
            }
        }
        if !cur.is_empty() {
            out.push(cur);
        }
    }
    if out.is_empty() {
        out.push(String::new());
    }
    out
}

/// Single header line: title left, workspace tabs centered in the
/// middle, per-workspace AIs right. Truncation order on narrow screens:
/// inactive tabs first, then AI names, never the title or active tab.
fn header_line(
    width: usize,
    workspaces: &[Workspace],
    active: usize,
    img_ckpt: &str,
    chat_model: Option<&str>,
) -> RLine<'static> {
    use unicode_width::UnicodeWidthStr;
    const TITLE: &str = "CCTI - Corbet ComfyUi Terminal Interface";
    let tabs: Vec<String> = workspaces
        .iter()
        .enumerate()
        .map(|(i, ws)| format!("[{} {}]", i + 1, ws.name))
        .collect();
    let mut ai = format!(
        "img: {} · chat: {}",
        short_ckpt(img_ckpt),
        chat_model.unwrap_or("default")
    );
    // Squeeze policy: drop inactive tabs, then shorten the AI block.
    // (Explicit index loop: keeps borrowck happy on stable toolchains.)
    let mut shown: Vec<usize> = (0..tabs.len()).collect();
    loop {
        let too_wide = TITLE.width() + 4 + tabs_width(&tabs, &shown) + ai.width() + 4 > width;
        if !too_wide || shown.len() <= 1 {
            break;
        }
        let mut drop_at: Option<usize> = None;
        for (p, &i) in shown.iter().enumerate().rev() {
            if i != active {
                drop_at = Some(p);
                break;
            }
        }
        match drop_at {
            Some(p) => {
                shown.remove(p);
            }
            None => break,
        }
    }
    while TITLE.width() + 4 + tabs_width(&tabs, &shown) + ai.width() + 4 > width && ai.width() > 12
    {
        ai.pop();
    }
    let mut spans = vec![Span::styled(TITLE, Style::default().fg(Color::Cyan))];
    let tabs_w = tabs_width(&tabs, &shown);
    let used = TITLE.width() + 2 + tabs_w + ai.width() + 2;
    let free = width.saturating_sub(used);
    // Tabs sit in the middle: split the free space around them.
    let pad_left = 2 + free / 2;
    let pad_right = width.saturating_sub(TITLE.width() + pad_left + tabs_w + ai.width());
    spans.push(Span::raw(" ".repeat(pad_left)));
    for &i in &shown {
        let tab = format!("{} ", tabs[i]);
        if i == active {
            spans.push(Span::styled(
                tab,
                Style::default().fg(Color::Black).bg(Color::Cyan),
            ));
        } else {
            spans.push(Span::styled(tab, Style::default().fg(Color::DarkGray)));
        }
    }
    spans.push(Span::raw(" ".repeat(pad_right.max(1))));
    spans.push(Span::styled(ai, Style::default().fg(Color::DarkGray)));
    RLine::from(spans)
}

fn tabs_width(tabs: &[String], shown: &[usize]) -> usize {
    use unicode_width::UnicodeWidthStr;
    shown.iter().map(|&i| tabs[i].width() + 1).sum()
}

fn fkey_bar() -> String {
    "F1 help · F2 model · F3 size · F4 steps · F5 batch · F6/F7 workspace · F8 new".to_string()
}

fn draw(f: &mut ratatui::Frame<'_>, app: &mut App) {
    let a = layout_areas(f.area(), app.picker.font_size());
    let ws = app.ws();

    f.render_widget(
        Paragraph::new(header_line(
            a.header.width as usize,
            &app.workspaces,
            app.active,
            ws.settings.ckpt.as_str(),
            ws.chat_model.as_deref(),
        )),
        a.header,
    );

    let title = if ws.gallery.is_empty() {
        "image — nothing rendered yet".to_string()
    } else {
        let item = &ws.gallery[ws.gidx];
        format!(
            "image {}/{} — {} (←/→)",
            ws.gidx + 1,
            ws.gallery.len(),
            item.label
        )
    };
    f.render_widget(pic_block(title), a.pic);
    if let Some(item) = ws.gallery.get(ws.gidx)
        && let Some(proto) = &item.proto
    {
        let inner = Rect {
            x: a.pic.x + 1,
            y: a.pic.y + 1,
            width: a.pic.width.saturating_sub(2),
            height: a.pic.height.saturating_sub(2),
        };
        f.render_widget(Image::new(proto), inner);
    }
    let prov = prov_lines(ws.gallery.get(ws.gidx).and_then(|i| i.generation.as_ref()));
    f.render_widget(
        Paragraph::new(prov.join("\n"))
            .block(prov_block())
            .wrap(Wrap { trim: true }),
        a.prov,
    );
    f.render_widget(
        Paragraph::new(ws.settings.lines().join("\n")).block(settings_block()),
        a.settings,
    );

    // Chat pane (right, ~20). The per-workspace AIs live in the header
    // now; this column is chat plus input only.
    let w = a.chat_msgs.width.saturating_sub(2).max(10) as usize;
    let mut lines: Vec<RLine> = Vec::new();
    for (role, text) in &ws.chat {
        let color = match role.as_str() {
            "you" => Color::Green,
            "agent" => Color::Cyan,
            "tool" => Color::Yellow,
            "err" => Color::Red,
            _ => Color::DarkGray,
        };
        let prefix = format!("[{role}] ");
        let mut first_visual = true;
        for phys in split_long(text) {
            let avail = w
                .saturating_sub(if first_visual { prefix.len() } else { 6 })
                .max(10);
            for visual in wrap_text(&phys, avail) {
                if first_visual {
                    lines.push(RLine::from(vec![
                        Span::styled(prefix.clone(), Style::default().fg(color)),
                        Span::raw(visual),
                    ]));
                    first_visual = false;
                } else {
                    lines.push(RLine::from(format!("      {visual}")));
                }
            }
        }
    }
    let visible = a.chat_msgs.height.saturating_sub(2) as usize;
    let skip = lines.len().saturating_sub(visible.max(1));
    let msg = Paragraph::new(lines.into_iter().skip(skip).collect::<Vec<_>>())
        .block(chat_block())
        .wrap(Wrap { trim: false });
    f.render_widget(msg, a.chat_msgs);
    let prompt = if app.busy {
        "…working (Esc cancels)"
    } else {
        ">"
    };
    let input = Paragraph::new(format!("{prompt} {}", app.input)).block(input_block());
    f.render_widget(input, a.chat_input);

    let fkeys = Paragraph::new(fkey_bar()).style(Style::default().fg(Color::DarkGray));
    f.render_widget(fkeys, a.fkeys);
    let status = Paragraph::new(app.status.clone()).style(Style::default().fg(Color::DarkGray));
    f.render_widget(status, a.status);

    if let Some(menu) = &app.menu {
        render_menu(f, f.area(), menu);
    }
}

/// Centered popup submenu.
fn render_menu(f: &mut ratatui::Frame<'_>, area: Rect, menu: &Menu) {
    let width = 46u16.min(area.width.saturating_sub(4)).max(20);
    let height = (menu.items.len() as u16 + 4)
        .min(area.height.saturating_sub(4))
        .max(6);
    let popup = Rect {
        x: area.x + (area.width.saturating_sub(width)) / 2,
        y: area.y + (area.height.saturating_sub(height)) / 2,
        width,
        height,
    };
    f.render_widget(ratatui::widgets::Clear, popup);
    let mut lines = Vec::new();
    for (i, item) in menu.items.iter().enumerate() {
        if i == menu.selected {
            lines.push(RLine::from(Span::styled(
                format!("> {item}"),
                Style::default().fg(Color::Black).bg(Color::Cyan),
            )));
        } else {
            lines.push(RLine::from(format!("  {item}")));
        }
    }
    lines.push(RLine::from(""));
    lines.push(RLine::from(Span::styled(
        "↑↓ navigate · Enter select · Esc close",
        Style::default().fg(Color::DarkGray),
    )));
    f.render_widget(
        Paragraph::new(lines).block(
            Block::default()
                .borders(Borders::ALL)
                .title(menu.title.clone()),
        ),
        popup,
    );
}

/// Local slash commands. Returns false to quit.
fn handle_command(
    line: &str,
    app: &App,
    agent: &Agent,
    comfy: &Client,
    tx: &mpsc::UnboundedSender<UiMsg>,
) -> bool {
    let line = line.trim();
    if !line.starts_with('/') {
        if app.agent_offline {
            let _ = tx.send(UiMsg::Error("agent offline — use /render <prompt>".into()));
        } else {
            agent.prompt(line.to_string(), app.active);
            let _ = tx.send(UiMsg::Chat {
                ws: None,
                role: "you".into(),
                text: line.to_string(),
            });
        }
        return true;
    }
    let mut parts = line[1..].split_whitespace();
    match parts.next().unwrap_or("") {
        "quit" | "q" => return false,
        "help" | "h" => {
            let _ = tx.send(UiMsg::Chat { ws: None, role: "sys".into(), text:
                "/render TEXT [--w N --h N --steps N --n 1-4] direct render (settings box defaults) · /models list checkpoints · /chat-model <id> switch chat model · /cancel stop agent turn · ←/→ images · F1 keys · F2 model · F3 size · F4 steps · F5 count · F6/F7 workspace · F8 new · /quit".into() });
        }
        "cancel" => agent.cancel(),
        "models" => {
            let tx = tx.clone();
            let c = comfy.clone();
            tokio::spawn(async move {
                let _ = tx.send(UiMsg::Status("waking ComfyUI…".into()));
                if let Err(e) = c
                    .wait_ready(Duration::from_secs(420), |p| {
                        let _ = tx.send(UiMsg::Status(p));
                    })
                    .await
                {
                    let _ = tx.send(UiMsg::Error(format!("{e:#}")));
                    return;
                }
                match c.checkpoints().await {
                    Ok(list) => {
                        let _ = tx.send(UiMsg::Checkpoints(list.clone()));
                        let _ = tx.send(UiMsg::Chat {
                            ws: None,
                            role: "sys".into(),
                            text: list.join(", "),
                        });
                    }
                    Err(e) => {
                        let _ = tx.send(UiMsg::Error(format!("models: {e:#}")));
                    }
                }
            });
        }
        "render" => {
            let rest = line[1..].trim_start_matches("render").trim().to_string();
            let args = parse_render(&rest);
            if args.prompt.is_empty() {
                let _ = tx.send(UiMsg::Error(
                    "usage: /render TEXT [--w N --h N --steps N --n 1-4]".into(),
                ));
                return true;
            }
            let r = args.resolve(&app.ws().settings);
            let tx2 = tx.clone();
            let c = comfy.clone();
            tokio::spawn(async move {
                direct_render(&c, &tx2, r).await;
            });
            let _ = tx.send(UiMsg::Chat {
                ws: None,
                role: "you".into(),
                text: format!("/render {}", args.prompt),
            });
        }
        "chat-model" => {
            let id = line[12..].trim().to_string();
            if id.is_empty() {
                let _ = tx.send(UiMsg::Error("usage: /chat-model <model-id>".into()));
                return true;
            }
            agent.set_chat_model(app.active, id);
            let _ = tx.send(UiMsg::Status("requesting chat model…".into()));
        }
        other => {
            let _ = tx.send(UiMsg::Error(format!("unknown /{other} — /help")));
        }
    }
    true
}

/// Parsed `/render` flags; every numeric is optional and falls back to
/// the live render settings box when absent.
#[derive(Debug, PartialEq)]
struct RenderArgs {
    prompt: String,
    w: Option<u32>,
    h: Option<u32>,
    steps: Option<u32>,
    n: Option<u32>,
}

/// Fully resolved render parameters: explicit flags win, the live
/// settings box (including its image model) supplies the rest.
#[derive(Debug, Clone, PartialEq)]
struct ResolvedRender {
    prompt: String,
    w: u32,
    h: u32,
    steps: u32,
    n: u32,
    ckpt: String,
}

impl RenderArgs {
    fn resolve(&self, settings: &RenderSettings) -> ResolvedRender {
        let (dw, dh) = settings.dims();
        ResolvedRender {
            prompt: self.prompt.clone(),
            w: self.w.unwrap_or(dw),
            h: self.h.unwrap_or(dh),
            steps: self.steps.unwrap_or(settings.steps),
            n: self.n.unwrap_or(settings.n),
            ckpt: settings.ckpt.clone(),
        }
    }
}

fn parse_render(rest: &str) -> RenderArgs {
    let (mut w, mut h, mut steps, mut n) = (None, None, None, None);
    let mut words: Vec<&str> = Vec::new();
    let mut it = rest.split_whitespace().peekable();
    while let Some(tok) = it.next() {
        match tok {
            "--w" => w = it.next().and_then(|v| v.parse().ok()).or(w),
            "--h" => h = it.next().and_then(|v| v.parse().ok()).or(h),
            "--steps" => steps = it.next().and_then(|v| v.parse().ok()).or(steps),
            "--n" => {
                n = it
                    .next()
                    .and_then(|v| v.parse().ok())
                    .map(|v: u32| v.clamp(1, 4))
                    .or(n)
            }
            _ => words.push(tok),
        }
    }
    RenderArgs {
        prompt: words.join(" "),
        w,
        h,
        steps,
        n,
    }
}

async fn direct_render(c: &Client, tx: &mpsc::UnboundedSender<UiMsg>, r: ResolvedRender) {
    let ResolvedRender {
        prompt,
        w,
        h,
        steps,
        n,
        ckpt,
    } = r;
    let send = |m: UiMsg| {
        let _ = tx.send(m);
    };
    if let Err(e) = c
        .wait_ready(Duration::from_secs(420), |p| send(UiMsg::Status(p)))
        .await
    {
        send(UiMsg::Error(format!("{e:#}")));
        return;
    }
    let seed = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(7);
    let refs = match c
        .render(
            comfy::RenderOpts {
                prompt: &prompt,
                ckpt: &ckpt,
                width: w,
                height: h,
                steps,
                seed,
                n,
            },
            |p| send(UiMsg::Status(p)),
        )
        .await
    {
        Ok(r) => r,
        Err(e) => {
            send(UiMsg::Error(format!("render: {e:#}")));
            return;
        }
    };
    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .map(|d| d.as_secs())
        .unwrap_or(0);
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let dir = std::path::PathBuf::from(format!("{home}/images/generated"));
    for (i, img) in refs.iter().enumerate() {
        let generation = Generation {
            prompt: prompt.clone(),
            negative: "blurry, watermark, text, deformed".to_string(),
            ckpt: ckpt.clone(),
            width: w,
            height: h,
            steps,
            cfg: 1.5,
            sampler: "euler".to_string(),
            scheduler: "normal".to_string(),
            seed,
            n: refs.len() as u32,
            index: i as u32,
            software: format!("ccti {}", env!("CARGO_PKG_VERSION")),
            created_unix: ts,
        };
        match c.download(img).await {
            Ok(bytes) => {
                let stem = format!("ccti_{ts}_{i}");
                match prov::store_rendered(&dir, &stem, &bytes, &generation) {
                    Ok((png_path, _)) => send(UiMsg::Image {
                        ws: None,
                        label: png_path.display().to_string(),
                        bytes,
                        generation: Some(generation),
                    }),
                    Err(e) => send(UiMsg::Error(format!("save: {e:#}"))),
                }
            }
            Err(e) => send(UiMsg::Error(format!("download: {e:#}"))),
        }
    }
    send(UiMsg::Status("ready".into()));
}

/// Surface agent-side renders (MCP tool saves ccti_*.png) in the gallery.
async fn watch_renders(tx: mpsc::UnboundedSender<UiMsg>) {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    let dir = format!("{home}/images/generated");
    let mut seen = std::time::SystemTime::now();
    loop {
        tokio::time::sleep(Duration::from_secs(3)).await;
        let Ok(mut rd) = tokio::fs::read_dir(&dir).await else {
            continue;
        };
        while let Ok(Some(ent)) = rd.next_entry().await {
            let name = ent.file_name().to_string_lossy().into_owned();
            if !name.starts_with("ccti_") || !name.ends_with(".png") {
                continue;
            }
            let Ok(meta) = ent.metadata().await else {
                continue;
            };
            let Ok(mtime) = meta.modified() else { continue };
            if mtime > seen {
                seen = mtime;
                if let Ok(bytes) = tokio::fs::read(ent.path()).await
                    && image::load_from_memory(&bytes).is_ok()
                {
                    let json_path = ent.path().with_extension("json");
                    let generation = Generation::load_sidecar(&json_path).ok();
                    let _ = tx.send(UiMsg::Image {
                        ws: None,
                        label: name,
                        bytes,
                        generation,
                    });
                }
            }
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn push_image_to_active(
        app: &mut App,
        label: String,
        bytes: Vec<u8>,
        generation: Option<Generation>,
    ) {
        let target = app.active;
        push_image_to(app, target, label, bytes, generation);
    }

    fn test_app() -> App {
        let mut ws = Workspace::new("main".to_string());
        ws.chat = Vec::new();
        App {
            workspaces: vec![ws],
            active: 0,
            input: String::new(),
            status: String::new(),
            picker: Picker::halfblocks(),
            busy: false,
            agent_offline: false,
            cells: Size::new(80, 24),
            menu: None,
            checkpoints: None,
        }
    }

    fn test_png() -> Vec<u8> {
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([200, 100, 50]));
        let dyn_img = image::DynamicImage::ImageRgb8(img);
        let mut buf = Vec::new();
        let mut cur = std::io::Cursor::new(&mut buf);
        dyn_img
            .write_to(&mut cur, image::ImageFormat::Png)
            .expect("encode test png");
        buf
    }

    #[test]
    fn wide_screens_split_80_20_side_by_side() {
        let (img, chat) = split(Rect::new(0, 0, 120, 40), 2);
        // 118 cells for panes + 2 gap cells.
        assert_eq!(img.width + chat.width, 118);
        assert_eq!(img.height, chat.height);
        assert_eq!(chat.x, img.x + img.width + 2);
    }

    #[test]
    fn narrow_screens_stack_image_over_chat() {
        let (img, chat) = split(Rect::new(0, 0, 80, 40), 2);
        assert_eq!(img.width, 80);
        // 39 cells for panes + 1 gap cell.
        assert_eq!(img.height + chat.height, 39);
        assert_eq!(chat.y, img.y + img.height + 1);
    }

    #[test]
    fn hgap_matches_pixels_not_cells() {
        use ratatui_image::FontSize;
        // Classic 1:2 terminal cell: two columns equal one row.
        assert_eq!(hgap_cols(FontSize::new(8, 16)), 2);
        assert_eq!(hgap_cols(FontSize::new(10, 20)), 2);
        assert_eq!(hgap_cols(FontSize::new(9, 18)), 2);
        // Square-ish cells collapse to a single column.
        assert_eq!(hgap_cols(FontSize::new(8, 8)), 1);
        // Degenerate input never yields zero.
        assert_eq!(hgap_cols(FontSize::new(0, 0)), 1);
    }

    #[test]
    fn history_walks_and_wraps() {
        let mut app = test_app();
        app.step_history(1); // empty: no panic, no move
        assert_eq!(app.ws().gidx, 0);
        push_image_to_active(&mut app, "a".into(), test_png(), None);
        push_image_to_active(&mut app, "b".into(), test_png(), None);
        assert_eq!(app.ws().gidx, 1);
        app.step_history(1);
        assert_eq!(app.ws().gidx, 0);
        app.step_history(-1);
        assert_eq!(app.ws().gidx, 1);
    }

    #[test]
    fn workspaces_switch_and_stay_isolated() {
        let mut app = test_app();
        push_image_to_active(&mut app, "a".into(), test_png(), None);
        app.new_workspace();
        assert_eq!(app.active, 1);
        assert_eq!(app.ws().name, "ws2");
        assert!(app.ws().gallery.is_empty());
        app.switch_workspace(-1);
        assert_eq!(app.active, 0);
        assert_eq!(app.ws().gallery.len(), 1);
        app.switch_workspace(1);
        assert_eq!(app.active, 1);
        app.switch_workspace(9); // wraps around two workspaces
        assert_eq!(app.active, 0);
    }

    #[test]
    fn menu_navigation_clamps_at_ends() {
        let mut m = Menu {
            kind: MenuKind::Count,
            title: "x".to_string(),
            items: vec!["1".to_string(), "2".to_string()],
            selected: 0,
        };
        m.up();
        assert_eq!(m.selected, 0);
        m.down();
        m.down();
        assert_eq!(m.selected, 1);
    }

    #[test]
    fn apply_menu_writes_active_workspace_settings() {
        let mut app = test_app();
        app.menu = Some(Menu {
            kind: MenuKind::Count,
            title: "x".to_string(),
            items: vec!["1 image".to_string(), "2 images".to_string()],
            selected: 1,
        });
        apply_menu(&mut app);
        assert_eq!(app.ws().settings.n, 2);
        assert!(app.menu.is_none());
    }

    fn args(
        prompt: &str,
        w: Option<u32>,
        h: Option<u32>,
        steps: Option<u32>,
        n: Option<u32>,
    ) -> RenderArgs {
        RenderArgs {
            prompt: prompt.to_string(),
            w,
            h,
            steps,
            n,
        }
    }

    #[test]
    fn parse_render_plain_prompt_leaves_everything_unset() {
        assert_eq!(parse_render("a fox"), args("a fox", None, None, None, None));
    }

    #[test]
    fn parse_render_honours_flags() {
        assert_eq!(
            parse_render("a fox --w 1024 --h 512 --steps 20 --n 3"),
            args("a fox", Some(1024), Some(512), Some(20), Some(3))
        );
    }

    #[test]
    fn parse_render_ignores_broken_flag_values() {
        assert_eq!(
            parse_render("x --steps abc"),
            args("x", None, None, None, None)
        );
        assert_eq!(
            parse_render("--w 100"),
            args("", Some(100), None, None, None)
        );
    }

    #[test]
    fn parse_render_clamps_batch() {
        assert_eq!(parse_render("x --n 99").n, Some(4));
        assert_eq!(parse_render("x --n 0").n, Some(1));
    }

    #[test]
    fn resolve_prefers_flags_over_settings() {
        let settings = RenderSettings::default(); // Fast 512, 8 steps, n=1
        let r = parse_render("a fox").resolve(&settings);
        assert_eq!(r.prompt, "a fox");
        assert_eq!((r.w, r.h, r.steps, r.n), (512, 512, 8, 1));
        assert_eq!(r.ckpt, comfy::DEFAULT_CKPT);
        let r = parse_render("a fox --w 1024 --n 2").resolve(&settings);
        assert_eq!((r.w, r.h, r.steps, r.n), (1024, 512, 8, 2));
        let mut quality = RenderSettings::default();
        quality.set_preset(2); // Quality 1024, 20 steps
        let r = parse_render("a fox").resolve(&quality);
        assert_eq!((r.w, r.h, r.steps, r.n), (1024, 1024, 20, 1));
    }

    #[test]
    fn settings_setters_clamp_and_describe() {
        let mut s = RenderSettings::default();
        assert_eq!(s.describe(), "Fast 512x512 · 8 steps · n=1");
        s.set_preset(2);
        assert_eq!(s.name(), "Quality");
        assert_eq!(s.steps, 20);
        s.set_preset(99);
        assert_eq!(s.name(), "Quality"); // clamps
        s.set_steps(1000);
        assert_eq!(s.steps, 50);
        s.set_steps(-1000);
        assert_eq!(s.steps, 4);
        s.set_count(3);
        assert_eq!(s.n, 3);
        s.set_count(99);
        assert_eq!(s.n, 4);
        s.set_preset(1);
        assert_eq!(s.describe(), "Balanced 768x768 · 14 steps · n=4");
        assert_eq!(s.ckpt, comfy::DEFAULT_CKPT);
    }

    #[test]
    fn wrap_breaks_on_words_not_mid_word() {
        assert_eq!(
            wrap_text("hello world foo", 8),
            vec!["hello", "world", "foo"]
        );
        assert_eq!(wrap_text("hello world", 11), vec!["hello world"]);
    }

    #[test]
    fn wrap_counts_cells_not_chars() {
        // Umlauts are one cell; CJK two.
        assert_eq!(wrap_text("Größe ändern", 7), vec!["Größe", "ändern"]);
        assert_eq!(wrap_text("日本語テスト", 6), vec!["日本語", "テスト"]);
    }

    #[test]
    fn wrap_splits_overlong_words_by_cells() {
        assert_eq!(wrap_text("abcdefghij", 4), vec!["abcd", "efgh", "ij"]);
    }

    #[test]
    fn split_long_never_panics_on_multibyte() {
        let s = "ä".repeat(3000);
        let out = split_long(&s);
        assert_eq!(out.len(), 1);
        assert!(out[0].chars().count() <= 2000);
    }

    #[test]
    fn render_wraps_long_german_prompt_inside_provenance_box() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut app = test_app();
        app.cells = Size::new(90, 20);
        app.ws_mut().chat.push((
            "agent".into(),
            "Größe prüfen: ein äußerst langes deutsches Wort wie Donaudampfschifffahrtsgesellschaft und 日本語混じり".into(),
        ));
        let generation = Generation {
            prompt: "Eine äußerst lange deutsche Beschreibung mit Umlauten äöü und scharfem ß, die auf schmaler Breite korrekt umbrechen muss ohne zu panicen".into(),
            negative: "unscharf".into(),
            ckpt: "model.safetensors".into(),
            width: 512,
            height: 512,
            steps: 8,
            cfg: 1.5,
            sampler: "euler".into(),
            scheduler: "normal".into(),
            seed: 1,
            n: 1,
            index: 0,
            software: "ccti".into(),
            created_unix: 1,
        };
        push_image_to_active(&mut app, "test".into(), test_png(), Some(generation));
        let backend = TestBackend::new(120, 40);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        let buf = term.backend().buffer().clone();
        let mut text = String::new();
        for y in 0..40 {
            let mut line = String::new();
            for x in 0..120 {
                line.push_str(buf[(x, y)].symbol());
            }
            // No visual line may exceed the terminal width in cells.
            assert!(
                line.chars().count() <= 120,
                "overflow on row {y}: {} cells",
                line.chars().count()
            );
            // No doubled vertical seams anywhere.
            assert!(
                !line.contains("││"),
                "doubled vertical border on row {y}: {line}"
            );
            text.push_str(line.trim_end());
            text.push('\n');
        }
        // Prompt survives wrapping (reflowed, so compare whitespace-collapsed).
        let flat: String = text.split_whitespace().collect::<Vec<_>>().join(" ");
        assert!(
            flat.contains("Eine äußerst lange deutsche Beschreibung"),
            "prompt lost in render"
        );
        assert!(flat.contains("Größe prüfen"), "chat lost in render");
        assert!(flat.contains("provenance"), "panel title lost");
        assert!(
            flat.contains("CCTI - Corbet ComfyUi Terminal Interface"),
            "header title lost"
        );
        assert!(flat.contains("[1 main]"), "workspace tab lost");
        assert!(flat.contains("img:"), "header AI info lost");
        assert!(flat.contains("F2 model"), "fkey bar lost");
        // Horizontal seam between picture and provenance is a single line:
        // recompute the layout and check the provenance top row has no ─ run.
        let full = Rect::new(0, 0, 120, 39);
        let (img_rect, _) = split(full, 2);
        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([
                Constraint::Min(0),
                Constraint::Length(8),
                Constraint::Length(8),
            ])
            .spacing(1)
            .split(img_rect);
        let seam_y = left[1].y as usize;
        let mut seam = String::new();
        for x in 0..120 {
            seam.push_str(buf[(x, seam_y as u16)].symbol());
        }
        let middle: String = seam.chars().skip(1).take(118).collect();
        assert!(!middle.contains('─'), "doubled horizontal seam: {seam}");
    }

    #[test]
    fn narrow_layout_has_no_doubled_seams_either() {
        use ratatui::{Terminal, backend::TestBackend};
        let mut app = test_app();
        app.cells = Size::new(60, 12);
        push_image_to_active(&mut app, "test".into(), test_png(), None);
        let backend = TestBackend::new(80, 24);
        let mut term = Terminal::new(backend).unwrap();
        term.draw(|f| draw(f, &mut app)).unwrap();
        let buf = term.backend().buffer().clone();
        for y in 0..24 {
            let mut line = String::new();
            for x in 0..80 {
                line.push_str(buf[(x, y)].symbol());
            }
            assert!(!line.contains("││"), "doubled vertical on row {y}");
        }
    }

    #[test]
    fn header_line_marks_active_tab_and_agent() {
        let ws = vec![
            Workspace::new("main".to_string()),
            Workspace::new("portraits".to_string()),
        ];
        let line = header_line(140, &ws, 1, "juggernautXL.safetensors", Some("pro"));
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.starts_with("CCTI - Corbet ComfyUi Terminal Interface"));
        assert!(text.contains("[2 portraits]"));
        assert!(text.contains("img: juggernautXL.safetensors"));
        assert!(text.contains("chat: pro"));
        let line = header_line(140, &ws, 0, "m.safetensors", None);
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("[1 main]"));
        assert!(text.contains("chat: default"));
    }

    #[test]
    fn header_line_squeezes_inactive_tabs_first() {
        let ws = vec![
            Workspace::new("main".to_string()),
            Workspace::new("portraits".to_string()),
            Workspace::new("extra".to_string()),
        ];
        // Narrow: active tab survives, others may go.
        let line = header_line(70, &ws, 2, "m.safetensors", None);
        let text: String = line.spans.iter().map(|s| s.content.as_ref()).collect();
        assert!(text.contains("[3 extra]"), "active tab lost: {text}");
        assert!(text.starts_with("CCTI - Corbet ComfyUi Terminal Interface"));
    }

    #[test]
    fn provenance_panel_shows_prompt_model_and_run() {
        let generation = Generation {
            prompt: "a fox in snow".into(),
            negative: "blurry".into(),
            ckpt: "some/dir/model.safetensors".into(),
            width: 512,
            height: 512,
            steps: 8,
            cfg: 1.5,
            sampler: "euler".into(),
            scheduler: "normal".into(),
            seed: 7,
            n: 2,
            index: 0,
            software: "ccti 0.1.2".into(),
            created_unix: 1,
        };
        let lines = prov_lines(Some(&generation));
        let all = lines.join("\n");
        assert!(all.contains("a fox in snow"), "prompt missing:\n{all}");
        assert!(all.contains("model.safetensors"), "model missing:\n{all}");
        assert!(all.contains("512x512"), "size missing:\n{all}");
        assert!(all.contains('7'), "seed missing:\n{all}");
        assert_eq!(
            prov_lines(None),
            vec!["no provenance recorded (pre-0.1.2 render)".to_string()]
        );
        assert_eq!(short_ckpt("a/b/c.safetensors"), "c.safetensors");
    }
}
