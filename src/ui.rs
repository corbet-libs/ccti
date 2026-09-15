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
    layout::{Constraint, Direction, Layout, Margin, Rect, Size},
    style::{Color, Style},
    text::{Line as RLine, Span},
    widgets::{Block, Borders, Paragraph, Wrap},
};
use ratatui_image::{Image, Resize, picker::Picker, protocol::Protocol};
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::comfy::{self, Client};
use crate::prov::{self, Generation};

/// Messages into the UI loop (agent task, render tasks, watcher).
#[derive(Debug)]
pub enum UiMsg {
    Chat {
        role: String,
        text: String,
    },
    Status(String),
    Image {
        label: String,
        bytes: Vec<u8>,
        generation: Option<Generation>,
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

struct App {
    chat: Vec<(String, String)>,
    input: String,
    status: String,
    gallery: Vec<GalleryItem>,
    gidx: usize,
    picker: Picker,
    busy: bool,
    agent_offline: bool,
    cells: Size,
    settings: RenderSettings,
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

/// Split main area into (image, chat) with one breathing cell between them.
/// Wide screens go 80/20 side by side, narrow ones stack image over chat.
fn split(area: Rect) -> (Rect, Rect) {
    if is_wide(area) {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(80), Constraint::Percentage(20)])
            .spacing(1)
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
    pic: Rect,
    prov: Rect,
    settings: Rect,
    chat_msgs: Rect,
    chat_input: Rect,
    status: Rect,
}

fn layout_areas(area: Rect) -> Areas {
    let outer = area.inner(Margin::new(1, 1));
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .spacing(1)
        .split(outer);
    let (img_col, chat_col) = split(rows[0]);
    let left = Layout::default()
        .direction(Direction::Vertical)
        .constraints([
            Constraint::Min(0),
            Constraint::Length(8),
            Constraint::Length(8),
        ])
        .spacing(1)
        .split(img_col);
    let right = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .spacing(1)
        .split(chat_col);
    Areas {
        pic: left[0],
        prov: left[1],
        settings: left[2],
        chat_msgs: right[0],
        chat_input: right[1],
        status: rows[1],
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
}

impl Default for RenderSettings {
    fn default() -> Self {
        Self {
            preset: 0,
            steps: PRESETS[0].3,
            n: 1,
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

    fn cycle_preset(&mut self, dir: i32) {
        let n = PRESETS.len() as i32;
        self.preset = (self.preset as i32 + dir).rem_euclid(n) as usize;
        self.steps = PRESETS[self.preset].3;
    }

    fn adjust_steps(&mut self, delta: i32) {
        self.steps = (self.steps as i32 + delta).clamp(4, 50) as u32;
    }

    fn cycle_count(&mut self) {
        self.n = if self.n >= 4 { 1 } else { self.n + 1 };
    }

    /// Single-key adjustment, active only while the input line is empty.
    /// Returns a status line when the key applied, None to keep typing it.
    fn apply_key(&mut self, c: char) -> Option<String> {
        match c {
            '[' => {
                self.cycle_preset(-1);
                Some(format!("preset → {}", self.describe()))
            }
            ']' => {
                self.cycle_preset(1);
                Some(format!("preset → {}", self.describe()))
            }
            '-' => {
                self.adjust_steps(-2);
                Some(format!("steps → {}", self.steps))
            }
            '+' | '=' => {
                self.adjust_steps(2);
                Some(format!("steps → {}", self.steps))
            }
            _ => None,
        }
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
            "[ ] preset · -/+ steps · Tab count (empty input only)".to_string(),
            format!("preset: {} · {w}x{h}", self.name()),
            format!("steps: {} · count: n={}", self.steps, self.n),
            format!("model: {}", short_ckpt(comfy::DEFAULT_CKPT)),
        ]
    }
}

fn gallery_cells(area: Rect) -> Size {
    let a = layout_areas(area);
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
        if let Some(item) = self.gallery.get_mut(self.gidx) {
            item.proto = decode(&item.bytes).ok().and_then(|img| {
                self.picker
                    .new_protocol(img, self.cells, Resize::Fit(None))
                    .ok()
            });
        }
    }

    fn push_image(&mut self, label: String, bytes: Vec<u8>, generation: Option<Generation>) {
        self.gallery.push(GalleryItem {
            label,
            bytes,
            generation,
            proto: None,
        });
        self.gidx = self.gallery.len() - 1;
        self.rebuild_proto();
    }

    fn step_history(&mut self, dir: i32) {
        if self.gallery.is_empty() {
            return;
        }
        let n = self.gallery.len() as i32;
        self.gidx = (self.gidx as i32 + dir).rem_euclid(n) as usize;
        let cells = self.cells;
        if let Some(item) = self.gallery.get_mut(self.gidx)
            && item.proto.is_none()
        {
            item.proto = decode(&item.bytes)
                .ok()
                .and_then(|img| self.picker.new_protocol(img, cells, Resize::Fit(None)).ok());
        }
    }
}

async fn run_inner() -> Result<()> {
    let backend = CrosstermBackend::new(stdout());
    let mut term = Terminal::new(backend)?;
    let size = term.size()?;
    let full = Rect::new(0, 0, size.width, size.height);

    let picker = Picker::from_query_stdio().unwrap_or_else(|_| Picker::halfblocks());
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
        chat: vec![("sys".into(), "ccti ready. Type text for the agent, or /render <prompt> for a direct render. /help lists commands.".into())],
        input: String::new(),
        status: "starting…".into(),
        gallery: Vec::new(),
        gidx: 0,
        picker,
        busy: false,
        agent_offline: false,
        cells: gallery_cells(full),
        settings: RenderSettings::default(),
    };
    let mut events = EventStream::new();

    loop {
        term.draw(|f| draw(f, &mut app))?;
        tokio::select! {
            msg = ui_rx.recv() => {
                let Some(msg) = msg else { break };
                match msg {
                    UiMsg::Chat { role, text } => {
                        for chunk in split_long(&text) {
                            app.chat.push((role.clone(), chunk));
                        }
                    }
                    UiMsg::Status(s) => app.status = s,
                    UiMsg::Image { label, bytes, generation } => {
                        app.push_image(label.clone(), bytes, generation);
                        app.chat.push(("sys".into(), format!("image: {label}")));
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
                        app.chat.push(("err".into(), e.clone()));
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
                        match k.code {
                            KeyCode::Enter => {
                                let line = std::mem::take(&mut app.input);
                                if line.trim().is_empty() { continue; }
                                if !handle_command(&line, &app, &agent, &comfy, &ui_tx) {
                                    break;
                                }
                            }
                            KeyCode::Char(c) => {
                                // Settings keys act only on an empty input
                                // line, so typing prompts never misfires.
                                if app.input.is_empty()
                                    && let Some(msg) = app.settings.apply_key(c)
                                {
                                    app.status = msg;
                                } else {
                                    app.input.push(c);
                                }
                            }
                            KeyCode::Tab if app.input.is_empty() => {
                                app.settings.cycle_count();
                                app.status = format!("count → n={}", app.settings.n);
                            }
                            KeyCode::Backspace => { app.input.pop(); }
                            KeyCode::Esc => {
                                if app.busy { agent.cancel(); }
                                else { app.input.clear(); }
                            }
                            KeyCode::Left if app.input.is_empty() => app.step_history(-1),
                            KeyCode::Right if app.input.is_empty() => app.step_history(1),
                            _ => {}
                        }
                    }
                    Event::Resize(_, _) => {
                        let s = term.size()?;
                        app.cells = gallery_cells(Rect::new(0, 0, s.width, s.height));
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

fn draw(f: &mut ratatui::Frame<'_>, app: &mut App) {
    let a = layout_areas(f.area());
    let title = if app.gallery.is_empty() {
        "image — nothing rendered yet".to_string()
    } else {
        let item = &app.gallery[app.gidx];
        format!(
            "image {}/{} — {} (←/→)",
            app.gidx + 1,
            app.gallery.len(),
            item.label
        )
    };
    f.render_widget(pic_block(title), a.pic);
    if let Some(item) = app.gallery.get(app.gidx)
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
    let prov = prov_lines(
        app.gallery
            .get(app.gidx)
            .and_then(|i| i.generation.as_ref()),
    );
    f.render_widget(
        Paragraph::new(prov.join("\n"))
            .block(prov_block())
            .wrap(Wrap { trim: true }),
        a.prov,
    );
    f.render_widget(
        Paragraph::new(app.settings.lines().join("\n")).block(settings_block()),
        a.settings,
    );

    // Chat pane (right, ~20).
    let w = a.chat_msgs.width.saturating_sub(2).max(10) as usize;
    let mut lines: Vec<RLine> = Vec::new();
    for (role, text) in &app.chat {
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

    let status = Paragraph::new(app.status.clone()).style(Style::default().fg(Color::DarkGray));
    f.render_widget(status, a.status);
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
            agent.prompt(line.to_string());
            let _ = tx.send(UiMsg::Chat {
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
            let _ = tx.send(UiMsg::Chat { role: "sys".into(), text:
                "/render TEXT [--w N --h N --steps N --n 1-4] direct render (fast defaults: 512px, 8 steps) · /models list checkpoints · /cancel stop agent turn · ←/→ image history · /quit".into() });
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
                        let _ = tx.send(UiMsg::Chat {
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
            let (prompt, w, h, steps, n) = args.resolve(&app.settings);
            let tx2 = tx.clone();
            let c = comfy.clone();
            let prompt2 = prompt.clone();
            tokio::spawn(async move {
                direct_render(&c, &tx2, &prompt2, w, h, steps, n).await;
            });
            let _ = tx.send(UiMsg::Chat {
                role: "you".into(),
                text: format!("/render {prompt}"),
            });
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

impl RenderArgs {
    fn resolve(&self, settings: &RenderSettings) -> (String, u32, u32, u32, u32) {
        let (dw, dh) = settings.dims();
        (
            self.prompt.clone(),
            self.w.unwrap_or(dw),
            self.h.unwrap_or(dh),
            self.steps.unwrap_or(settings.steps),
            self.n.unwrap_or(settings.n),
        )
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

async fn direct_render(
    c: &Client,
    tx: &mpsc::UnboundedSender<UiMsg>,
    prompt: &str,
    w: u32,
    h: u32,
    steps: u32,
    n: u32,
) {
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
                prompt,
                ckpt: comfy::DEFAULT_CKPT,
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
            prompt: prompt.to_string(),
            negative: "blurry, watermark, text, deformed".to_string(),
            ckpt: comfy::DEFAULT_CKPT.to_string(),
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

    fn test_app() -> App {
        App {
            chat: Vec::new(),
            input: String::new(),
            status: String::new(),
            gallery: Vec::new(),
            gidx: 0,
            picker: Picker::halfblocks(),
            busy: false,
            agent_offline: false,
            cells: Size::new(80, 24),
            settings: RenderSettings::default(),
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
        let (img, chat) = split(Rect::new(0, 0, 120, 40));
        // 119 cells for panes + 1 gap cell.
        assert_eq!(img.width + chat.width, 119);
        assert_eq!(img.height, chat.height);
        assert_eq!(chat.x, img.x + img.width + 1);
    }

    #[test]
    fn narrow_screens_stack_image_over_chat() {
        let (img, chat) = split(Rect::new(0, 0, 80, 40));
        assert_eq!(img.width, 80);
        // 39 cells for panes + 1 gap cell.
        assert_eq!(img.height + chat.height, 39);
        assert_eq!(chat.y, img.y + img.height + 1);
    }

    #[test]
    fn history_walks_and_wraps() {
        let mut app = test_app();
        app.step_history(1); // empty: no panic, no move
        assert_eq!(app.gidx, 0);
        app.push_image("a".into(), test_png(), None);
        app.push_image("b".into(), test_png(), None);
        assert_eq!(app.gidx, 1);
        app.step_history(1);
        assert_eq!(app.gidx, 0);
        app.step_history(-1);
        assert_eq!(app.gidx, 1);
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
        assert_eq!(
            parse_render("a fox").resolve(&settings),
            ("a fox".to_string(), 512, 512, 8, 1)
        );
        assert_eq!(
            parse_render("a fox --w 1024 --n 2").resolve(&settings),
            ("a fox".to_string(), 1024, 512, 8, 2)
        );
        let mut quality = RenderSettings::default();
        quality.cycle_preset(2); // Quality 1024, 20 steps
        assert_eq!(
            parse_render("a fox").resolve(&quality),
            ("a fox".to_string(), 1024, 1024, 20, 1)
        );
    }

    #[test]
    fn settings_cycle_and_clamp() {
        let mut s = RenderSettings::default();
        assert_eq!(s.describe(), "Fast 512x512 · 8 steps · n=1");
        s.cycle_preset(1);
        assert_eq!(s.name(), "Balanced");
        s.cycle_preset(1);
        assert_eq!(s.name(), "Quality");
        s.cycle_preset(1);
        assert_eq!(s.name(), "Fast"); // wraps
        s.adjust_steps(1000);
        assert_eq!(s.steps, 50);
        s.adjust_steps(-1000);
        assert_eq!(s.steps, 4);
        s.cycle_count();
        s.cycle_count();
        assert_eq!(s.n, 3);
        assert_eq!(
            s.apply_key(']'),
            Some("preset → Balanced 768x768 · 14 steps · n=3".to_string())
        );
        assert_eq!(s.apply_key('x'), None);
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
        app.chat.push((
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
        app.push_image("test".into(), test_png(), Some(generation));
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
        // Horizontal seam between picture and provenance is a single line:
        // recompute the layout and check the provenance top row has no ─ run.
        let full = Rect::new(0, 0, 120, 39);
        let (img_rect, _) = split(full);
        let left = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Min(0), Constraint::Length(8)])
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
        app.push_image("test".into(), test_png(), None);
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
