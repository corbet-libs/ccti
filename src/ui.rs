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
use ratatui_image::{Image, Resize, picker::Picker, protocol::Protocol};
use tokio::sync::mpsc;

use crate::agent::Agent;
use crate::comfy::{self, Client};

/// Messages into the UI loop (agent task, render tasks, watcher).
#[derive(Debug)]
pub enum UiMsg {
    Chat { role: String, text: String },
    Status(String),
    Image { label: String, bytes: Vec<u8> },
    AgentBusy(bool),
    AgentDone,
    Error(String),
}

struct GalleryItem {
    label: String,
    bytes: Vec<u8>,
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
}

pub async fn run() -> Result<()> {
    enable_raw_mode().map_err(|_| anyhow::anyhow!("ccti needs a real terminal"))?;
    execute!(stdout(), EnterAlternateScreen)?;
    let res = run_inner().await;
    let _ = disable_raw_mode();
    let _ = execute!(stdout(), LeaveAlternateScreen);
    res
}

/// Split main area into (image, chat). Wide screens go 80/20 side by side,
/// narrow ones stack image over chat.
fn split(area: Rect) -> (Rect, Rect) {
    if area.width >= 100 {
        let cols = Layout::default()
            .direction(Direction::Horizontal)
            .constraints([Constraint::Percentage(80), Constraint::Percentage(20)])
            .split(area);
        (cols[0], cols[1])
    } else {
        let rows = Layout::default()
            .direction(Direction::Vertical)
            .constraints([Constraint::Percentage(60), Constraint::Percentage(40)])
            .split(area);
        (rows[0], rows[1])
    }
}

fn gallery_cells(area: Rect) -> Size {
    let (img, _) = split(area);
    Size::new(img.width.saturating_sub(2), img.height.saturating_sub(2))
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

    fn push_image(&mut self, label: String, bytes: Vec<u8>) {
        self.gallery.push(GalleryItem {
            label,
            bytes,
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
                    UiMsg::Image { label, bytes } => {
                        app.push_image(label.clone(), bytes);
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
                            KeyCode::Char(c) => app.input.push(c),
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

fn split_long(text: &str) -> Vec<String> {
    text.lines()
        .map(|l| {
            if l.len() <= 2000 {
                l.to_string()
            } else {
                format!("{}…", &l[..2000])
            }
        })
        .collect()
}

fn draw(f: &mut ratatui::Frame<'_>, app: &mut App) {
    let area = f.area();
    let rows = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(1)])
        .split(area);
    let (img_rect, chat_rect) = split(rows[0]);

    // Image pane (left, ~80).
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
    f.render_widget(
        Block::default().borders(Borders::ALL).title(title),
        img_rect,
    );
    if let Some(item) = app.gallery.get(app.gidx)
        && let Some(proto) = &item.proto
    {
        let inner = Rect {
            x: img_rect.x + 1,
            y: img_rect.y + 1,
            width: img_rect.width.saturating_sub(2),
            height: img_rect.height.saturating_sub(2),
        };
        f.render_widget(Image::new(proto), inner);
    }

    // Chat pane (right, ~20).
    let cols = Layout::default()
        .direction(Direction::Vertical)
        .constraints([Constraint::Min(0), Constraint::Length(3)])
        .split(chat_rect);
    let w = chat_rect.width.saturating_sub(2).max(10) as usize;
    let mut lines: Vec<RLine> = Vec::new();
    for (role, text) in &app.chat {
        let color = match role.as_str() {
            "you" => Color::Green,
            "agent" => Color::Cyan,
            "tool" => Color::Yellow,
            "err" => Color::Red,
            _ => Color::DarkGray,
        };
        for (i, chunk) in text.chars().collect::<Vec<_>>().chunks(w).enumerate() {
            let s: String = chunk.iter().collect();
            if i == 0 {
                lines.push(RLine::from(vec![
                    Span::styled(format!("[{role}] "), Style::default().fg(color)),
                    Span::raw(s),
                ]));
            } else {
                lines.push(RLine::from(format!("      {s}")));
            }
        }
    }
    let visible = cols[0].height.saturating_sub(2) as usize;
    let skip = lines.len().saturating_sub(visible.max(1));
    let msg = Paragraph::new(lines.into_iter().skip(skip).collect::<Vec<_>>())
        .block(Block::default().borders(Borders::ALL).title("chat"))
        .wrap(Wrap { trim: false });
    f.render_widget(msg, cols[0]);
    let prompt = if app.busy {
        "…working (Esc cancels)"
    } else {
        ">"
    };
    let input = Paragraph::new(format!("{prompt} {}", app.input))
        .block(Block::default().borders(Borders::ALL));
    f.render_widget(input, cols[1]);

    let status = Paragraph::new(app.status.clone()).style(Style::default().fg(Color::DarkGray));
    f.render_widget(status, rows[1]);
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
                "/render TEXT [--w N --h N --steps N] direct render · /models list checkpoints · /cancel stop agent turn · ←/→ image history · /quit".into() });
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
            let (prompt, w, h, steps) = parse_render(&rest);
            if prompt.is_empty() {
                let _ = tx.send(UiMsg::Error(
                    "usage: /render TEXT [--w N --h N --steps N]".into(),
                ));
                return true;
            }
            let tx2 = tx.clone();
            let c = comfy.clone();
            let prompt2 = prompt.clone();
            tokio::spawn(async move {
                direct_render(&c, &tx2, &prompt2, w, h, steps).await;
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

fn parse_render(rest: &str) -> (String, u32, u32, u32) {
    let (mut w, mut h, mut steps) = (768_u32, 768_u32, 10_u32);
    let mut words: Vec<&str> = Vec::new();
    let mut it = rest.split_whitespace().peekable();
    while let Some(tok) = it.next() {
        match tok {
            "--w" => w = it.next().and_then(|v| v.parse().ok()).unwrap_or(w),
            "--h" => h = it.next().and_then(|v| v.parse().ok()).unwrap_or(h),
            "--steps" => steps = it.next().and_then(|v| v.parse().ok()).unwrap_or(steps),
            _ => words.push(tok),
        }
    }
    (words.join(" "), w, h, steps)
}

async fn direct_render(
    c: &Client,
    tx: &mpsc::UnboundedSender<UiMsg>,
    prompt: &str,
    w: u32,
    h: u32,
    steps: u32,
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
    let img = match c
        .render(
            comfy::RenderOpts {
                prompt,
                ckpt: comfy::DEFAULT_CKPT,
                width: w,
                height: h,
                steps,
                seed,
            },
            |p| send(UiMsg::Status(p)),
        )
        .await
    {
        Ok(i) => i,
        Err(e) => {
            send(UiMsg::Error(format!("render: {e:#}")));
            return;
        }
    };
    match c.download(&img).await {
        Ok(bytes) => send(UiMsg::Image {
            label: format!("{prompt} ({seed})"),
            bytes,
        }),
        Err(e) => send(UiMsg::Error(format!("download: {e:#}"))),
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
                    let _ = tx.send(UiMsg::Image { label: name, bytes });
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
        assert_eq!(img.width, 96);
        assert_eq!(chat.width, 24);
        assert_eq!(img.height, chat.height);
    }

    #[test]
    fn narrow_screens_stack_image_over_chat() {
        let (img, chat) = split(Rect::new(0, 0, 80, 40));
        assert_eq!(img.width, 80);
        assert_eq!(img.height, 24);
        assert_eq!(chat.y, 24);
    }

    #[test]
    fn history_walks_and_wraps() {
        let mut app = test_app();
        app.step_history(1); // empty: no panic, no move
        assert_eq!(app.gidx, 0);
        app.push_image("a".into(), test_png());
        app.push_image("b".into(), test_png());
        assert_eq!(app.gidx, 1);
        app.step_history(1);
        assert_eq!(app.gidx, 0);
        app.step_history(-1);
        assert_eq!(app.gidx, 1);
    }

    #[test]
    fn parse_render_plain_prompt_keeps_defaults() {
        assert_eq!(parse_render("a fox"), ("a fox".to_string(), 768, 768, 10));
    }

    #[test]
    fn parse_render_honours_flags() {
        assert_eq!(
            parse_render("a fox --w 1024 --h 512 --steps 20"),
            ("a fox".to_string(), 1024, 512, 20)
        );
    }

    #[test]
    fn parse_render_ignores_broken_flag_values() {
        assert_eq!(
            parse_render("x --steps abc"),
            ("x".to_string(), 768, 768, 10)
        );
        assert_eq!(parse_render("--w 100"), ("".to_string(), 100, 768, 10));
    }
}
