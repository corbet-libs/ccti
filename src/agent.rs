//! Agent side: ccht native ACP session against `opencode acp`.
//! ccht owns session mechanics; ccti owns UI, tools (via bundled MCP
//! server) and the permission policy below.

use std::path::PathBuf;

use ccht::{
    Conversation, Event, Prompt, SessionEvent, WireEvent,
    acp::{
        ContentBlock, McpServer, McpServerStdio, PermissionOptionKind, RequestPermissionOutcome,
        SelectedPermissionOutcome, SessionUpdate,
    },
    native::{AgentCommand, NativeClient, NativeOptions, PermissionPolicy, SessionOptions},
};
use tokio::sync::mpsc;

use crate::ui::UiMsg;

/// MVP permission policy: the turn stays fully automatic. Every permission
/// request is granted once and announced in chat, so the user watches what
/// the agent does and can /cancel the turn. Revisit before any untrusted use.
fn decide(request: &ccht::acp::RequestPermissionRequest) -> RequestPermissionOutcome {
    for opt in &request.options {
        if opt.kind == PermissionOptionKind::AllowOnce {
            return RequestPermissionOutcome::Selected(SelectedPermissionOutcome::new(
                opt.option_id.clone(),
            ));
        }
    }
    RequestPermissionOutcome::Cancelled
}

const SYSTEM: &str = "You are the ccti image assistant. You have an MCP tool \
render_image(prompt, width, height, steps, n, seed) that renders with ComfyUI. \
Defaults are tuned for fast iteration (512x512, 8 steps, 1 image); use n (up \
to 4) for variants and larger sizes/steps for finals. Every render saves PNGs \
(with embedded provenance plus JSON sidecars) and returns their paths plus a \
preview image. If attached images are not visible to you, Read the saved PNG \
file(s) to view them. Always look at the result (preview or file), describe \
it briefly, and offer tweaks. The cold start is handled inside the tool; \
just wait for it. Answer in the user's language.";

enum Cmd {
    Prompt(String),
    Cancel,
}

pub struct Agent {
    tx: mpsc::UnboundedSender<Cmd>,
}

impl Agent {
    /// Spawn the background session task. `exe_mcp` is this binary (for --mcp).
    pub async fn spawn(
        exe_mcp: PathBuf,
        cwd: PathBuf,
        ui: mpsc::UnboundedSender<UiMsg>,
    ) -> anyhow::Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        tokio::spawn(run_session(exe_mcp, cwd, rx, ui));
        Ok(Self { tx })
    }

    pub fn prompt(&self, text: String) {
        let _ = self.tx.send(Cmd::Prompt(text));
    }

    pub fn cancel(&self) {
        let _ = self.tx.send(Cmd::Cancel);
    }
}

async fn run_session(
    exe_mcp: PathBuf,
    cwd: PathBuf,
    mut cmds: mpsc::UnboundedReceiver<Cmd>,
    ui: mpsc::UnboundedSender<UiMsg>,
) {
    let send = |m: UiMsg| {
        let _ = ui.send(m);
    };
    let client = match NativeClient::connect(
        AgentCommand::new("opencode").args(["acp"]),
        NativeOptions {
            prompt_timeout: std::time::Duration::from_secs(1800),
            permissions: PermissionPolicy::Ask,
            ..NativeOptions::default()
        },
    )
    .await
    {
        Ok(c) => c,
        Err(e) => {
            send(UiMsg::Error(format!(
                "agent offline ({e}); /render still works"
            )));
            return;
        }
    };
    let mut mcp = McpServerStdio::new("ccti-comfy", exe_mcp);
    mcp.args = vec!["--mcp".to_string()];
    let mcp = McpServer::Stdio(mcp);
    let mut session = match client
        .new_session(SessionOptions {
            cwd,
            model: None,
            configuration: Vec::new(),
            mcp_servers: vec![mcp],
        })
        .await
    {
        Ok(s) => s,
        Err(e) => {
            send(UiMsg::Error(format!("agent session failed: {e}")));
            return;
        }
    };
    let handle = session.handle();
    let mut events = match session.take_events() {
        Ok(rx) => rx,
        Err(e) => {
            send(UiMsg::Error(format!("agent events failed: {e}")));
            return;
        }
    };
    let mut conv = Conversation::new(handle.id().to_string());
    let mut seq = 0_u64;
    let mut req_no = 0_u64;
    let mut busy = false;
    let mut first = true;
    let mut buf = String::new();
    send(UiMsg::Status("agent connected".into()));

    loop {
        tokio::select! {
            cmd = cmds.recv() => {
                let Some(cmd) = cmd else { break };
                match cmd {
                    Cmd::Prompt(text) => {
                        if busy {
                            send(UiMsg::Error("agent is busy (/cancel first)".into()));
                            continue;
                        }
                        busy = true;
                        req_no += 1;
                        let request_id = format!("ccti-{req_no}");
                        let full = if first {
                            first = false;
                            format!("{SYSTEM}\n\n{text}")
                        } else { text };
                        let h = handle.clone();
                        let ui2 = ui.clone();
                        let rid = request_id.clone();
                        tokio::spawn(async move {
                            match h.prompt(Prompt::text(&rid, full)).await {
                                Ok(_) => { let _ = ui2.send(UiMsg::AgentDone); }
                                Err(e) => { let _ = ui2.send(UiMsg::Error(format!("prompt failed: {e}"))); }
                            }
                        });
                        send(UiMsg::AgentBusy(true));
                    }
                    Cmd::Cancel => {
                        if let Err(e) = handle.cancel().await {
                            send(UiMsg::Error(format!("cancel failed: {e}")));
                        }
                    }
                }
            }
            ev = events.recv() => {
                let Some(se) = ev else {
                    send(UiMsg::Error("agent stream closed".into()));
                    break;
                };
                seq += 1;
                let rid = se.request_id.clone().unwrap_or_default();
                let SessionEvent { event, .. } = &se;
                let _ = conv.apply(WireEvent::new(conv.id().to_string(), rid.clone(), seq, event.clone()));
                match event {
                    Event::Update { update } => match update {
                        SessionUpdate::AgentMessageChunk(chunk) => {
                            if let ContentBlock::Text(t) = &chunk.content {
                                buf.push_str(&t.text);
                            }
                        }
                        SessionUpdate::AgentThoughtChunk(_) => {}
                        SessionUpdate::ToolCall(tc) => {
                            send(UiMsg::Chat { role: "tool".into(), text: format!("{} …", tc.title) });
                        }
                        SessionUpdate::ToolCallUpdate(u) => {
                            if let Some(title) = u.fields.title.as_deref() {
                                let st = u.fields.status.map(|s| format!("{s:?}")).unwrap_or_default();
                                send(UiMsg::Status(format!("{title} {st}")));
                            }
                        }
                        _ => {}
                    },
                    Event::Permission { request_id, request } => {
                        let title = request.tool_call.fields.title.clone().unwrap_or_default();
                        let decision = decide(request);
                        let verdict = match &decision {
                            RequestPermissionOutcome::Selected(_) => "auto-allowed",
                            _ => "denied",
                        };
                        if handle.respond_permission(request_id, decision).await.is_err() {
                            send(UiMsg::Error("permission response failed".into()));
                        } else {
                            send(UiMsg::Chat { role: "sys".into(), text: format!("permission {verdict}: {title}") });
                        }
                    }
                    Event::Completed { .. } => {
                        if !buf.trim().is_empty() {
                            send(UiMsg::Chat { role: "agent".into(), text: std::mem::take(&mut buf) });
                        }
                        busy = false;
                        send(UiMsg::AgentBusy(false));
                    }
                    Event::Error { code, message } => {
                        busy = false;
                        send(UiMsg::AgentBusy(false));
                        send(UiMsg::Error(format!("{code}: {message}")));
                    }
                }
            }
        }
    }
    let _ = client.close().await;
}
