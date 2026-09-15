//! Agent side: one ccht native ACP session per workspace, created lazily
//! on first use. Turns in different workspaces run in parallel
//! automatically; each workspace serializes its own turns, and every chat
//! line is tagged so late answers still land where they were asked.
//!
//! ccht owns session mechanics; ccti owns UI, tools (via bundled MCP
//! server) and the permission policy below.

use std::{collections::HashMap, path::PathBuf};

use ccht::{
    Conversation, Event, Prompt, SessionEvent, WireEvent,
    acp::{
        ContentBlock, McpServer, McpServerStdio, PermissionOptionKind, RequestPermissionOutcome,
        SelectedPermissionOutcome, SessionConfigSelectOptions, SessionUpdate,
    },
    native::{
        AgentCommand, NativeClient, NativeOptions, PermissionPolicy, SessionHandle, SessionOptions,
    },
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

/// Human-readable current model of a session configuration: the option
/// label when resolvable, else the raw value id. Pure: unit-tested below.
pub fn session_model_name(cfg: &ccht::SessionConfiguration) -> Option<String> {
    let opt = cfg.model_option()?;
    let ccht::acp::SessionConfigKind::Select(sel) = &opt.kind else {
        return None;
    };
    let current = sel.current_value.to_string();
    let label = match &sel.options {
        SessionConfigSelectOptions::Ungrouped(opts) => opts
            .iter()
            .find(|o| o.value.to_string() == current)
            .map(|o| o.name.clone()),
        SessionConfigSelectOptions::Grouped(groups) => groups
            .iter()
            .flat_map(|g| &g.options)
            .find(|o| o.value.to_string() == current)
            .map(|o| o.name.clone()),
        _ => None,
    };
    Some(label.unwrap_or(current))
}

/// Default chat model for new workspace sessions: vision-capable, so the
/// agent can actually look at its renders instead of guessing from prompts.
pub const DEFAULT_CHAT_MODEL: &str = "opencode-go/muse-spark-1.3-contributor";

enum Cmd {
    Prompt { ws: usize, text: String },
    SetChatModel { ws: usize, model: String },
    Cancel { ws: usize },
}

pub struct Agent {
    tx: mpsc::UnboundedSender<Cmd>,
}

impl Agent {
    /// Connect the shared client and spawn the router. Sessions are created
    /// lazily per workspace on first use.
    pub async fn spawn(
        exe_mcp: PathBuf,
        cwd: PathBuf,
        ui: mpsc::UnboundedSender<UiMsg>,
    ) -> anyhow::Result<Self> {
        let (tx, rx) = mpsc::unbounded_channel();
        let shared = Shared::connect(exe_mcp, cwd, ui.clone()).await?;
        tokio::spawn(run_router(shared, rx, ui));
        Ok(Self { tx })
    }

    pub fn prompt(&self, text: String, ws: usize) {
        let _ = self.tx.send(Cmd::Prompt { ws, text });
    }

    pub fn set_chat_model(&self, ws: usize, model: String) {
        let _ = self.tx.send(Cmd::SetChatModel { ws, model });
    }

    pub fn cancel(&self, ws: usize) {
        let _ = self.tx.send(Cmd::Cancel { ws });
    }
}

struct Shared {
    client: Option<NativeClient>,
    exe_mcp: PathBuf,
    cwd: PathBuf,
    ui: mpsc::UnboundedSender<UiMsg>,
}

impl Shared {
    async fn connect(
        exe_mcp: PathBuf,
        cwd: PathBuf,
        ui: mpsc::UnboundedSender<UiMsg>,
    ) -> anyhow::Result<Self> {
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
            Ok(c) => {
                send(UiMsg::Status("agent connected".into()));
                Some(c)
            }
            Err(e) => {
                send(UiMsg::Error(format!(
                    "agent offline ({e}); /render still works"
                )));
                None
            }
        };
        Ok(Self {
            client,
            exe_mcp,
            cwd,
            ui,
        })
    }
}

struct Session {
    handle: SessionHandle,
    conv: Conversation,
    buf: String,
    busy: bool,
    first: bool,
    seq: u64,
    req_no: u64,
}

async fn run_router(
    shared: Shared,
    mut cmds: mpsc::UnboundedReceiver<Cmd>,
    ui: mpsc::UnboundedSender<UiMsg>,
) {
    let send = |m: UiMsg| {
        let _ = ui.send(m);
    };
    let (ev_tx, mut ev_rx) = mpsc::unbounded_channel::<(usize, Option<SessionEvent>)>();
    let mut sessions: HashMap<usize, Session> = HashMap::new();

    // Ensure a live session for ws, creating it (plus its event forwarder)
    // on first use. Returns false when the backend is offline.
    async fn ensure(
        shared: &Shared,
        sessions: &mut HashMap<usize, Session>,
        ev_tx: &mpsc::UnboundedSender<(usize, Option<SessionEvent>)>,
        ws: usize,
    ) -> bool {
        if sessions.contains_key(&ws) {
            return true;
        }
        let Some(client) = &shared.client else {
            shared
                .ui
                .send(UiMsg::Error(format!(
                    "agent offline (ws{}); /render still works",
                    ws + 1
                )))
                .ok();
            return false;
        };
        let mut mcp = McpServerStdio::new("ccti-comfy", shared.exe_mcp.clone());
        mcp.args = vec!["--mcp".to_string()];
        let mut session = match client
            .new_session(SessionOptions {
                cwd: shared.cwd.clone(),
                model: Some(DEFAULT_CHAT_MODEL.to_string()),
                configuration: Vec::new(),
                mcp_servers: vec![McpServer::Stdio(mcp)],
            })
            .await
        {
            Ok(s) => s,
            Err(e) => {
                shared
                    .ui
                    .send(UiMsg::Error(format!("agent session failed: {e}")))
                    .ok();
                return false;
            }
        };
        let handle = session.handle();
        if let Some(model) = session_model_name(&handle.configuration()) {
            shared.ui.send(UiMsg::SessionModel { ws, model }).ok();
        }
        let mut events = match session.take_events() {
            Ok(rx) => rx,
            Err(e) => {
                shared
                    .ui
                    .send(UiMsg::Error(format!("agent events failed: {e}")))
                    .ok();
                return false;
            }
        };
        drop(session);
        let fwd = ev_tx.clone();
        tokio::spawn(async move {
            while let Some(ev) = events.recv().await {
                if fwd.send((ws, Some(ev))).is_err() {
                    break;
                }
            }
            let _ = fwd.send((ws, None));
        });
        sessions.insert(
            ws,
            Session {
                handle,
                conv: Conversation::new(format!("ccti-ws{ws}")),
                buf: String::new(),
                busy: false,
                first: true,
                seq: 0,
                req_no: 0,
            },
        );
        true
    }

    let any_busy = |sessions: &HashMap<usize, Session>| sessions.values().any(|s| s.busy);

    loop {
        tokio::select! {
            cmd = cmds.recv() => {
                let Some(cmd) = cmd else { break };
                match cmd {
                    Cmd::Prompt { ws, text } => {
                        if !ensure(&shared, &mut sessions, &ev_tx, ws).await {
                            continue;
                        }
                        let Some(session) = sessions.get_mut(&ws) else {
                            continue;
                        };
                        if session.busy {
                            send(UiMsg::Error(format!("ws{} is busy (/cancel first)", ws + 1)));
                            continue;
                        }
                        session.busy = true;
                        session.req_no += 1;
                        let request_id = format!("ccti-{}-{}", ws, session.req_no);
                        let full = if session.first {
                            session.first = false;
                            format!("{SYSTEM}\n\n{text}")
                        } else { text };
                        let h = session.handle.clone();
                        let ui2 = ui.clone();
                        let rid = request_id.clone();
                        tokio::spawn(async move {
                            match h.prompt(Prompt::text(&rid, full)).await {
                                Ok(_) => { let _ = ui2.send(UiMsg::AgentDone); }
                                Err(e) => { let _ = ui2.send(UiMsg::Error(format!("prompt failed: {e}"))); }
                            }
                        });
                        send(UiMsg::AgentBusy { ws, busy: true });
                    }
                    Cmd::SetChatModel { ws, model } => {
                        if !ensure(&shared, &mut sessions, &ev_tx, ws).await {
                            continue;
                        }
                        let h = match sessions.get(&ws) {
                            Some(s) => s.handle.clone(),
                            None => continue,
                        };
                        let ui2 = ui.clone();
                        tokio::spawn(async move {
                            match h.set_model(&model).await {
                                Ok(cfg) => {
                                    let shown = session_model_name(&cfg).unwrap_or(model.clone());
                                    let _ = ui2.send(UiMsg::ChatModel { ws, model: shown });
                                }
                                Err(e) => {
                                    let _ = ui2.send(UiMsg::Error(format!("chat model switch failed: {e}")));
                                }
                            }
                        });
                    }
                    Cmd::Cancel { ws } => {
                        if let Some(session) = sessions.get(&ws) {
                            let h = session.handle.clone();
                            let ui2 = ui.clone();
                            tokio::spawn(async move {
                                if let Err(e) = h.cancel().await {
                                    let _ = ui2.send(UiMsg::Error(format!("cancel failed: {e}")));
                                }
                            });
                        }
                    }
                }
            }
            msg = ev_rx.recv() => {
                let Some((ws, ev)) = msg else { break };
                let Some(session) = sessions.get_mut(&ws) else { continue };
                let Some(se) = ev else {
                    // Forwarder ended: session stream closed.
                    session.busy = false;
                    send(UiMsg::AgentBusy { ws, busy: false });
                    send(UiMsg::Error(format!("agent stream closed (ws{})", ws + 1)));
                    continue;
                };
                session.seq += 1;
                let rid = se.request_id.clone().unwrap_or_default();
                let SessionEvent { event, .. } = &se;
                let conv_id = session.conv.id().to_string();
                let _ = session.conv.apply(WireEvent::new(conv_id, rid.clone(), session.seq, event.clone()));
                match event {
                    Event::Update { update } => match update {
                        SessionUpdate::AgentMessageChunk(chunk) => {
                            if let ContentBlock::Text(t) = &chunk.content {
                                session.buf.push_str(&t.text);
                            }
                        }
                        SessionUpdate::AgentThoughtChunk(_) => {}
                        SessionUpdate::ToolCall(tc) => {
                            send(UiMsg::Chat { ws: Some(ws), role: "tool".into(), text: format!("{} …", tc.title) });
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
                        let h = session.handle.clone();
                        let ui2 = ui.clone();
                        let request_id = request_id.clone();
                        tokio::spawn(async move {
                            if h.respond_permission(&request_id, decision).await.is_err() {
                                let _ = ui2.send(UiMsg::Error("permission response failed".into()));
                            } else {
                                let _ = ui2.send(UiMsg::Chat { ws: Some(ws), role: "sys".into(), text: format!("permission {verdict}: {title}") });
                            }
                        });
                    }
                    Event::Completed { .. } => {
                        if !session.buf.trim().is_empty() {
                            let text = std::mem::take(&mut session.buf);
                            send(UiMsg::Chat { ws: Some(ws), role: "agent".into(), text });
                        }
                        session.busy = false;
                        send(UiMsg::AgentBusy { ws, busy: any_busy(&sessions) });
                    }
                    Event::Error { code, message } => {
                        session.busy = false;
                        send(UiMsg::AgentBusy { ws, busy: any_busy(&sessions) });
                        send(UiMsg::Error(format!("{code}: {message}")));
                    }
                }
            }
        }
    }
    if let Some(client) = shared.client {
        let _ = client.close().await;
    }
}

const SYSTEM: &str = "You are the ccti image assistant. You have ComfyUI tools: \
render_image (waits for the whole render), render_submit (starts a background \
render and returns a render_id immediately), render_status (cheap poll) and \
render_result (paths plus preview when done). Prefer submit/status/result: \
fresh renders include GPU cold starts of several minutes that exceed a single \
tool-call timeout. Defaults are tuned for fast iteration (512x512, 8 steps, \
1 image); use n (up to 4) for variants and larger sizes/steps for finals. \
Every render saves PNGs (with embedded provenance plus JSON sidecars) and \
returns their paths plus a preview image. If attached images are not visible \
to you, Read the saved PNG file(s) to view them. Always look at the result \
(preview or file), describe it briefly, and offer tweaks. Answer in the \
user's language.";

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    fn config_with_model(value: &str, label: &str) -> ccht::SessionConfiguration {
        serde_json::from_value(json!({
            "options": [{
                "id": "model", "name": "Model", "category": "model", "type": "select",
                "currentValue": value,
                "options": [{"value": value, "name": label}]
            }],
            "modes": null
        }))
        .expect("test config")
    }

    #[test]
    fn session_model_prefers_label_over_value() {
        let cfg = config_with_model("muse-spark", "Muse Spark 1.3");
        assert_eq!(session_model_name(&cfg), Some("Muse Spark 1.3".to_string()));
    }

    #[test]
    fn session_model_falls_back_to_raw_value() {
        let cfg: ccht::SessionConfiguration = serde_json::from_value(json!({
            "options": [{
                "id": "model", "name": "Model", "category": "model", "type": "select",
                "currentValue": "mystery-9",
                "options": [{"value": "other-1", "name": "Other"}]
            }],
            "modes": null
        }))
        .expect("test config");
        assert_eq!(session_model_name(&cfg), Some("mystery-9".to_string()));
    }

    #[test]
    fn session_model_absent_without_model_option() {
        let cfg: ccht::SessionConfiguration = serde_json::from_value(json!({
            "options": [{
                "id": "thinking", "name": "Thinking", "type": "boolean",
                "currentValue": false
            }],
            "modes": null
        }))
        .expect("test config");
        assert_eq!(session_model_name(&cfg), None);
    }
}
