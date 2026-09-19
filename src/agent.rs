//! Agent side: chat turns through ccht's keyed [`SessionPool`].
//!
//! Pool keys are workspace ids (`ws{index}`); each key owns exactly one
//! session, created lazily on first use. Turns on different keys run in
//! parallel, each key serializes its own turns, and every event arrives
//! tagged so late answers still land where they were asked.
//!
//! ccht owns session mechanics (per-turn request identity, busy guards,
//! conversation snapshots); ccti owns the UI, the first-turn prompt,
//! tool wiring (bundled `--mcp` server per session) and the permission
//! policy below.

use std::{collections::HashSet, path::PathBuf, time::Duration};

use ccht::{
    Event,
    acp::{
        McpServer, McpServerStdio, PermissionOptionKind, RequestPermissionOutcome,
        SelectedPermissionOutcome, SessionUpdate,
    },
    native::{
        AgentCommand, NativeError, NativeOptions, PermissionPolicy, PoolEvent, SessionOptions,
        SessionPool,
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

/// Default chat model for new workspace sessions: vision-capable, so the
/// agent can actually look at its renders instead of guessing from prompts.
pub const DEFAULT_CHAT_MODEL: &str = "opencode-go/muse-spark-1.3-contributor";

/// Pool key for a workspace index. Pure: unit-tested below.
fn pool_key(ws: usize) -> String {
    format!("ws{ws}")
}

/// Workspace index back from a pool key, if it is one of ours.
fn key_ws(key: &str) -> Option<usize> {
    key.strip_prefix("ws")?.parse().ok()
}

/// Prompt text for a turn: the first turn of a session carries the system
/// preamble (pool sessions are created on first use, so first-use implies
/// first turn); later turns send the user text untouched. Pure: tested below.
fn first_turn_text(first: bool, text: &str) -> String {
    if first {
        format!("{SYSTEM}\n\n{text}")
    } else {
        text.to_string()
    }
}

pub struct Agent {
    pool: SessionPool,
    exe_mcp: PathBuf,
    cwd: PathBuf,
    ui: mpsc::UnboundedSender<UiMsg>,
}

impl Agent {
    /// Connect the shared pool and spawn the event router.
    pub async fn spawn(
        exe_mcp: PathBuf,
        cwd: PathBuf,
        ui: mpsc::UnboundedSender<UiMsg>,
    ) -> anyhow::Result<Self> {
        let (pool, rx) = SessionPool::connect(
            AgentCommand::new("opencode").args(["acp"]),
            NativeOptions {
                prompt_timeout: Duration::from_secs(1800),
                permissions: PermissionPolicy::Ask,
                ..NativeOptions::default()
            },
        )
        .await;
        tokio::spawn(run_router(pool.clone(), rx, ui.clone()));
        Ok(Self {
            pool,
            exe_mcp,
            cwd,
            ui,
        })
    }

    pub fn prompt(&self, text: String, ws: usize) {
        let pool = self.pool.clone();
        let ui = self.ui.clone();
        let exe_mcp = self.exe_mcp.clone();
        let cwd = self.cwd.clone();
        tokio::spawn(async move {
            let key = pool_key(ws);
            let first = !pool.has_session(&key).await;
            let mut mcp = McpServerStdio::new("ccti-comfy", exe_mcp);
            mcp.args = vec!["--mcp".to_string()];
            // Only the first use consumes these: the pool creates the key's
            // session from them once and reuses it for later turns.
            let opts = SessionOptions {
                cwd,
                model: Some(DEFAULT_CHAT_MODEL.to_string()),
                configuration: Vec::new(),
                mcp_servers: vec![McpServer::Stdio(mcp)],
            };
            let _ = ui.send(UiMsg::AgentBusy { ws, busy: true });
            match pool.prompt(&key, opts, first_turn_text(first, &text)).await {
                Ok(()) => {
                    let _ = ui.send(UiMsg::AgentDone);
                }
                Err(NativeError::Busy) => {
                    let _ = ui.send(UiMsg::AgentBusy { ws, busy: false });
                    let _ = ui.send(UiMsg::Error(format!(
                        "ws{} is busy (/cancel first)",
                        ws + 1
                    )));
                }
                Err(NativeError::Closed) => {
                    let _ = ui.send(UiMsg::AgentBusy { ws, busy: false });
                    let _ = ui.send(UiMsg::Error(format!(
                        "agent offline (ws{}); /render still works",
                        ws + 1
                    )));
                }
                Err(e) => {
                    let _ = ui.send(UiMsg::AgentBusy { ws, busy: false });
                    let _ = ui.send(UiMsg::Error(format!("prompt failed: {e}")));
                }
            }
        });
    }

    pub fn set_chat_model(&self, ws: usize, model: String) {
        let pool = self.pool.clone();
        let ui = self.ui.clone();
        tokio::spawn(async move {
            let key = pool_key(ws);
            if !pool.has_session(&key).await {
                let _ = ui.send(UiMsg::Error(format!(
                    "no chat yet (ws{}) — send a message first",
                    ws + 1
                )));
                return;
            }
            match pool.set_model(&key, &model).await {
                Ok(cfg) => {
                    let shown = cfg.model_display_name().unwrap_or(model.clone());
                    let _ = ui.send(UiMsg::ChatModel { ws, model: shown });
                }
                Err(e) => {
                    let _ = ui.send(UiMsg::Error(format!("chat model switch failed: {e}")));
                }
            }
        });
    }

    pub fn cancel(&self, ws: usize) {
        let pool = self.pool.clone();
        let ui = self.ui.clone();
        tokio::spawn(async move {
            if let Err(e) = pool.cancel(&pool_key(ws)).await {
                let _ = ui.send(UiMsg::Error(format!("cancel failed: {e}")));
            }
        });
    }
}

/// Route tagged pool events to UI messages. Assistant text is read back from
/// the key's conversation snapshot (which the pool updates before re-emitting
/// each event), so turns render as one message on completion instead of
/// streamed fragments.
async fn run_router(
    pool: SessionPool,
    mut rx: mpsc::UnboundedReceiver<PoolEvent>,
    ui: mpsc::UnboundedSender<UiMsg>,
) {
    let send = |m: UiMsg| {
        let _ = ui.send(m);
    };
    let mut connected = false;
    let mut announced: HashSet<String> = HashSet::new();
    while let Some(msg) = rx.recv().await {
        match msg {
            PoolEvent::Session { key, event } => {
                if !connected {
                    connected = true;
                    send(UiMsg::Status("agent connected".into()));
                }
                // Announce the session's model once its configuration arrives.
                if !announced.contains(&key)
                    && let Some(cfg) = pool.session_configuration(&key).await
                    && let Some(model) = cfg.model_display_name()
                    && let Some(ws) = key_ws(&key)
                {
                    announced.insert(key.clone());
                    send(UiMsg::SessionModel { ws, model });
                }
                let rid = event.request_id.clone().unwrap_or_default();
                let ccht::SessionEvent { event: payload, .. } = &*event;
                match payload {
                    Event::Update { update } => match update {
                        SessionUpdate::AgentMessageChunk(_)
                        | SessionUpdate::AgentThoughtChunk(_) => {}
                        SessionUpdate::ToolCall(tc) => {
                            send(UiMsg::Chat {
                                ws: key_ws(&key),
                                role: "tool".into(),
                                text: format!("{} …", tc.title),
                            });
                        }
                        SessionUpdate::ToolCallUpdate(u) => {
                            if let Some(title) = u.fields.title.as_deref() {
                                let st = u
                                    .fields
                                    .status
                                    .map(|s| format!("{s:?}"))
                                    .unwrap_or_default();
                                send(UiMsg::Status(format!("{title} {st}")));
                            }
                        }
                        _ => {}
                    },
                    Event::Permission {
                        request_id,
                        request,
                    } => {
                        let title = request.tool_call.fields.title.clone().unwrap_or_default();
                        let decision = decide(request);
                        let verdict = match &decision {
                            RequestPermissionOutcome::Selected(_) => "auto-allowed",
                            _ => "denied",
                        };
                        let pool2 = pool.clone();
                        let ui2 = ui.clone();
                        let key2 = key.clone();
                        let request_id = request_id.clone();
                        tokio::spawn(async move {
                            if pool2
                                .respond_permission(&key2, &request_id, decision)
                                .await
                                .is_err()
                            {
                                let _ = ui2.send(UiMsg::Error("permission response failed".into()));
                            } else {
                                let _ = ui2.send(UiMsg::Chat {
                                    ws: key_ws(&key2),
                                    role: "sys".into(),
                                    text: format!("permission {verdict}: {title}"),
                                });
                            }
                        });
                    }
                    Event::Completed { .. } => {
                        let mut text = String::new();
                        if let Some(conv) = pool.conversation(&key).await
                            && let Some(turn) = conv.turns().iter().find(|t| t.request_id == rid)
                        {
                            text = turn.text.clone();
                        }
                        if !text.trim().is_empty() {
                            send(UiMsg::Chat {
                                ws: key_ws(&key),
                                role: "agent".into(),
                                text,
                            });
                        }
                        if let Some(ws) = key_ws(&key) {
                            send(UiMsg::AgentBusy { ws, busy: false });
                        }
                    }
                    Event::Error { code, message } => {
                        if let Some(ws) = key_ws(&key) {
                            send(UiMsg::AgentBusy { ws, busy: false });
                        }
                        send(UiMsg::Error(format!("{code}: {message}")));
                    }
                }
            }
            PoolEvent::Ended { key } => {
                // Forwarder drained: the session stream closed with no turn
                // result. Clear the indicator and say so; local tools like
                // /render keep working.
                if let Some(ws) = key_ws(&key) {
                    send(UiMsg::AgentBusy { ws, busy: false });
                    send(UiMsg::Error(format!("agent stream closed (ws{})", ws + 1)));
                }
            }
        }
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

    #[test]
    fn pool_keys_roundtrip_workspace_indexes() {
        assert_eq!(pool_key(0), "ws0");
        assert_eq!(pool_key(3), "ws3");
        assert_eq!(key_ws("ws0"), Some(0));
        assert_eq!(key_ws("ws3"), Some(3));
        assert_eq!(key_ws("main"), None);
        assert_eq!(key_ws(""), None);
    }

    #[test]
    fn first_turn_carries_system_preamble_once() {
        let first = first_turn_text(true, "hello");
        assert!(first.starts_with(SYSTEM));
        assert!(first.ends_with("hello"));
        assert_eq!(first_turn_text(false, "hello"), "hello");
    }
}
