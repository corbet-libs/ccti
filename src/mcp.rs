//! Minimal MCP server over stdio (newline-delimited JSON-RPC).
//! One tool: `render_image` -> ComfyUI txt2img. Served via `ccti --mcp`.

use anyhow::Result;
use serde_json::{Value, json};
use std::collections::HashMap;
use std::time::{Duration, SystemTime, UNIX_EPOCH};
use tokio::io::{AsyncBufReadExt, AsyncWriteExt, BufReader};

use crate::comfy::{self, Client};
use crate::prov::{self, Generation};

const TOOL_DESC: &str = "Render images with ComfyUI (SDXL txt2img). \
Defaults are tuned for fast iteration: 512x512, 8 steps, 1 image. \
Use n (up to 4) for variants, larger sizes/steps for finals. \
Handles the GPU cold start itself; may take minutes. Returns saved paths \
(with generation metadata embedded plus JSON sidecars) and a preview of the \
first image. If attached images are not visible to you, Read the saved PNG \
file(s) to view them, then iterate: adjust prompt, size, steps or seed and \
call render_image again.";

/// Serve MCP on stdin/stdout until EOF.
pub async fn serve() -> Result<()> {
    let stdin = tokio::io::stdin();
    let mut lines = BufReader::new(stdin).lines();
    let mut stdout = tokio::io::stdout();
    while let Some(line) = lines.next_line().await? {
        let line = line.trim().to_string();
        if line.is_empty() {
            continue;
        }
        let req: Value = match serde_json::from_str(&line) {
            Ok(v) => v,
            Err(_) => continue, // never break the loop on garbage
        };
        if let Some(resp) = handle(&req).await {
            stdout
                .write_all(serde_json::to_string(&resp)?.as_bytes())
                .await?;
            stdout.write_all(b"\n").await?;
            stdout.flush().await?;
        }
    }
    Ok(())
}

async fn handle(req: &Value) -> Option<Value> {
    if req.get("jsonrpc") != Some(&json!("2.0")) {
        return None;
    }
    let id = req.get("id")?.clone();
    let method = req.get("method")?.as_str()?;
    let params = req.get("params").cloned().unwrap_or(Value::Null);
    let result = match method {
        "initialize" => Ok(init_result(&params)),
        "ping" => Ok(json!({})),
        "tools/list" => Ok(json!({"tools": tool_defs()})),
        "tools/call" => call_tool(&params).await,
        _ if method.starts_with("notifications/") => return None,
        _ => Err(err(-32601, "method not found")),
    };
    Some(match result {
        Ok(r) => json!({"jsonrpc": "2.0", "id": id, "result": r}),
        Err(e) => json!({"jsonrpc": "2.0", "id": id, "error": e}),
    })
}

fn init_result(params: &Value) -> Value {
    let version = params
        .get("protocolVersion")
        .and_then(|v| v.as_str())
        .unwrap_or("2025-06-18")
        .to_string();
    json!({
        "protocolVersion": version,
        "capabilities": {"tools": {}},
        "serverInfo": {"name": "ccti-comfy", "version": env!("CARGO_PKG_VERSION")},
    })
}

fn tool_def() -> Value {
    json!({
        "name": "render_image",
        "description": TOOL_DESC,
        "inputSchema": {
            "type": "object",
            "required": ["prompt"],
            "properties": {
                "prompt": {"type": "string", "description": "Image prompt, English, descriptive"},
                "width": {"type": "integer", "default": 512},
                "height": {"type": "integer", "default": 512},
                "steps": {"type": "integer", "default": 8},
                "n": {"type": "integer", "default": 1, "description": "How many variants (1-4)"},
                "seed": {"type": "integer"},
                "ckpt": {"type": "string", "description": "Checkpoint override"}
            }
        }
    })
}

fn tool_defs() -> Vec<Value> {
    vec![
        tool_def(),
        json!({
            "name": "render_submit",
            "description": "Start a ComfyUI render in the background and return a render_id immediately. \
                Use this for fresh renders (cold starts take minutes and would exceed a single \
                tool-call timeout); poll render_status until done, then render_result. Same \
                arguments as render_image.",
            "inputSchema": tool_def()["inputSchema"].clone(),
        }),
        json!({
            "name": "render_status",
            "description": "Poll a background render by render_id. Cheap: call repeatedly until done/failed.",
            "inputSchema": {
                "type": "object",
                "required": ["render_id"],
                "properties": {
                    "render_id": {"type": "string"}
                }
            }
        }),
        json!({
            "name": "render_result",
            "description": "Fetch a finished background render: saved paths plus a preview image. \
                Fails honestly while the render is still running.",
            "inputSchema": {
                "type": "object",
                "required": ["render_id"],
                "properties": {
                    "render_id": {"type": "string"}
                }
            }
        }),
    ]
}

fn err(code: i32, msg: &str) -> Value {
    json!({"code": code, "message": msg})
}

async fn call_tool(params: &Value) -> Result<Value, Value> {
    let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    let a = params.get("arguments").cloned().unwrap_or(Value::Null);
    match name {
        "render_image" => {
            let job = parse_render_args(&a)?;
            render_image_sync(job).await
        }
        "render_submit" => {
            let job = parse_render_args(&a)?;
            Ok(submit_job(job))
        }
        "render_status" => {
            let id = a
                .get("render_id")
                .and_then(|v| v.as_str())
                .ok_or_else(|| err(-32602, "render_id is required"))?;
            status_text(id)
        }
        "render_result" => result_tool(&a).await,
        _ => Err(err(-32602, "unknown tool")),
    }
}

/// Owned render parameters shared by all three render tools.
#[derive(Clone)]
struct JobParams {
    prompt: String,
    ckpt: String,
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    n: u32,
}

fn parse_render_args(a: &Value) -> Result<JobParams, Value> {
    let prompt = a
        .get("prompt")
        .and_then(|p| p.as_str())
        .ok_or_else(|| err(-32602, "prompt is required"))?;
    let width = a.get("width").and_then(|v| v.as_u64()).unwrap_or(512) as u32;
    let height = a.get("height").and_then(|v| v.as_u64()).unwrap_or(512) as u32;
    let steps = a.get("steps").and_then(|v| v.as_u64()).unwrap_or(8) as u32;
    let n = a.get("n").and_then(|v| v.as_u64()).unwrap_or(1).clamp(1, 4) as u32;
    let seed = a.get("seed").and_then(|v| v.as_u64()).unwrap_or_else(|| {
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(7)
    });
    let ckpt = a
        .get("ckpt")
        .and_then(|v| v.as_str())
        .unwrap_or(comfy::DEFAULT_CKPT)
        .to_string();
    Ok(JobParams {
        prompt: prompt.to_string(),
        ckpt,
        width,
        height,
        steps,
        seed,
        n,
    })
}

#[derive(Clone)]
enum JobState {
    Queued,
    Rendering(String),
    Done(Vec<String>),
    Failed(String),
}

static JOBS: std::sync::LazyLock<std::sync::Mutex<HashMap<String, JobState>>> =
    std::sync::LazyLock::new(|| std::sync::Mutex::new(HashMap::new()));
static NEXT_ID: std::sync::atomic::AtomicU64 = std::sync::atomic::AtomicU64::new(1);

/// Synchronous render_image path: waits for the whole render, then answers
/// with summary text plus an inline preview.
async fn render_image_sync(job: JobParams) -> Result<Value, Value> {
    let base = std::env::var("CCTI_COMFY_URL").unwrap_or_else(|_| comfy::DEFAULT_BASE.into());
    let paths = match run_job(&job, &base, |_| {}).await {
        Ok(p) => p,
        Err(e) => {
            return Ok(json!({
                "content": [{"type": "text", "text": format!("render failed: {e:#}")}],
                "isError": true,
            }));
        }
    };
    let mut content = vec![json!({
        "type": "text",
        "text": format!(
            "rendered {} image(s): {}. Provenance is embedded plus sidecar JSON. \
             If attached images are not visible to you, Read the saved PNG file(s) \
             to view them, then iterate.",
            paths.len(),
            paths.join(", "),
        ),
    })];
    if let Some(data) = preview_of_first(&paths).await {
        content.push(json!({"type": "image", "data": data, "mimeType": "image/png"}));
    }
    Ok(json!({ "content": content }))
}

/// Background submit: answers immediately so slow cold starts never hit a
/// client-side tool timeout. The worker drives the same core and records
/// progress for render_status/render_result.
fn submit_job(job: JobParams) -> Value {
    let id = format!(
        "r{}_{}",
        SystemTime::now()
            .duration_since(UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0),
        NEXT_ID.fetch_add(1, std::sync::atomic::Ordering::Relaxed)
    );
    JOBS.lock().unwrap().insert(id.clone(), JobState::Queued);
    let base = std::env::var("CCTI_COMFY_URL").unwrap_or_else(|_| comfy::DEFAULT_BASE.into());
    let summary = format!(
        "render {id} queued ({},{},{}x{}, {} steps). Poll render_status until done, then render_result.",
        job.prompt.chars().take(80).collect::<String>(),
        job.ckpt,
        job.width,
        job.height,
        job.steps,
    );
    let task_id = id.clone();
    tokio::spawn(async move {
        let set = |state: JobState| {
            JOBS.lock().unwrap().insert(task_id.clone(), state);
        };
        let out = run_job(&job, &base, |p| set(JobState::Rendering(p))).await;
        match out {
            Ok(paths) => set(JobState::Done(paths)),
            Err(e) => set(JobState::Failed(format!("{e:#}"))),
        }
    });
    json!({
        "content": [{
            "type": "text",
            "text": summary,
        }],
        // Machine-readable id alongside the prose.
        "render_id": id,
    })
}

fn job_state(id: &str) -> Result<JobState, Value> {
    JOBS.lock()
        .unwrap()
        .get(id)
        .cloned()
        .ok_or_else(|| err(-32602, "unknown render_id (server restart drops jobs)"))
}

fn status_text(id: &str) -> Result<Value, Value> {
    let text = match job_state(id)? {
        JobState::Queued => format!("render {id}: queued, not started yet"),
        JobState::Rendering(p) => format!("render {id}: rendering — {p}"),
        JobState::Done(paths) => format!(
            "render {id}: done, {} image(s): {}",
            paths.len(),
            paths.join(", ")
        ),
        JobState::Failed(e) => {
            return Ok(json!({
                "content": [{"type": "text", "text": format!("render {id} FAILED: {e}")}],
                "isError": true,
            }));
        }
    };
    Ok(json!({ "content": [{"type": "text", "text": text}] }))
}

/// Full result including the preview image. Honest while running: says so
/// instead of failing, so polling loops stay simple.
async fn result_tool(params: &Value) -> Result<Value, Value> {
    let id = params
        .get("render_id")
        .and_then(|v| v.as_str())
        .ok_or_else(|| err(-32602, "render_id is required"))?;
    match job_state(id)? {
        JobState::Done(paths) => {
            let mut content = vec![json!({
                "type": "text",
                "text": format!(
                    "render {id} done, {} image(s): {}. Provenance is embedded plus sidecar JSON. \
                     If attached images are not visible to you, Read the saved PNG file(s) to view \
                     them, then iterate.",
                    paths.len(),
                    paths.join(", "),
                ),
            })];
            if let Some(data) = preview_of_first(&paths).await {
                content.push(json!({"type": "image", "data": data, "mimeType": "image/png"}));
            }
            Ok(json!({ "content": content }))
        }
        JobState::Failed(e) => Ok(json!({
            "content": [{"type": "text", "text": format!("render {id} FAILED: {e}")}],
            "isError": true,
        })),
        JobState::Queued => Ok(json!({
            "content": [{"type": "text", "text": format!("render {id} queued, not started yet — poll render_status")}]
        })),
        JobState::Rendering(p) => Ok(json!({
            "content": [{"type": "text", "text": format!("render {id} still rendering — {p}; poll render_status")}]
        })),
    }
}

/// Downscaled preview of the first finished image, if it still reads.
async fn preview_of_first(paths: &[String]) -> Option<String> {
    let first = paths.first()?.clone();
    tokio::task::spawn_blocking(move || -> Result<String> {
        let bytes = std::fs::read(&first)?;
        let dyn_img = image::load_from_memory(&bytes)?;
        let small = dyn_img.thumbnail(512, 512);
        let mut buf = Vec::new();
        {
            let mut cur = std::io::Cursor::new(&mut buf);
            small.write_to(&mut cur, image::ImageFormat::Png)?;
        }
        Ok(base64::Engine::encode(
            &base64::engine::general_purpose::STANDARD,
            &buf,
        ))
    })
    .await
    .ok()?
    .ok()
}

fn images_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    std::path::PathBuf::from(format!("{home}/images/generated"))
}

/// Shared render core for all three tools: wait, render the batch,
/// download, store with provenance. Reports progress for status polling.
async fn run_job(
    job: &JobParams,
    base: &str,
    mut progress: impl FnMut(String),
) -> Result<Vec<String>> {
    let client = Client::new(base)?;
    client
        .wait_ready(Duration::from_secs(420), &mut progress)
        .await?;
    let refs = client
        .render(
            comfy::RenderOpts {
                prompt: &job.prompt,
                ckpt: &job.ckpt,
                width: job.width,
                height: job.height,
                steps: job.steps,
                seed: job.seed,
                n: job.n,
            },
            &mut progress,
        )
        .await?;
    let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let dir = images_dir();
    let mut paths = Vec::new();
    for (i, img) in refs.iter().enumerate() {
        let bytes = client.download(img).await?;
        let generation = Generation {
            prompt: job.prompt.clone(),
            negative: "blurry, watermark, text, deformed".to_string(),
            ckpt: job.ckpt.clone(),
            width: job.width,
            height: job.height,
            steps: job.steps,
            cfg: 1.5,
            sampler: "euler".to_string(),
            scheduler: "normal".to_string(),
            seed: job.seed,
            n: refs.len() as u32,
            index: i as u32,
            software: format!("ccti {}", env!("CARGO_PKG_VERSION")),
            created_unix: ts,
        };
        let stem = format!("ccti_{ts}_{i}");
        let (png_path, _) = prov::store_rendered(&dir, &stem, &bytes, &generation)?;
        paths.push(png_path.display().to_string());
    }
    if paths.is_empty() {
        anyhow::bail!("ComfyUI returned no images");
    }
    Ok(paths)
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;

    async fn call(req: Value) -> Option<Value> {
        handle(&req).await
    }

    #[tokio::test]
    async fn initialize_echoes_protocol_version() {
        let resp = call(json!({
            "jsonrpc": "2.0", "id": 1, "method": "initialize",
            "params": {"protocolVersion": "2025-06-18", "capabilities": {},
                       "clientInfo": {"name": "t", "version": "0"}}
        }))
        .await
        .expect("must answer");
        assert_eq!(resp["result"]["protocolVersion"], "2025-06-18");
        assert_eq!(resp["result"]["serverInfo"]["name"], "ccti-comfy");
    }

    #[tokio::test]
    async fn tools_list_exposes_all_render_tools() {
        let resp = call(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}))
            .await
            .expect("must answer");
        let tools = resp["result"]["tools"].as_array().expect("tools array");
        let names: Vec<&str> = tools.iter().filter_map(|t| t["name"].as_str()).collect();
        assert_eq!(
            names,
            vec![
                "render_image",
                "render_submit",
                "render_status",
                "render_result"
            ]
        );
        assert!(
            tools[0]["inputSchema"]["required"]
                .as_array()
                .unwrap()
                .contains(&json!("prompt"))
        );
    }

    #[test]
    fn parse_args_defaults_fast_and_clamps_batch() {
        let a = parse_render_args(&json!({"prompt": "x"})).unwrap();
        assert_eq!((a.width, a.height, a.steps, a.n), (512, 512, 8, 1));
        let a = parse_render_args(&json!({"prompt": "x", "n": 99})).unwrap();
        assert_eq!(a.n, 4);
        assert!(parse_render_args(&json!({})).is_err());
    }

    #[tokio::test]
    async fn unknown_render_id_is_an_error() {
        for method in ["render_status", "render_result"] {
            let resp = call(json!({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
                "params": {"name": method, "arguments": {"render_id": "r-nope"}}}))
            .await
            .expect("must answer");
            assert_eq!(resp["error"]["code"], -32602, "{method}");
        }
    }

    #[tokio::test]
    async fn pending_render_reports_honestly() {
        // Seed a queued job directly: no network involved.
        let id = "r-test-pending";
        JOBS.lock().unwrap().insert(
            id.to_string(),
            JobState::Rendering("warming up".to_string()),
        );
        let resp = call(json!({"jsonrpc": "2.0", "id": 9, "method": "tools/call",
            "params": {"name": "render_result", "arguments": {"render_id": id}}}))
        .await
        .expect("must answer");
        let text = resp["result"]["content"][0]["text"].as_str().unwrap();
        assert!(text.contains("still rendering"), "{text}");
        assert!(resp["result"]["content"].as_array().unwrap().len() == 1);
        JOBS.lock().unwrap().remove(id);
    }

    #[tokio::test]
    async fn unknown_method_is_protocol_error() {
        let resp = call(json!({"jsonrpc": "2.0", "id": 3, "method": "tools/bogus", "params": {}}))
            .await
            .expect("must answer");
        assert_eq!(resp["error"]["code"], -32601);
    }

    #[tokio::test]
    async fn notification_without_id_gets_no_reply() {
        let resp = call(json!({"jsonrpc": "2.0", "method": "notifications/initialized"})).await;
        assert!(resp.is_none());
    }

    #[tokio::test]
    async fn unknown_tool_is_invalid_params() {
        let resp = call(json!({"jsonrpc": "2.0", "id": 4, "method": "tools/call",
                               "params": {"name": "nope", "arguments": {}}}))
        .await
        .expect("must answer");
        assert_eq!(resp["error"]["code"], -32602);
    }

    #[tokio::test]
    async fn render_call_without_prompt_is_invalid_params() {
        let resp = call(json!({"jsonrpc": "2.0", "id": 5, "method": "tools/call",
                               "params": {"name": "render_image", "arguments": {}}}))
        .await
        .expect("must answer");
        assert_eq!(resp["error"]["code"], -32602);
    }
}
