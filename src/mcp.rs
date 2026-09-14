//! Minimal MCP server over stdio (newline-delimited JSON-RPC).
//! One tool: `render_image` -> ComfyUI txt2img. Served via `ccti --mcp`.

use anyhow::Result;
use serde_json::{Value, json};
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
        "tools/list" => Ok(json!({"tools": [tool_def()]})),
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

fn err(code: i32, msg: &str) -> Value {
    json!({"code": code, "message": msg})
}

async fn call_tool(params: &Value) -> Result<Value, Value> {
    let name = params.get("name").and_then(|n| n.as_str()).unwrap_or("");
    if name != "render_image" {
        return Err(err(-32602, "unknown tool"));
    }
    let a = params.get("arguments").cloned().unwrap_or(Value::Null);
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

    let base = std::env::var("CCTI_COMFY_URL").unwrap_or_else(|_| comfy::DEFAULT_BASE.into());
    let out = match render_and_store(RenderJob {
        base: &base,
        prompt,
        ckpt: &ckpt,
        width,
        height,
        steps,
        seed,
        n,
    })
    .await
    {
        Ok(o) => o,
        Err(e) => {
            return Ok(json!({
                "content": [{"type": "text", "text": format!("render failed: {e:#}")}],
                "isError": true,
            }));
        }
    };
    Ok(json!({
        "content": [
            {"type": "text", "text": out.summary},
            {"type": "image", "data": out.preview_b64, "mimeType": "image/png"},
        ],
    }))
}

struct RenderOut {
    summary: String,
    preview_b64: String,
}

fn images_dir() -> std::path::PathBuf {
    let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
    std::path::PathBuf::from(format!("{home}/images/generated"))
}

struct RenderJob<'a> {
    base: &'a str,
    prompt: &'a str,
    ckpt: &'a str,
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    n: u32,
}

async fn render_and_store(job: RenderJob<'_>) -> Result<RenderOut> {
    let RenderJob {
        base,
        prompt,
        ckpt,
        width,
        height,
        steps,
        seed,
        n,
    } = job;
    let client = Client::new(base)?;
    client.wait_ready(Duration::from_secs(420), |_| {}).await?;
    let refs = client
        .render(
            comfy::RenderOpts {
                prompt,
                ckpt,
                width,
                height,
                steps,
                seed,
                n,
            },
            |_| {},
        )
        .await?;
    let ts = SystemTime::now().duration_since(UNIX_EPOCH)?.as_secs();
    let dir = images_dir();
    let mut paths = Vec::new();
    let mut first_bytes: Option<Vec<u8>> = None;
    for (i, img) in refs.iter().enumerate() {
        let bytes = client.download(img).await?;
        let generation = Generation {
            prompt: prompt.to_string(),
            negative: "blurry, watermark, text, deformed".to_string(),
            ckpt: ckpt.to_string(),
            width,
            height,
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
        let stem = format!("ccti_{ts}_{i}");
        let (png_path, _) = prov::store_rendered(&dir, &stem, &bytes, &generation)?;
        paths.push(png_path.display().to_string());
        if first_bytes.is_none() {
            first_bytes = Some(bytes);
        }
    }
    if paths.is_empty() {
        anyhow::bail!("ComfyUI returned no images");
    }

    // Downscaled preview so callers with vision can see the result at a glance.
    let preview_b64 = tokio::task::spawn_blocking(move || -> Result<String> {
        let bytes = first_bytes.unwrap_or_default();
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
    .await??;

    Ok(RenderOut {
        summary: format!(
            "rendered {} image(s) {width}x{height} in {steps} steps (seed {seed}, {ckpt}): {}. \
             Provenance is embedded plus sidecar JSON. If attached images are not visible to you, \
             Read the saved PNG file(s) to view them, then iterate.",
            paths.len(),
            paths.join(", "),
        ),
        preview_b64,
    })
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
    async fn tools_list_exposes_render_image() {
        let resp = call(json!({"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}))
            .await
            .expect("must answer");
        let tools = resp["result"]["tools"].as_array().expect("tools array");
        assert_eq!(tools.len(), 1);
        assert_eq!(tools[0]["name"], "render_image");
        assert!(
            tools[0]["inputSchema"]["required"]
                .as_array()
                .unwrap()
                .contains(&json!("prompt"))
        );
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
