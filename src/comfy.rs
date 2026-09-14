//! Minimal ComfyUI HTTP client (txt2img). No UI, no state.

use anyhow::{Context, Result};
use serde_json::{Value, json};
use std::time::{Duration, Instant};

pub const DEFAULT_BASE: &str = "https://comfyui.corbet.ch";
pub const DEFAULT_CKPT: &str = "realvisxlV50_v50LightningBakedvae.safetensors";

/// Render parameters (one struct to keep the call sites readable).
pub struct RenderOpts<'a> {
    pub prompt: &'a str,
    pub ckpt: &'a str,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub seed: u64,
    /// Batch size: how many images this render produces.
    pub n: u32,
}
/// Reference to a finished image on the ComfyUI server.
pub struct ImageRef {
    pub filename: String,
    pub subfolder: String,
    pub typ: String,
}

#[derive(Clone)]
pub struct Client {
    base: String,
    http: reqwest::Client,
}

impl Client {
    pub fn new(base: &str) -> Result<Self> {
        let http = reqwest::Client::builder()
            .timeout(Duration::from_secs(120))
            .user_agent("ccti/0.1")
            .build()
            .context("http client")?;
        Ok(Self {
            base: base.trim_end_matches('/').to_string(),
            http,
        })
    }

    /// Wait until the API answers JSON (covers the Sablier cold start).
    pub async fn wait_ready(
        &self,
        timeout: Duration,
        mut progress: impl FnMut(String),
    ) -> Result<()> {
        let start = Instant::now();
        loop {
            if let Ok(r) = self
                .http
                .get(format!("{}/system_stats", self.base))
                .send()
                .await
                && let Ok(v) = r.json::<Value>().await
                && v.get("system").is_some()
            {
                return Ok(());
            }
            if start.elapsed() > timeout {
                anyhow::bail!("ComfyUI did not become ready in time");
            }
            progress(format!("waking ComfyUI ({}s)…", start.elapsed().as_secs()));
            tokio::time::sleep(Duration::from_secs(10)).await;
        }
    }

    /// Checkpoint names accepted by CheckpointLoaderSimple.
    pub async fn checkpoints(&self) -> Result<Vec<String>> {
        let v: Value = self
            .http
            .get(format!("{}/object_info/CheckpointLoaderSimple", self.base))
            .send()
            .await
            .context("object_info")?
            .error_for_status()?
            .json()
            .await?;
        let names = v["CheckpointLoaderSimple"]["input"]["required"]["ckpt_name"][0]
            .as_array()
            .context("ckpt list shape")?
            .iter()
            .filter_map(|n| n.as_str().map(str::to_string))
            .collect();
        Ok(names)
    }

    /// Submit txt2img and wait for the finished image references (batch).
    pub async fn render(
        &self,
        opts: RenderOpts<'_>,
        mut progress: impl FnMut(String),
    ) -> Result<Vec<ImageRef>> {
        let RenderOpts {
            prompt,
            ckpt,
            width,
            height,
            steps,
            seed,
            n,
        } = opts;
        let wf = build_workflow(prompt, ckpt, width, height, steps, seed, n);
        let pid: String = self
            .http
            .post(format!("{}/prompt", self.base))
            .json(&json!({"prompt": wf, "client_id": "ccti"}))
            .send()
            .await
            .context("submit prompt")?
            .error_for_status()?
            .json::<Value>()
            .await?
            .get("prompt_id")
            .and_then(|v| v.as_str())
            .context("prompt_id missing")?
            .to_string();
        let start = Instant::now();
        loop {
            tokio::time::sleep(Duration::from_secs(5)).await;
            progress(format!("rendering ({}s)…", start.elapsed().as_secs()));
            if start.elapsed() > Duration::from_secs(900) {
                anyhow::bail!("render timed out (GPU contention?)");
            }
            let h: Value = match self
                .http
                .get(format!("{}/history/{}", self.base, pid))
                .send()
                .await
            {
                Ok(r) => r.json().await.unwrap_or(Value::Null),
                Err(_) => continue,
            };
            if let Some(images) = h
                .get(&pid)
                .and_then(|p| p.get("outputs"))
                .and_then(|o| o.get("7"))
                .and_then(|n| n.get("images"))
                .and_then(|i| i.as_array())
                .filter(|a| !a.is_empty())
            {
                return Ok(images
                    .iter()
                    .map(|img| ImageRef {
                        filename: img["filename"].as_str().unwrap_or("").to_string(),
                        subfolder: img["subfolder"].as_str().unwrap_or("").to_string(),
                        typ: img["type"].as_str().unwrap_or("output").to_string(),
                    })
                    .collect());
            }
        }
    }

    /// Download finished image bytes via /view.
    pub async fn download(&self, img: &ImageRef) -> Result<Vec<u8>> {
        let url = format!(
            "{}/view?filename={}&subfolder={}&type={}",
            self.base,
            urlenc(&img.filename),
            urlenc(&img.subfolder),
            urlenc(&img.typ)
        );
        let bytes = self
            .http
            .get(&url)
            .send()
            .await
            .context("view")?
            .error_for_status()?
            .bytes()
            .await?;
        Ok(bytes.to_vec())
    }
}

fn urlenc(s: &str) -> String {
    let mut out = String::new();
    for b in s.bytes() {
        if b.is_ascii_alphanumeric() || b"-_.~".contains(&b) {
            out.push(b as char);
        } else {
            out.push_str(&format!("%{b:02X}"));
        }
    }
    out
}

fn build_workflow(
    prompt: &str,
    ckpt: &str,
    width: u32,
    height: u32,
    steps: u32,
    seed: u64,
    n: u32,
) -> Value {
    json!({
        "1": {"class_type": "CheckpointLoaderSimple", "inputs": {"ckpt_name": ckpt}},
        "2": {"class_type": "CLIPTextEncode", "inputs": {"text": prompt, "clip": ["1", 1]}},
        "3": {"class_type": "CLIPTextEncode",
              "inputs": {"text": "blurry, watermark, text, deformed", "clip": ["1", 1]}},
        "4": {"class_type": "EmptyLatentImage",
              "inputs": {"width": width, "height": height, "batch_size": n}},
        "5": {"class_type": "KSampler",
              "inputs": {"model": ["1", 0], "positive": ["2", 0], "negative": ["3", 0],
                         "latent_image": ["4", 0], "seed": seed, "steps": steps, "cfg": 1.5,
                         "sampler_name": "euler", "scheduler": "normal", "denoise": 1.0}},
        "6": {"class_type": "VAEDecode", "inputs": {"samples": ["5", 0], "vae": ["1", 2]}},
        "7": {"class_type": "SaveImage",
              "inputs": {"images": ["6", 0], "filename_prefix": "ccti"}},
    })
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn urlenc_leaves_unreserved_chars_untouched() {
        assert_eq!(urlenc("abcXYZ019-_.~"), "abcXYZ019-_.~");
    }

    #[test]
    fn urlenc_escapes_space_slash_and_multibyte() {
        assert_eq!(urlenc("a b/c"), "a%20b%2Fc");
        assert_eq!(urlenc("ä"), "%C3%A4");
    }

    #[test]
    fn workflow_wires_prompt_ckpt_geometry_and_sampler() {
        let wf = build_workflow("a fox", "my.safetensors", 512, 1024, 7, 42, 3);
        assert_eq!(wf["1"]["inputs"]["ckpt_name"], "my.safetensors");
        assert_eq!(wf["2"]["inputs"]["text"], "a fox");
        assert_eq!(wf["4"]["inputs"]["width"], 512);
        assert_eq!(wf["4"]["inputs"]["height"], 1024);
        assert_eq!(wf["5"]["inputs"]["steps"], 7);
        assert_eq!(wf["5"]["inputs"]["seed"], 42);
        assert_eq!(wf["5"]["inputs"]["sampler_name"], "euler");
        assert_eq!(wf["6"]["class_type"], "VAEDecode");
        assert_eq!(wf["7"]["class_type"], "SaveImage");
        assert_eq!(wf["7"]["inputs"]["filename_prefix"], "ccti");
    }
}
