//! ccti — ComfyUI TUI: image left, chat right.
//! Modes: default TUI · `--mcp` MCP stdio server · `--render` headless render.

mod agent;
mod comfy;
mod mcp;
mod prov;
mod ui;

use std::time::Duration;

use anyhow::Result;
use tokio::sync::mpsc;

use crate::ui::UiMsg;

fn flag(args: &[String], name: &str) -> Option<String> {
    args.windows(2).find(|w| w[0] == name).map(|w| w[1].clone())
}

#[tokio::main(flavor = "multi_thread")]
async fn main() -> Result<()> {
    let args: Vec<String> = std::env::args().collect();
    let has = |n: &str| args.iter().any(|a| a == n);

    if has("--mcp") {
        return mcp::serve().await;
    }
    if has("--agent-probe") {
        return agent_probe().await;
    }
    if has("--help") || has("-h") {
        println!("{}", help_text());
        return Ok(());
    }
    if has("--version") || has("-V") {
        println!("ccti {}", env!("CARGO_PKG_VERSION"));
        return Ok(());
    }
    if has("--models") {
        let base = std::env::var("CCTI_COMFY_URL").unwrap_or_else(|_| comfy::DEFAULT_BASE.into());
        let c = comfy::Client::new(&base)?;
        c.wait_ready(Duration::from_secs(420), |p| eprintln!("{p}"))
            .await?;
        let list = c.checkpoints().await?;
        println!("{}", list.join("\n"));
        return Ok(());
    }
    if let Some(prompt) = flag(&args, "--render") {
        let base = std::env::var("CCTI_COMFY_URL").unwrap_or_else(|_| comfy::DEFAULT_BASE.into());
        let out = flag(&args, "--out").unwrap_or_else(|| {
            let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
            format!("{home}/images/generated/ccti_headless.png")
        });
        let w: u32 = flag(&args, "--w")
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let h: u32 = flag(&args, "--h")
            .and_then(|v| v.parse().ok())
            .unwrap_or(512);
        let steps: u32 = flag(&args, "--steps")
            .and_then(|v| v.parse().ok())
            .unwrap_or(8);
        let n: u32 = flag(&args, "--n")
            .and_then(|v| v.parse().ok())
            .unwrap_or(1)
            .clamp(1, 4);
        let c = comfy::Client::new(&base)?;
        c.wait_ready(Duration::from_secs(420), |p| eprintln!("{p}"))
            .await?;
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(7);
        let refs = c
            .render(
                comfy::RenderOpts {
                    prompt: &prompt,
                    ckpt: comfy::DEFAULT_CKPT,
                    width: w,
                    height: h,
                    steps,
                    seed,
                    n,
                },
                |p| eprintln!("{p}"),
            )
            .await?;
        let ts = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(0);
        let home = std::env::var("HOME").unwrap_or_else(|_| "/tmp".into());
        let dir = std::path::PathBuf::from(format!("{home}/images/generated"));
        for (i, img) in refs.iter().enumerate() {
            let bytes = c.download(img).await?;
            let generation = prov::Generation {
                prompt: prompt.clone(),
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
            // --out names the first image; batch siblings get suffixed names.
            if i == 0 && refs.len() == 1 {
                let raw = std::path::PathBuf::from(&out);
                let stem = raw
                    .file_stem()
                    .and_then(|s| s.to_str())
                    .unwrap_or("ccti_headless")
                    .to_string();
                let parent = raw.parent().map(|p| p.to_path_buf()).unwrap_or(dir.clone());
                let (png_path, json_path) =
                    prov::store_rendered(&parent, &stem, &bytes, &generation)?;
                println!("saved: {} (+ {})", png_path.display(), json_path.display());
            } else {
                let stem = format!("ccti_{ts}_{i}");
                let (png_path, _) = prov::store_rendered(&dir, &stem, &bytes, &generation)?;
                println!("saved: {}", png_path.display());
            }
        }
        return Ok(());
    }
    ui::run().await
}

/// Diagnostic: fire one prompt per workspace concurrently and print where
/// each answer lands. With per-workspace sessions both must complete; the
/// old shared session would reject the second as busy.
async fn agent_probe() -> Result<()> {
    let (tx, mut rx) = mpsc::unbounded_channel::<UiMsg>();
    let exe = std::env::current_exe()?;
    let agent = agent::Agent::spawn(exe, std::env::temp_dir(), tx).await?;
    agent.prompt("Reply with exactly: PROBE-OK".to_string(), 0);
    // Small stagger so both turns overlap instead of queueing politely.
    tokio::time::sleep(Duration::from_millis(500)).await;
    agent.prompt("Reply with exactly: PROBE-OK".to_string(), 1);
    let deadline = tokio::time::sleep(Duration::from_secs(420));
    tokio::pin!(deadline);
    let mut got = [false, false];
    let mut models = [None, None];
    loop {
        tokio::select! {
            msg = rx.recv() => {
                let Some(msg) = msg else { break };
                match msg {
                    UiMsg::Chat { ws, role, text } if role == "agent" => {
                        let i = ws.unwrap_or(0);
                        println!("ws{i} agent: {}", text.trim().chars().take(120).collect::<String>());
                        if text.contains("PROBE-OK") && i < 2 {
                            got[i] = true;
                        }
                    }
                    UiMsg::SessionModel { ws, model } => {
                        println!("ws{ws} session model: {model}");
                        if ws < 2 {
                            models[ws] = Some(model);
                        }
                    }
                    UiMsg::Error(e) => println!("error: {e}"),
                    UiMsg::Status(s) => println!("status: {s}"),
                    _ => {}
                }
                if got == [true, true] {
                    break;
                }
            }
            _ = &mut deadline => {
                println!("TIMEOUT waiting for both workspaces");
                break;
            }
        }
    }
    println!("ws0 ok: {}, ws1 ok: {}", got[0], got[1]);
    println!("models: {:?} {:?}", models[0], models[1]);
    if got == [true, true] {
        Ok(())
    } else {
        anyhow::bail!("parallel probe incomplete")
    }
}

fn help_text() -> String {
    format!(
        "ccti {} — ComfyUI TUI: agentic image chat with inline terminal rendering\n\n\
         Usage:\n  \
           ccti                              Start the TUI (needs a real terminal)\n  \
           ccti --render PROMPT [--out FILE] [--w N --h N --steps N --n 1-4]\n  \
           ccti --models                     List ComfyUI checkpoints\n  \
           ccti --mcp                        MCP stdio server (render_image tool)\n  \
           ccti --agent-probe                Two parallel prompts (parallelism check)\n  \
           ccti --help | --version",
        env!("CARGO_PKG_VERSION")
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn help_mentions_all_modes() {
        let h = help_text();
        for token in [
            "ccti --render",
            "ccti --models",
            "ccti --mcp",
            "ccti --help",
            "TUI",
        ] {
            assert!(h.contains(token), "help missing {token}");
        }
    }
}
