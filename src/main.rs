//! ccti — ComfyUI TUI: image left, chat right.
//! Modes: default TUI · `--mcp` MCP stdio server · `--render` headless render.

mod agent;
mod comfy;
mod mcp;
mod ui;

use std::time::Duration;

use anyhow::Result;

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
            .unwrap_or(768);
        let h: u32 = flag(&args, "--h")
            .and_then(|v| v.parse().ok())
            .unwrap_or(768);
        let steps: u32 = flag(&args, "--steps")
            .and_then(|v| v.parse().ok())
            .unwrap_or(10);
        let c = comfy::Client::new(&base)?;
        c.wait_ready(Duration::from_secs(420), |p| eprintln!("{p}"))
            .await?;
        let seed = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .map(|d| d.as_secs())
            .unwrap_or(7);
        let img = c
            .render(
                comfy::RenderOpts {
                    prompt: &prompt,
                    ckpt: comfy::DEFAULT_CKPT,
                    width: w,
                    height: h,
                    steps,
                    seed,
                },
                |p| eprintln!("{p}"),
            )
            .await?;
        let bytes = c.download(&img).await?;
        tokio::fs::write(&out, &bytes).await?;
        println!("saved: {out}");
        return Ok(());
    }
    ui::run().await
}

fn help_text() -> String {
    format!(
        "ccti {} — ComfyUI TUI: agentic image chat with inline terminal rendering\n\n\
         Usage:\n  \
           ccti                              Start the TUI (needs a real terminal)\n  \
           ccti --render PROMPT [--out FILE] [--w N --h N --steps N]\n  \
           ccti --models                     List ComfyUI checkpoints\n  \
           ccti --mcp                        MCP stdio server (render_image tool)\n  \
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
