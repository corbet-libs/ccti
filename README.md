# ccti — ComfyUI TUI (image left, chat right)

MVP. Chat on the right (ccht session against `opencode acp`), rendered
image on the left via `ratatui-image` (protocol auto-detect: Kitty where
available, Sixel in foot/cterm, halfblocks fallback).

## Run

```bash
cargo run                                  # TUI (needs a real terminal)
cargo run -- --render "a lighthouse"       # headless render -> ~/images/generated/
cargo run -- --models                      # list checkpoints
cargo run -- --mcp                         # MCP stdio server (used by the TUI's agent)
```

TUI keys: text = agent prompt · `/render TEXT [--w N --h N --steps N]` ·
`/models` · `/cancel` · `←/→` image history · `/quit`. `Esc` cancels a turn.

## Notes

- ComfyUI base URL: `https://comfyui.corbet.ch` (override: `CCTI_COMFY_URL`).
- Cold start (~2 min Sablier wake) is handled inside every path.
- Permission policy (MVP): permission requests are auto-allowed once and
  announced in chat; watch the TUI and `/cancel` turns. Tighten before any
  untrusted use.
- Agent workspace: directory the TUI was started in (`CCTI_WORKDIR` overrides).
