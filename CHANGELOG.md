# Changelog

## 0.1.6 — 2026-09-15

- Header bar (title, workspace tabs, agent state), multiple workspaces with
  isolated galleries/chats/settings, per-workspace image + chat AI display.
- F-key bar with submenus: model, size, steps, batch; F6/F7/F8 manage
  workspaces. Pixel-matched gaps from the live font metrics.

## 0.1.5 — 2026-09-15

- Breathing room: one-cell gaps plus outer margin between all panes; every
  box keeps its full frame and top title (provenance stroke restored).
- Render settings box (bottom left): size presets Fast/Balanced/Quality,
  steps and batch count, live-adjustable via `[` `]` `-` `+` `Tab` while the
  input line is empty; `/render` flags override, plain `/render` uses them.

## 0.1.4 — 2026-09-15

- Single-line seams: neighbouring panes share one border instead of drawing
  two; panel titles move to the remaining outer edge.

## 0.1.3 — 2026-09-15

- Correct line wrapping: word-aware, display-cell based (CJK counts double,
  umlauts single), no more mid-word breaks; fixes a panic on multibyte text
  past the length cap.

## 0.1.1 — 2026-09-14

`--help`/`-h` and `--version`/`-V` print usage instead of dropping into the
TUI (which needs a real terminal).

## 0.1.2 — 2026-09-15

- Provenance: every render stores a JSON sidecar plus an embedded
  human-readable `parameters` PNG chunk (ComfyUI's own `prompt` chunk is
  preserved); the TUI shows prompt, model, size, steps, seed and batch
  position in a bottom-left panel.
- Fast iteration defaults: 512x512, 8 steps. New batch size `n` (1-4) for
  variants across MCP, TUI `/render` and headless mode.
- Agent vision fallback: tool results point text-only models at `Read`ing
  the saved PNG instead of relying on attached images.

## 0.1.0 — 2026-09-14

Initial release: ComfyUI TUI with image-left/chat-right layout, ccht-backed
agent session (`opencode acp`), `render_image` MCP tool, direct `/render`
commands, image history navigation, headless `--render`/`--models` modes.
