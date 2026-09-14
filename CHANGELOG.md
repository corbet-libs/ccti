# Changelog

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
