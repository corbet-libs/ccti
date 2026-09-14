//! Generation provenance: sidecar JSON plus a human-readable PNG text chunk.
//!
//! The split mirrors photography: embedded data (~EXIF) travels inside the
//! file, the sidecar (~XMP) stays grep-able and survives hosts that strip
//! chunks. ComfyUI's own `prompt` chunk is always preserved untouched.

use std::path::{Path, PathBuf};

use anyhow::{Context, Result};
use serde::{Deserialize, Serialize};

/// Everything needed to reproduce a render and to credit it in the UI.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct Generation {
    pub prompt: String,
    pub negative: String,
    pub ckpt: String,
    pub width: u32,
    pub height: u32,
    pub steps: u32,
    pub cfg: f32,
    pub sampler: String,
    pub scheduler: String,
    pub seed: u64,
    /// Batch size requested...
    pub n: u32,
    /// ...and this image's index within it (0-based).
    pub index: u32,
    pub software: String,
    pub created_unix: u64,
}

impl Generation {
    /// Human-readable, A1111-style. Viewers that understand `parameters` show this.
    pub fn parameters_text(&self) -> String {
        format!(
            "{}\nNegative prompt: {}\nSteps: {}, Sampler: {}, Schedule: {}, CFG: {}, Seed: {}, Size: {}x{}, Model: {}, Batch: {}/{}, Software: {}",
            self.prompt,
            self.negative,
            self.steps,
            self.sampler,
            self.scheduler,
            self.cfg,
            self.seed,
            self.width,
            self.height,
            self.ckpt,
            self.index + 1,
            self.n,
            self.software,
        )
    }

    pub fn save_sidecar(&self, json_path: &Path) -> Result<()> {
        let text = serde_json::to_string_pretty(self)?;
        std::fs::write(json_path, text)
            .with_context(|| format!("write {}", json_path.display()))?;
        Ok(())
    }

    pub fn load_sidecar(json_path: &Path) -> Result<Generation> {
        let text = std::fs::read_to_string(json_path)
            .with_context(|| format!("read {}", json_path.display()))?;
        Ok(serde_json::from_str(&text)?)
    }

    /// Re-encode `png_bytes` with our `parameters` chunk added, preserving
    /// every chunk ComfyUI wrote (notably `prompt`). Never fails the image:
    /// on any error the input bytes come back untouched.
    pub fn embed_png(&self, png_bytes: &[u8]) -> Vec<u8> {
        match embed_fallible(png_bytes, &self.parameters_text()) {
            Ok(out) => out,
            Err(_) => png_bytes.to_vec(),
        }
    }
}

fn embed_fallible(src: &[u8], params: &str) -> Result<Vec<u8>> {
    const SIG: &[u8; 8] = b"\x89PNG\r\n\x1a\n";
    anyhow::ensure!(src.len() > 8 && &src[..8] == SIG, "not a PNG");
    // iTXt with compression flag 0 carries UTF-8 text verbatim.
    let mut data = b"parameters\0".to_vec();
    data.extend_from_slice(&[0, 0]); // compression flag + method
    data.extend_from_slice(b"\0"); // empty language tag
    data.extend_from_slice(b"\0"); // empty translated keyword
    data.extend_from_slice(params.as_bytes());
    let chunk = make_chunk(b"iTXt", &data);

    let mut out = Vec::with_capacity(src.len() + chunk.len());
    out.extend_from_slice(&src[..8]);
    let mut pos = 8;
    let mut inserted = false;
    // Text chunks conventionally precede IDAT, and readers (including this
    // crate's read_info) only surface pre-IDAT metadata. Insert before the
    // first IDAT; files without one are degenerate, take IEND as fallback.
    let mut first_idat_at: Option<usize> = None;
    let mut scan = 8;
    while scan + 8 <= src.len() {
        let len = u32::from_be_bytes(src[scan..scan + 4].try_into().unwrap()) as usize;
        let typ = &src[scan + 4..scan + 8];
        let end = scan + 8 + len + 4;
        anyhow::ensure!(end <= src.len(), "truncated PNG chunk");
        if typ == b"IDAT" && first_idat_at.is_none() {
            first_idat_at = Some(scan);
        }
        if typ == b"IEND" {
            break;
        }
        scan = end;
    }
    let at = first_idat_at.unwrap_or(scan);
    while pos + 8 <= src.len() {
        let len = u32::from_be_bytes(src[pos..pos + 4].try_into().unwrap()) as usize;
        let typ = &src[pos + 4..pos + 8];
        let end = pos + 8 + len + 4;
        anyhow::ensure!(end <= src.len(), "truncated PNG chunk");
        if pos == at && !inserted {
            out.extend_from_slice(&chunk);
            inserted = true;
        }
        out.extend_from_slice(&src[pos..end]);
        pos = end;
        if typ == b"IEND" {
            break;
        }
    }
    anyhow::ensure!(inserted, "no insertion point found");
    Ok(out)
}

fn make_chunk(typ: &[u8; 4], data: &[u8]) -> Vec<u8> {
    let mut v = Vec::with_capacity(12 + data.len());
    v.extend_from_slice(&(data.len() as u32).to_be_bytes());
    v.extend_from_slice(typ);
    v.extend_from_slice(data);
    v.extend_from_slice(&crc32(&[typ.as_slice(), data].concat()).to_be_bytes());
    v
}

fn crc32(data: &[u8]) -> u32 {
    // Runtime-generated IEEE table: small, obviously correct, no 256-entry literal.
    let mut table = [0u32; 256];
    for (i, slot) in table.iter_mut().enumerate() {
        let mut c = i as u32;
        for _ in 0..8 {
            c = if c & 1 == 1 {
                0xEDB88320 ^ (c >> 1)
            } else {
                c >> 1
            };
        }
        *slot = c;
    }
    let mut crc = 0xFFFF_FFFFu32;
    for b in data {
        crc = table[((crc ^ (*b as u32)) & 0xFF) as usize] ^ (crc >> 8);
    }
    crc ^ 0xFFFF_FFFF
}

/// Store one finished render: `<stem>.png` (with embedded `parameters`)
/// plus `<stem>.json` sidecar. Returns both paths.
pub fn store_rendered(
    dir: &Path,
    stem: &str,
    png_bytes: &[u8],
    generation: &Generation,
) -> Result<(PathBuf, PathBuf)> {
    std::fs::create_dir_all(dir).with_context(|| format!("mkdir {}", dir.display()))?;
    let png_path = dir.join(format!("{stem}.png"));
    let json_path = dir.join(format!("{stem}.json"));
    std::fs::write(&png_path, generation.embed_png(png_bytes))
        .with_context(|| format!("write {}", png_path.display()))?;
    generation.save_sidecar(&json_path)?;
    Ok((png_path, json_path))
}

#[cfg(test)]
mod tests {
    use super::*;

    fn sample() -> Generation {
        Generation {
            prompt: "a fox".into(),
            negative: "blurry".into(),
            ckpt: "m.safetensors".into(),
            width: 512,
            height: 512,
            steps: 8,
            cfg: 1.5,
            sampler: "euler".into(),
            scheduler: "normal".into(),
            seed: 42,
            n: 2,
            index: 1,
            software: "ccti 0.1.2".into(),
            created_unix: 1_700_000_000,
        }
    }

    fn tiny_png() -> Vec<u8> {
        let img = image::RgbImage::from_pixel(4, 4, image::Rgb([200, 100, 50]));
        let dyn_img = image::DynamicImage::ImageRgb8(img);
        let mut buf = Vec::new();
        let mut cur = std::io::Cursor::new(&mut buf);
        dyn_img
            .write_to(&mut cur, image::ImageFormat::Png)
            .expect("encode");
        buf
    }

    #[test]
    fn parameters_text_carries_everything_human() {
        let t = sample().parameters_text();
        for token in [
            "a fox",
            "blurry",
            "m.safetensors",
            "512x512",
            "8",
            "42",
            "euler",
            "2/2",
        ] {
            assert!(t.contains(token), "parameters missing {token}:\n{t}");
        }
    }

    #[test]
    fn sidecar_roundtrips() {
        let dir = std::env::temp_dir().join("ccti-prov-test");
        std::fs::create_dir_all(&dir).unwrap();
        let p = dir.join("x.json");
        sample().save_sidecar(&p).unwrap();
        assert_eq!(Generation::load_sidecar(&p).unwrap(), sample());
        std::fs::remove_file(p).ok();
    }

    #[test]
    fn embed_adds_parameters_and_keeps_existing_chunks() {
        // Seed a pre-existing chunk the way ComfyUI does.
        let raw = tiny_png();
        let mut enc_src = Vec::new();
        {
            let dec = png::Decoder::new(std::io::Cursor::new(raw.as_slice()));
            let mut r = dec.read_info().unwrap();
            let mut buf = vec![0; r.output_buffer_size().unwrap()];
            let info = r.next_frame(&mut buf).unwrap();
            let mut enc = png::Encoder::new(&mut enc_src, info.width, info.height);
            enc.set_color(info.color_type);
            enc.set_depth(info.bit_depth);
            enc.add_text_chunk("prompt".to_string(), "{\"1\":{}}".to_string())
                .unwrap();
            enc.write_header().unwrap().write_image_data(&buf).unwrap();
        }
        let out = sample().embed_png(&enc_src);
        // Splice must succeed (no silent fallback): list chunks on failure.
        let kinds: Vec<String> = {
            let mut v = Vec::new();
            let mut pos = 8;
            while pos + 8 <= out.len() {
                let len = u32::from_be_bytes(out[pos..pos + 4].try_into().unwrap()) as usize;
                v.push(String::from_utf8_lossy(&out[pos + 4..pos + 8]).into_owned());
                pos += 8 + len + 4;
                if v.len() > 20 {
                    break;
                }
            }
            v
        };
        assert!(
            kinds.contains(&"iTXt".to_string()),
            "no iTXt spliced, chunks: {kinds:?}"
        );
        // Read back: both chunks present, pixels intact.
        let dec = png::Decoder::new(std::io::Cursor::new(out.as_slice()));
        let mut r = dec.read_info().unwrap();
        let info = r.info();
        let get = |k: &str| {
            info.uncompressed_latin1_text
                .iter()
                .find(|t| t.keyword == k)
                .map(|t| t.text.clone())
                .or_else(|| {
                    info.utf8_text
                        .iter()
                        .find(|t| t.keyword == k)
                        .and_then(|t| t.get_text().ok())
                })
                .unwrap_or_default()
        };
        assert_eq!(get("prompt"), "{\"1\":{}}");
        assert!(
            get("parameters").contains("a fox"),
            "parameters chunk missing"
        );
        let mut buf = vec![0; r.output_buffer_size().unwrap()];
        r.next_frame(&mut buf).unwrap();
        let img = image::load_from_memory(&out).unwrap().to_rgb8();
        assert_eq!(img.get_pixel(0, 0), &image::Rgb([200, 100, 50]));
    }

    #[test]
    fn embed_never_loses_the_image() {
        let garbage = b"definitely not a png";
        assert_eq!(sample().embed_png(garbage), garbage);
    }

    #[test]
    fn store_writes_png_plus_sidecar() {
        let dir = std::env::temp_dir().join("ccti-store-test");
        let (png, json) = store_rendered(&dir, "s", &tiny_png(), &sample()).unwrap();
        assert!(png.exists() && json.exists());
        assert_eq!(Generation::load_sidecar(&json).unwrap(), sample());
        std::fs::remove_file(png).ok();
        std::fs::remove_file(json).ok();
    }
}
