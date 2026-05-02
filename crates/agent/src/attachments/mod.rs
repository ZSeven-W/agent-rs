//! Attachments — image / file uploads (Tier 4 / claude-code parity).
//!
//! Mirrors `services/attachments/`. Helpers for turning user-supplied
//! files (images, large text snippets, PDFs) into [`ContentBlock`]
//! values the provider can consume.
//!
//! Cross-provider details:
//!
//! - Anthropic accepts inline base64 image data via
//!   `ContentBlock::Image { source: ImageSource::Base64 { ... } }`.
//! - Anthropic + OpenAI also accept image URLs via
//!   `ImageSource::Url { url }`.
//! - For "large text" attachments (long file contents the user
//!   pastes), we fold them into a `ContentBlock::Text` with a clear
//!   `[file: <path>]` header so the model can identify the source.
//!
//! Functions here are pure — no I/O. Hosts read the file/buffer
//! themselves and call [`image_from_bytes`], [`image_from_url`], or
//! [`text_attachment`] to build the right block.

use serde::{Deserialize, Serialize};

use crate::message::{ContentBlock, ImageSource};

/// MIME types we recognise for image attachments.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
#[non_exhaustive]
pub enum ImageMime {
    Png,
    Jpeg,
    Gif,
    Webp,
}

impl ImageMime {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Png => "image/png",
            Self::Jpeg => "image/jpeg",
            Self::Gif => "image/gif",
            Self::Webp => "image/webp",
        }
    }

    /// Sniff the mime from the magic bytes at the start of the file.
    /// Returns `None` for unknown formats — the caller can fall back
    /// to extension-based detection or refuse.
    pub fn sniff(bytes: &[u8]) -> Option<Self> {
        if bytes.starts_with(&[0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a]) {
            return Some(Self::Png);
        }
        if bytes.starts_with(&[0xff, 0xd8, 0xff]) {
            return Some(Self::Jpeg);
        }
        if bytes.starts_with(b"GIF87a") || bytes.starts_with(b"GIF89a") {
            return Some(Self::Gif);
        }
        if bytes.len() >= 12 && &bytes[..4] == b"RIFF" && &bytes[8..12] == b"WEBP" {
            return Some(Self::Webp);
        }
        None
    }
}

#[derive(Debug, thiserror::Error)]
pub enum AttachmentError {
    #[error("unknown image format — magic bytes did not match png/jpeg/gif/webp")]
    UnknownImageFormat,
    #[error("attachment too large: {size_bytes} bytes (limit {limit})")]
    TooLarge { size_bytes: usize, limit: usize },
}

/// Maximum inline-base64 image size accepted by the Anthropic
/// Messages API. Files above this limit must be uploaded via the
/// Files API and referenced by id (not yet implemented).
pub const MAX_INLINE_IMAGE_BYTES: usize = 5 * 1024 * 1024;

/// Build an inline image content block from raw bytes. Sniffs the
/// mime type from the magic bytes; errors on unknown formats and
/// oversized inputs.
pub fn image_from_bytes(bytes: &[u8]) -> Result<ContentBlock, AttachmentError> {
    if bytes.len() > MAX_INLINE_IMAGE_BYTES {
        return Err(AttachmentError::TooLarge {
            size_bytes: bytes.len(),
            limit: MAX_INLINE_IMAGE_BYTES,
        });
    }
    let mime = ImageMime::sniff(bytes).ok_or(AttachmentError::UnknownImageFormat)?;
    Ok(ContentBlock::Image {
        source: ImageSource::Base64 {
            media_type: mime.as_str().to_string(),
            data: base64_encode(bytes),
        },
    })
}

/// Build a URL-source image content block. The URL is NOT validated
/// — the host is responsible for ensuring the provider can fetch it.
pub fn image_from_url(url: impl Into<String>) -> ContentBlock {
    ContentBlock::Image {
        source: ImageSource::Url { url: url.into() },
    }
}

/// Build a text content block representing a file attachment, with a
/// clear `[file: <path>]\n` prefix so the model can identify the
/// source. Useful for hosts that paste in large source files.
pub fn text_attachment(path: &str, content: &str) -> ContentBlock {
    let header = format!("[file: {path}]\n");
    ContentBlock::Text {
        text: format!("{header}{content}"),
    }
}

/// Standard base64 (RFC 4648) WITH padding, no line breaks. Inline
/// implementation to avoid pulling a base64 crate just for image
/// uploads.
fn base64_encode(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/";
    let mut out = String::with_capacity(bytes.len().div_ceil(3) * 4);
    let chunks = bytes.chunks(3);
    for chunk in chunks {
        let b0 = chunk[0];
        let b1 = chunk.get(1).copied().unwrap_or(0);
        let b2 = chunk.get(2).copied().unwrap_or(0);
        let n = ((b0 as u32) << 16) | ((b1 as u32) << 8) | (b2 as u32);
        out.push(ALPHABET[((n >> 18) & 0x3f) as usize] as char);
        out.push(ALPHABET[((n >> 12) & 0x3f) as usize] as char);
        if chunk.len() > 1 {
            out.push(ALPHABET[((n >> 6) & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(n & 0x3f) as usize] as char);
        } else {
            out.push('=');
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn sniff_png_magic_bytes() {
        let png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00];
        assert_eq!(ImageMime::sniff(&png), Some(ImageMime::Png));
    }

    #[test]
    fn sniff_jpeg_magic_bytes() {
        let jpeg = [0xff, 0xd8, 0xff, 0xe0];
        assert_eq!(ImageMime::sniff(&jpeg), Some(ImageMime::Jpeg));
    }

    #[test]
    fn sniff_gif_87_and_89() {
        assert_eq!(ImageMime::sniff(b"GIF87a..."), Some(ImageMime::Gif));
        assert_eq!(ImageMime::sniff(b"GIF89a..."), Some(ImageMime::Gif));
    }

    #[test]
    fn sniff_webp_riff_marker() {
        let mut webp = vec![0u8; 12];
        webp[0..4].copy_from_slice(b"RIFF");
        webp[8..12].copy_from_slice(b"WEBP");
        assert_eq!(ImageMime::sniff(&webp), Some(ImageMime::Webp));
    }

    #[test]
    fn sniff_unknown_returns_none() {
        assert_eq!(ImageMime::sniff(b"not an image"), None);
        assert_eq!(ImageMime::sniff(&[]), None);
    }

    #[test]
    fn image_from_bytes_builds_base64_block() {
        let png = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a, 0x00];
        let block = image_from_bytes(&png).unwrap();
        match block {
            ContentBlock::Image {
                source: ImageSource::Base64 { media_type, data },
            } => {
                assert_eq!(media_type, "image/png");
                assert!(!data.is_empty());
            }
            _ => panic!("expected Base64 image"),
        }
    }

    #[test]
    fn image_from_bytes_rejects_unknown_format() {
        let r = image_from_bytes(b"not an image");
        assert!(matches!(
            r.unwrap_err(),
            AttachmentError::UnknownImageFormat
        ));
    }

    #[test]
    fn image_from_bytes_rejects_oversized() {
        let png_header = [0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
        let mut data = png_header.to_vec();
        data.resize(MAX_INLINE_IMAGE_BYTES + 1, 0);
        let r = image_from_bytes(&data);
        assert!(matches!(
            r.unwrap_err(),
            AttachmentError::TooLarge { size_bytes, .. } if size_bytes > MAX_INLINE_IMAGE_BYTES
        ));
    }

    #[test]
    fn image_from_url_returns_url_source() {
        let block = image_from_url("https://example.com/x.png");
        match block {
            ContentBlock::Image {
                source: ImageSource::Url { url },
            } => {
                assert_eq!(url, "https://example.com/x.png");
            }
            _ => panic!("expected Url image"),
        }
    }

    #[test]
    fn text_attachment_includes_path_header() {
        let block = text_attachment("/tmp/x.rs", "fn main() {}");
        match block {
            ContentBlock::Text { text } => {
                assert!(text.starts_with("[file: /tmp/x.rs]"));
                assert!(text.contains("fn main() {}"));
            }
            _ => panic!("expected text"),
        }
    }

    #[test]
    fn base64_known_vectors() {
        assert_eq!(base64_encode(b""), "");
        assert_eq!(base64_encode(b"f"), "Zg==");
        assert_eq!(base64_encode(b"fo"), "Zm8=");
        assert_eq!(base64_encode(b"foo"), "Zm9v");
        assert_eq!(base64_encode(b"foob"), "Zm9vYg==");
        assert_eq!(base64_encode(b"fooba"), "Zm9vYmE=");
        assert_eq!(base64_encode(b"foobar"), "Zm9vYmFy");
    }

    #[test]
    fn image_mime_round_trip_serde() {
        for mime in [
            ImageMime::Png,
            ImageMime::Jpeg,
            ImageMime::Gif,
            ImageMime::Webp,
        ] {
            let json = serde_json::to_string(&mime).unwrap();
            let parsed: ImageMime = serde_json::from_str(&json).unwrap();
            assert_eq!(parsed, mime);
        }
    }
}
