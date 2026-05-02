//! Size-aware attachment helpers.
//!
//! These helpers wrap a host-supplied [`FilesClient`] so callers
//! don't have to decide between "inline base64" and "upload + file
//! ref" manually:
//!
//! - [`image_smart`]: inline base64 when raw bytes ≤
//!   [`super::MAX_INLINE_IMAGE_BYTES`] (5 MiB), otherwise upload
//!   via Files API and produce an `ImageSource::File` block.
//! - [`pdf_attachment`]: same threshold pattern with
//!   [`MAX_INLINE_PDF_BYTES`] (8 MiB raw). Hosts can pass
//!   `force_upload = true` to skip the inline path even for small
//!   PDFs (e.g., when the same PDF is reused across many turns).
//! - [`text_attachment_via_files`]: always uploads — useful for
//!   multi-megabyte logs / transcripts where token-window pressure
//!   matters.
//!
//! All three return a ready-to-use [`ContentBlock`] plus an
//! [`AttachmentDestination`] indicating whether an upload happened
//! (so the host can persist the id for later cleanup).

use super::files::{FilesClient, FilesError, UploadedFile};
use super::{ImageMime, MAX_INLINE_IMAGE_BYTES};
use crate::message::{ContentBlock, DocumentSource, ImageSource};

/// Maximum inline-base64 PDF size we'll attempt before forcing the
/// Files API path. Anthropic's `application/pdf` document blocks
/// accept inline-base64 up to ~32 MiB but the wire payload doubles
/// once base64-encoded, so the practical sweet spot for "ship it
/// inline" is ~8 MiB. Anything bigger goes through Files.
pub const MAX_INLINE_PDF_BYTES: usize = 8 * 1024 * 1024;

/// Where the smart helper put the bytes — useful for logging and
/// for hosts that want to know if an upload occurred.
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum AttachmentDestination {
    Inline,
    Uploaded(UploadedFile),
}

impl AttachmentDestination {
    pub fn uploaded_file(&self) -> Option<&UploadedFile> {
        match self {
            Self::Inline => None,
            Self::Uploaded(f) => Some(f),
        }
    }
}

/// Build an image content block, choosing inline or uploaded form
/// based on size. Sniffs the mime type from the magic bytes; errors
/// on unknown formats.
///
/// The threshold is [`super::MAX_INLINE_IMAGE_BYTES`] (5 MiB) — the
/// largest payload Anthropic accepts inline. Larger images upload
/// via `client` and produce an `ImageSource::File` block.
pub async fn image_smart<C: FilesClient + ?Sized>(
    client: &C,
    bytes: Vec<u8>,
    filename: Option<&str>,
) -> Result<(ContentBlock, AttachmentDestination), SmartAttachmentError> {
    let mime = ImageMime::sniff(&bytes).ok_or(SmartAttachmentError::UnknownImageFormat)?;
    if bytes.len() <= MAX_INLINE_IMAGE_BYTES {
        let block = ContentBlock::Image {
            source: ImageSource::Base64 {
                media_type: mime.as_str().to_string(),
                data: super::base64_encode(&bytes),
            },
        };
        return Ok((block, AttachmentDestination::Inline));
    }
    let uploaded = client
        .upload(bytes, mime.as_str(), filename)
        .await
        .map_err(SmartAttachmentError::Files)?;
    let block = ContentBlock::Image {
        source: ImageSource::File {
            file_id: uploaded.id.clone(),
        },
    };
    Ok((block, AttachmentDestination::Uploaded(uploaded)))
}

/// Build a PDF document attachment. PDFs ≤ [`MAX_INLINE_PDF_BYTES`]
/// can ride inline as base64; larger ones upload via the Files API.
/// Hosts can force-upload by passing `force_upload = true` (e.g.,
/// when the same PDF will be reused across many turns and they
/// don't want to repeat the base64 cost).
pub async fn pdf_attachment<C: FilesClient + ?Sized>(
    client: &C,
    bytes: Vec<u8>,
    filename: Option<&str>,
    force_upload: bool,
) -> Result<(ContentBlock, AttachmentDestination), SmartAttachmentError> {
    if !looks_like_pdf(&bytes) {
        return Err(SmartAttachmentError::NotAPdf);
    }
    if !force_upload && bytes.len() <= MAX_INLINE_PDF_BYTES {
        let block = ContentBlock::Document {
            source: DocumentSource::Base64 {
                media_type: "application/pdf".to_string(),
                data: super::base64_encode(&bytes),
            },
        };
        return Ok((block, AttachmentDestination::Inline));
    }
    let uploaded = client
        .upload(bytes, "application/pdf", filename)
        .await
        .map_err(SmartAttachmentError::Files)?;
    let block = ContentBlock::Document {
        source: DocumentSource::File {
            file_id: uploaded.id.clone(),
        },
    };
    Ok((block, AttachmentDestination::Uploaded(uploaded)))
}

/// Upload a long text payload as a `text/plain` Files object and
/// return a `Document` block referencing it. Useful when a host
/// wants to attach a multi-megabyte log/transcript without burning
/// the token window — the model fetches the file server-side.
///
/// Hosts that want the legacy `[file: <path>]\n<contents>` text
/// embedding should keep using [`super::text_attachment`] —
/// uploading a tiny snippet to the Files API is wasteful.
pub async fn text_attachment_via_files<C: FilesClient + ?Sized>(
    client: &C,
    text: &str,
    filename: Option<&str>,
) -> Result<(ContentBlock, UploadedFile), FilesError> {
    let bytes = text.as_bytes().to_vec();
    let uploaded = client.upload(bytes, "text/plain", filename).await?;
    let block = ContentBlock::Document {
        source: DocumentSource::File {
            file_id: uploaded.id.clone(),
        },
    };
    Ok((block, uploaded))
}

#[derive(Debug, thiserror::Error)]
#[non_exhaustive]
pub enum SmartAttachmentError {
    #[error("unknown image format — magic bytes did not match png/jpeg/gif/webp")]
    UnknownImageFormat,
    #[error("payload is not a PDF (missing %PDF- header)")]
    NotAPdf,
    #[error(transparent)]
    Files(#[from] FilesError),
}

/// Recognise a PDF payload tolerantly: scan the first 1 KiB for the
/// `%PDF-` signature after optionally skipping a UTF-8/16 BOM and
/// any leading whitespace. Some authoring tools prepend metadata or
/// a BOM before the PDF header, and the spec allows up to ~1024
/// bytes of preamble before the signature.
fn looks_like_pdf(bytes: &[u8]) -> bool {
    let needle = b"%PDF-";
    let limit = bytes.len().min(1024);
    bytes[..limit].windows(needle.len()).any(|w| w == needle)
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::attachments::files::InMemoryFilesClient;

    fn png_bytes(extra: usize) -> Vec<u8> {
        let mut v = vec![0x89, 0x50, 0x4e, 0x47, 0x0d, 0x0a, 0x1a, 0x0a];
        v.extend(std::iter::repeat(0u8).take(extra));
        v
    }

    fn pdf_bytes(extra: usize) -> Vec<u8> {
        let mut v = b"%PDF-1.7\n".to_vec();
        v.extend(std::iter::repeat(0u8).take(extra));
        v
    }

    #[tokio::test]
    async fn small_image_stays_inline() {
        let client = InMemoryFilesClient::new();
        let bytes = png_bytes(1024); // ~1 KiB
        let (block, dest) = image_smart(&client, bytes, None).await.unwrap();
        assert!(matches!(dest, AttachmentDestination::Inline));
        assert_eq!(client.upload_count(), 0);
        match block {
            ContentBlock::Image {
                source: ImageSource::Base64 { media_type, data },
            } => {
                assert_eq!(media_type, "image/png");
                assert!(!data.is_empty());
            }
            _ => panic!("expected inline base64"),
        }
    }

    #[tokio::test]
    async fn large_image_uploads_and_returns_file_id() {
        let client = InMemoryFilesClient::new();
        // > 5 MiB triggers upload.
        let bytes = png_bytes(MAX_INLINE_IMAGE_BYTES + 1);
        let (block, dest) = image_smart(&client, bytes, Some("big.png")).await.unwrap();
        let uploaded = match &dest {
            AttachmentDestination::Uploaded(u) => u,
            _ => panic!("expected upload"),
        };
        assert_eq!(uploaded.mime, "image/png");
        assert_eq!(uploaded.filename.as_deref(), Some("big.png"));
        match block {
            ContentBlock::Image {
                source: ImageSource::File { file_id },
            } => {
                assert_eq!(file_id, uploaded.id);
            }
            _ => panic!("expected file ref"),
        }
        assert_eq!(client.upload_count(), 1);
    }

    #[tokio::test]
    async fn unknown_image_format_short_circuits() {
        let client = InMemoryFilesClient::new();
        let err = image_smart(&client, b"not an image".to_vec(), None)
            .await
            .expect_err("should fail");
        assert!(matches!(err, SmartAttachmentError::UnknownImageFormat));
        assert_eq!(client.upload_count(), 0);
    }

    #[tokio::test]
    async fn small_pdf_stays_inline() {
        let client = InMemoryFilesClient::new();
        let bytes = pdf_bytes(2048);
        let (block, dest) = pdf_attachment(&client, bytes, None, false).await.unwrap();
        assert!(matches!(dest, AttachmentDestination::Inline));
        assert_eq!(client.upload_count(), 0);
        match block {
            ContentBlock::Document {
                source: DocumentSource::Base64 { media_type, .. },
            } => {
                assert_eq!(media_type, "application/pdf");
            }
            _ => panic!("expected inline base64 pdf"),
        }
    }

    #[tokio::test]
    async fn large_pdf_uploads() {
        let client = InMemoryFilesClient::new();
        let bytes = pdf_bytes(MAX_INLINE_PDF_BYTES + 1);
        let (block, dest) = pdf_attachment(&client, bytes, Some("doc.pdf"), false)
            .await
            .unwrap();
        let uploaded = dest.uploaded_file().expect("uploaded");
        assert_eq!(uploaded.mime, "application/pdf");
        match block {
            ContentBlock::Document {
                source: DocumentSource::File { file_id },
            } => {
                assert_eq!(file_id, uploaded.id);
            }
            _ => panic!("expected file ref"),
        }
        assert_eq!(client.upload_count(), 1);
    }

    #[tokio::test]
    async fn pdf_force_upload_skips_inline_path() {
        let client = InMemoryFilesClient::new();
        let bytes = pdf_bytes(1024); // tiny but force upload anyway.
        let (_, dest) = pdf_attachment(&client, bytes, None, true).await.unwrap();
        assert!(matches!(dest, AttachmentDestination::Uploaded(_)));
        assert_eq!(client.upload_count(), 1);
    }

    #[tokio::test]
    async fn pdf_with_leading_bom_still_recognized() {
        let client = InMemoryFilesClient::new();
        let mut bytes = vec![0xef, 0xbb, 0xbf]; // UTF-8 BOM
        bytes.extend_from_slice(b"%PDF-1.4\n");
        bytes.extend(std::iter::repeat(0u8).take(2048));
        let (_, dest) = pdf_attachment(&client, bytes, None, false).await.unwrap();
        assert!(matches!(dest, AttachmentDestination::Inline));
    }

    #[tokio::test]
    async fn pdf_with_leading_whitespace_still_recognized() {
        let client = InMemoryFilesClient::new();
        let mut bytes = b"   \n\t".to_vec();
        bytes.extend_from_slice(b"%PDF-1.4\n");
        bytes.extend(std::iter::repeat(0u8).take(2048));
        let (_, dest) = pdf_attachment(&client, bytes, None, false).await.unwrap();
        assert!(matches!(dest, AttachmentDestination::Inline));
    }

    #[tokio::test]
    async fn pdf_rejects_non_pdf_payload() {
        let client = InMemoryFilesClient::new();
        let err = pdf_attachment(&client, b"not a pdf".to_vec(), None, false)
            .await
            .expect_err("should fail");
        assert!(matches!(err, SmartAttachmentError::NotAPdf));
        assert_eq!(client.upload_count(), 0);
    }

    #[tokio::test]
    async fn text_attachment_via_files_uploads_plain_text() {
        let client = InMemoryFilesClient::new();
        let big = "x".repeat(50_000);
        let (block, uploaded) = text_attachment_via_files(&client, &big, Some("log.txt"))
            .await
            .unwrap();
        assert_eq!(uploaded.mime, "text/plain");
        assert_eq!(uploaded.size_bytes, big.len() as u64);
        match block {
            ContentBlock::Document {
                source: DocumentSource::File { file_id },
            } => {
                assert_eq!(file_id, uploaded.id);
            }
            _ => panic!("expected document file ref"),
        }
    }
}
