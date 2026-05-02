//! Line-delimited JSON codec.
//!
//! Each frame is one JSON object terminated by `\n`. Mirrors the
//! Language Server Protocol's *minimal* style — no `Content-Length`
//! header — because for line-buffered stdio the simpler format is
//! sufficient and keeps both ends easier to debug with `cat`.
//!
//! For very large payloads (Files API, batched results), hosts can
//! switch to LSP-style headers; that codec lives outside this
//! module to keep both formats independent.

use super::protocol::RpcMessage;

#[derive(Debug, thiserror::Error)]
pub enum FrameError {
    #[error("frame parse: {0}")]
    Parse(String),
    #[error("frame encode: {0}")]
    Encode(String),
}

/// Try to decode one frame from the start of `buf`. On success
/// returns the message AND the number of bytes consumed (so the
/// caller can drain the buffer prefix). Returns `Ok(None)` when the
/// buffer doesn't yet contain a complete `\n`-terminated line.
pub fn decode_frame(buf: &[u8]) -> Result<Option<(RpcMessage, usize)>, FrameError> {
    let Some(eol) = buf.iter().position(|&b| b == b'\n') else {
        return Ok(None);
    };
    let line = &buf[..eol];
    let line = std::str::from_utf8(line)
        .map_err(|e| FrameError::Parse(format!("non-UTF-8 frame: {e}")))?;
    if line.trim().is_empty() {
        return Ok(Some((dummy_pong(), eol + 1)));
    }
    let msg: RpcMessage =
        serde_json::from_str(line).map_err(|e| FrameError::Parse(e.to_string()))?;
    Ok(Some((msg, eol + 1)))
}

/// Special-case empty-line shim — frames that are just whitespace
/// decode to a no-op ping notification. Avoids spamming hard-error
/// when a host sends a stray newline.
fn dummy_pong() -> RpcMessage {
    RpcMessage::Notification(super::protocol::RpcNotification::new("ping", None))
}

/// Encode a frame to bytes. Always terminates with `\n`. Strips any
/// embedded newlines from the JSON output to preserve the
/// one-message-per-line invariant.
pub fn encode_frame(msg: &RpcMessage) -> Result<Vec<u8>, FrameError> {
    let json = serde_json::to_string(msg).map_err(|e| FrameError::Encode(e.to_string()))?;
    if json.contains('\n') {
        return Err(FrameError::Encode(
            "encoded JSON contains a newline — would corrupt the line-delimited stream".into(),
        ));
    }
    let mut out = json.into_bytes();
    out.push(b'\n');
    Ok(out)
}

#[cfg(test)]
mod tests {
    use super::super::protocol::{
        method, RpcId, RpcMessage, RpcNotification, RpcRequest, RpcResponse,
    };
    use super::*;

    #[test]
    fn decode_returns_none_when_buffer_lacks_newline() {
        let r = decode_frame(b"{\"jsonrpc\":\"2.0\"").unwrap();
        assert!(r.is_none());
    }

    #[test]
    fn decode_consumes_one_frame_returns_remaining_offset() {
        let req_json = r#"{"jsonrpc":"2.0","id":1,"method":"ping"}"#;
        let mut buf = req_json.as_bytes().to_vec();
        buf.push(b'\n');
        let (msg, n) = decode_frame(&buf).unwrap().unwrap();
        assert_eq!(n, buf.len());
        assert!(matches!(msg, RpcMessage::Request(_)));
    }

    #[test]
    fn decode_two_frames_in_one_buffer() {
        let req1 = r#"{"jsonrpc":"2.0","id":1,"method":"a"}"#;
        let req2 = r#"{"jsonrpc":"2.0","id":2,"method":"b"}"#;
        let mut buf = req1.as_bytes().to_vec();
        buf.push(b'\n');
        buf.extend_from_slice(req2.as_bytes());
        buf.push(b'\n');
        let (m1, n1) = decode_frame(&buf).unwrap().unwrap();
        match m1 {
            RpcMessage::Request(r) => assert_eq!(r.method, "a"),
            _ => panic!(),
        }
        let (m2, n2) = decode_frame(&buf[n1..]).unwrap().unwrap();
        match m2 {
            RpcMessage::Request(r) => assert_eq!(r.method, "b"),
            _ => panic!(),
        }
        assert_eq!(n1 + n2, buf.len());
    }

    #[test]
    fn encode_appends_newline() {
        let msg = RpcMessage::Notification(RpcNotification::new(method::PING, None));
        let bytes = encode_frame(&msg).unwrap();
        assert_eq!(bytes.last(), Some(&b'\n'));
        let s = std::str::from_utf8(&bytes[..bytes.len() - 1]).unwrap();
        assert!(s.starts_with("{"));
    }

    #[test]
    fn encode_round_trip_with_decode() {
        let msg = RpcMessage::Request(RpcRequest::new(
            RpcId::Num(7),
            method::SEND,
            Some(serde_json::json!({"text": "hi"})),
        ));
        let bytes = encode_frame(&msg).unwrap();
        let (decoded, _) = decode_frame(&bytes).unwrap().unwrap();
        assert_eq!(decoded, msg);
    }

    #[test]
    fn decode_invalid_json_errors() {
        let r = decode_frame(b"not json\n");
        match r {
            Err(FrameError::Parse(_)) => {}
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn decode_blank_line_is_a_silent_ping() {
        let (msg, n) = decode_frame(b"\n").unwrap().unwrap();
        assert_eq!(n, 1);
        // Treated as a ping notification, not a hard error.
        assert!(matches!(msg, RpcMessage::Notification(ref n) if n.method == "ping"));
    }

    #[test]
    fn decode_non_utf8_errors() {
        let mut buf: Vec<u8> = vec![0xff, 0xfe];
        buf.push(b'\n');
        match decode_frame(&buf) {
            Err(FrameError::Parse(_)) => {}
            other => panic!("expected Parse error, got {other:?}"),
        }
    }

    #[test]
    fn encode_response_round_trip() {
        let msg = RpcMessage::Response(RpcResponse::ok(
            RpcId::Str("x".into()),
            serde_json::json!({"ok": true}),
        ));
        let bytes = encode_frame(&msg).unwrap();
        let (decoded, _) = decode_frame(&bytes).unwrap().unwrap();
        assert_eq!(decoded, msg);
    }
}
