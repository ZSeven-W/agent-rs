//! `WebFetch` — HTTP GET → text/markdown.
//!
//! ReadOnly tool. Fetches a single URL via `reqwest`, caps the
//! response size, and returns either the raw text body or a
//! best-effort HTML→plain-text conversion (`format = "text"`).
//!
//! No HTML→Markdown converter dep — that's a 1+ MB tree shake. The
//! built-in HTML stripper handles `<script>` / `<style>` removal +
//! tag elimination, which is the 80%-case for "model wants to read
//! a docs page". Hosts that need richer extraction can swap in
//! their own `WebFetch` impl.
//!
//! Honors `ctx.abort` via `tokio::select!`. Caps response at 5 MiB
//! by default; hard ceiling 50 MiB regardless of caller request.
//!
//! # SSRF defense
//!
//! By default the tool refuses URLs that resolve to **loopback,
//! private RFC 1918, link-local (incl. cloud metadata at
//! 169.254.169.254), unique-local IPv6, or IPv4-mapped versions of
//! any of the above (`::ffff:127.0.0.1` etc.)**. For each hop we:
//!
//! 1. Resolve the host (`tokio::net::lookup_host`) and reject if any
//!    returned IP falls in a blocked range.
//! 2. **Pin** the resolved IP into the request via reqwest's
//!    `Client::builder().resolve()` so the connect-time DNS lookup
//!    can't swap to a private address (DNS rebinding mitigation).
//! 3. Disable reqwest auto-redirects (`Policy::none()`) and walk the
//!    redirect chain manually, re-applying steps 1–2 on each
//!    `Location:` header, capped at `MAX_REDIRECTS` hops.
//!
//! Hosts that legitimately need to hit private networks (intranet
//! docs, dev servers) pass `allow_private_networks: true`, which
//! bypasses both the IP screen and the per-hop pinning (the host's
//! injected `with_client` is then used).

use std::net::{IpAddr, SocketAddr};
use std::time::{Duration, Instant};

use agent::error::AgentError;
use agent::tool::{SafetyClass, Tool, ToolUseContext};
use async_trait::async_trait;
use reqwest::redirect::Policy;
use serde::Deserialize;
use serde_json::json;
use tokio::time::timeout;

const DEFAULT_TIMEOUT_SECS: u64 = 30;
const MAX_TIMEOUT_SECS: u64 = 120;
const DEFAULT_MAX_BYTES: u64 = 5 * 1024 * 1024;
/// Hard ceiling on `max_bytes` regardless of caller request.
/// Stops the model from asking for a 50 GB body and OOMing the host.
const HARD_MAX_BYTES: u64 = 50 * 1024 * 1024;
/// `reqwest` follows up to 10 redirects by default; we cap at 5 to
/// limit the worst-case latency for misbehaving URL chains.
const MAX_REDIRECTS: usize = 5;

#[derive(Debug)]
pub struct WebFetchTool {
    /// Used only when `allow_private_networks: true`. The default
    /// secure path builds a fresh client per hop with the resolved
    /// IP pinned via `Client::builder().resolve()`.
    fallback_client: reqwest::Client,
}

impl Default for WebFetchTool {
    fn default() -> Self {
        Self::new()
    }
}

impl WebFetchTool {
    pub fn new() -> Self {
        let client = reqwest::Client::builder()
            .redirect(Policy::limited(MAX_REDIRECTS))
            .user_agent(format!("agent-tools-code/{}", env!("CARGO_PKG_VERSION")))
            .build()
            .unwrap_or_else(|_| reqwest::Client::new());
        Self {
            fallback_client: client,
        }
    }

    /// Inject a custom `reqwest::Client` — useful for hosts that
    /// want to add a proxy, a corporate root CA, or a custom
    /// timeout. **Used only on the `allow_private_networks: true`
    /// path**; the SSRF-guarded path builds its own per-hop client
    /// so it can pin resolved IPs and disable auto-redirects. Hosts
    /// that need both a custom transport AND SSRF guarding should
    /// implement their own `WebFetch` `Tool`.
    pub fn with_client(client: reqwest::Client) -> Self {
        Self {
            fallback_client: client,
        }
    }
}

#[derive(Debug, Deserialize, Default, PartialEq, Eq)]
#[serde(rename_all = "lowercase")]
enum WebFormat {
    #[default]
    Text,
    Html,
}

#[derive(Debug, Deserialize)]
struct WebFetchInput {
    url: String,
    /// Output format. `text` (default) strips HTML. `html` returns
    /// raw bytes as UTF-8.
    #[serde(default)]
    format: WebFormat,
    #[serde(default)]
    timeout_secs: Option<u64>,
    #[serde(default)]
    max_bytes: Option<u64>,
    /// When `false` (default), URLs that resolve to loopback /
    /// private / link-local IPs are rejected to defend against
    /// SSRF. Hosts that legitimately need intranet access set
    /// this to `true`.
    #[serde(default)]
    allow_private_networks: bool,
}

#[async_trait]
impl Tool for WebFetchTool {
    fn name(&self) -> &str {
        "WebFetch"
    }
    fn description(&self) -> &str {
        "GET a URL. Returns text or raw HTML. Capped at 5 MiB / 30s by default; max 120s timeout."
    }
    fn input_schema(&self) -> serde_json::Value {
        json!({
            "type": "object",
            "properties": {
                "url": {"type": "string"},
                "format": {"type": "string", "enum": ["text", "html"], "default": "text"},
                "timeout_secs": {"type": "integer", "minimum": 1, "maximum": MAX_TIMEOUT_SECS},
                "max_bytes": {"type": "integer", "minimum": 1024, "maximum": HARD_MAX_BYTES},
                "allow_private_networks": {"type": "boolean", "default": false}
            },
            "required": ["url"]
        })
    }
    fn safety_class(&self) -> SafetyClass {
        SafetyClass::ReadOnly
    }
    async fn call(
        &self,
        ctx: &ToolUseContext,
        input: serde_json::Value,
    ) -> Result<serde_json::Value, AgentError> {
        let parsed: WebFetchInput = serde_json::from_value(input)
            .map_err(|e| AgentError::other(format!("WebFetch invalid input: {e}")))?;
        let url = parsed.url.trim().to_string();
        if url.is_empty() {
            return Err(AgentError::other("WebFetch url must be non-empty"));
        }
        // Refuse non-http(s) schemes — `file:`, `data:`,
        // `javascript:` etc. don't belong in a model-driven fetch.
        if !(url.starts_with("http://") || url.starts_with("https://")) {
            return Err(AgentError::other(format!(
                "WebFetch only supports http(s) URLs; got '{url}'"
            )));
        }
        let timeout_secs = parsed
            .timeout_secs
            .unwrap_or(DEFAULT_TIMEOUT_SECS)
            .clamp(1, MAX_TIMEOUT_SECS);
        let max_bytes = parsed
            .max_bytes
            .unwrap_or(DEFAULT_MAX_BYTES)
            .clamp(1024, HARD_MAX_BYTES);
        let abort = ctx.abort.clone();

        // Walk redirects ourselves so we can re-validate every hop's
        // host against the SSRF guard and pin its resolved IP. When
        // the caller has opted out of the guard we fall back to the
        // shared client which honors auto-redirects normally.
        let response = if parsed.allow_private_networks {
            let request = self
                .fallback_client
                .get(&url)
                .timeout(Duration::from_secs(timeout_secs));
            tokio::select! {
                biased;
                _ = abort.cancelled() => {
                    return Err(AgentError::Aborted(
                        abort.reason().unwrap_or_else(|| "aborted".into()),
                    ));
                }
                r = timeout(Duration::from_secs(timeout_secs + 5), request.send()) => {
                    match r {
                        Ok(Ok(resp)) => resp,
                        Ok(Err(e)) => return Err(AgentError::other(format!("WebFetch '{url}' failed: {e}"))),
                        Err(_) => return Err(AgentError::other(format!(
                            "WebFetch '{url}' timed out after {timeout_secs}s"
                        ))),
                    }
                }
            }
        } else {
            fetch_with_pinned_redirects(&url, timeout_secs, &abort).await?
        };

        let status = response.status().as_u16();
        let final_url = response.url().to_string();
        let content_type = response
            .headers()
            .get(reqwest::header::CONTENT_TYPE)
            .and_then(|v| v.to_str().ok())
            .map(|s| s.to_string());

        // Stream body chunks so we can enforce max_bytes mid-stream
        // rather than allocating up to the full server response.
        let mut bytes: Vec<u8> = Vec::with_capacity(8 * 1024);
        let mut truncated = false;
        let mut stream = response.bytes_stream();
        use futures::StreamExt;
        loop {
            tokio::select! {
                biased;
                _ = abort.cancelled() => {
                    return Err(AgentError::Aborted(
                        abort.reason().unwrap_or_else(|| "aborted".into()),
                    ));
                }
                next = stream.next() => {
                    match next {
                        None => break,
                        Some(Ok(chunk)) => {
                            if (bytes.len() as u64).saturating_add(chunk.len() as u64) > max_bytes {
                                let remaining = (max_bytes - bytes.len() as u64) as usize;
                                bytes.extend_from_slice(&chunk[..remaining.min(chunk.len())]);
                                truncated = true;
                                break;
                            }
                            bytes.extend_from_slice(&chunk);
                        }
                        Some(Err(e)) => {
                            return Err(AgentError::other(format!(
                                "WebFetch '{final_url}' body read failed: {e}"
                            )));
                        }
                    }
                }
            }
        }

        let raw = String::from_utf8_lossy(&bytes).into_owned();
        let body = match parsed.format {
            WebFormat::Html => raw,
            WebFormat::Text => html_to_text(&raw),
        };

        Ok(json!({
            "url": url,
            "final_url": final_url,
            "status": status,
            "content_type": content_type,
            "body": body,
            "truncated": truncated,
            "size_bytes": bytes.len(),
        }))
    }
}

/// Resolve a URL's host and reject if any returned IP is blocked.
/// Returns `(host, port, addrs)` so the caller can pin those exact
/// addresses into reqwest and skip the connect-time DNS lookup.
async fn resolve_and_validate(url: &str) -> Result<(String, u16, Vec<SocketAddr>), AgentError> {
    let (host, port) = host_port_of(url)?;
    let addrs: Vec<SocketAddr> = tokio::net::lookup_host(format!("{host}:{port}"))
        .await
        .map_err(|e| AgentError::other(format!("WebFetch DNS lookup for '{host}' failed: {e}")))?
        .collect();
    if addrs.is_empty() {
        return Err(AgentError::other(format!(
            "WebFetch '{url}' resolved to no addresses"
        )));
    }
    for addr in &addrs {
        if is_blocked_ip(addr.ip()) {
            return Err(AgentError::other(format!(
                "WebFetch '{url}' refused: host resolves to a private/loopback/link-local address ({}). Pass allow_private_networks=true to override.",
                addr.ip()
            )));
        }
    }
    Ok((host, port, addrs))
}

fn host_port_of(url: &str) -> Result<(String, u16), AgentError> {
    if let Ok(parsed) = url::Url::parse(url) {
        let host = parsed
            .host_str()
            .ok_or_else(|| AgentError::other(format!("WebFetch '{url}' has no host")))?
            .to_string();
        let port = parsed.port_or_known_default().unwrap_or(443);
        return Ok((host, port));
    }
    // Fallback parser — `url` crate refuses some weird-but-valid-ish
    // shapes; the earlier scheme check ensures this is http(s).
    let after_scheme = url
        .strip_prefix("http://")
        .or_else(|| url.strip_prefix("https://"))
        .ok_or_else(|| AgentError::other(format!("WebFetch '{url}' has no scheme")))?;
    let host_port = after_scheme
        .split(['/', '?', '#'])
        .next()
        .unwrap_or(after_scheme);
    let (host, port) = match host_port.rsplit_once(':') {
        Some((h, p)) => (h.to_string(), p.parse().unwrap_or(443)),
        None => (host_port.to_string(), 443),
    };
    Ok((host, port))
}

/// Fetch a URL with the SSRF guard applied at every redirect hop.
/// Disables reqwest auto-redirects, then for each hop:
/// 1. Resolves + validates the host's IP(s).
/// 2. Builds a fresh `reqwest::Client` with `resolve_to_addrs()`
///    pinning **all** pre-validated addresses in one call so multi-
///    address (dual-stack) responses survive intact, defeating
///    connect-time DNS rebinding.
/// 3. Sends the GET; on 30x, parses `Location:` and loops, but the
///    total wall-time is bounded by a single shared deadline so a
///    chain of slow hops can't bypass the caller's timeout.
async fn fetch_with_pinned_redirects(
    initial_url: &str,
    timeout_secs: u64,
    abort: &agent::abort::AbortController,
) -> Result<reqwest::Response, AgentError> {
    let total_budget = Duration::from_secs(timeout_secs);
    let deadline = Instant::now() + total_budget;
    let mut current = initial_url.to_string();
    let mut hops = 0usize;
    loop {
        let remaining = deadline.saturating_duration_since(Instant::now());
        if remaining.is_zero() {
            return Err(AgentError::other(format!(
                "WebFetch '{initial_url}' timed out after {timeout_secs}s (across {hops} redirect hop(s))"
            )));
        }

        let (host, _port, addrs) = resolve_and_validate(&current).await?;
        // Single `resolve_to_addrs` call so all pre-validated IPs end
        // up in the dns_overrides entry — `resolve()` per-addr in a
        // loop would overwrite each prior pin, leaving only the last.
        let client = reqwest::Client::builder()
            .redirect(Policy::none())
            .user_agent(format!("agent-tools-code/{}", env!("CARGO_PKG_VERSION")))
            .resolve_to_addrs(&host, &addrs)
            .build()
            .map_err(|e| AgentError::other(format!("WebFetch client build failed: {e}")))?;

        // Per-hop request timeout = remaining budget. We add a small
        // grace to the outer `tokio::time::timeout` so the inner
        // request timeout fires first with a clearer error.
        let request = client.get(&current).timeout(remaining);
        let outer = remaining + Duration::from_secs(5);

        let resp = tokio::select! {
            biased;
            _ = abort.cancelled() => {
                return Err(AgentError::Aborted(
                    abort.reason().unwrap_or_else(|| "aborted".into()),
                ));
            }
            r = timeout(outer, request.send()) => {
                match r {
                    Ok(Ok(resp)) => resp,
                    Ok(Err(e)) => return Err(AgentError::other(format!("WebFetch '{current}' failed: {e}"))),
                    Err(_) => return Err(AgentError::other(format!(
                        "WebFetch '{current}' timed out (budget exhausted)"
                    ))),
                }
            }
        };

        let status = resp.status();
        if status.is_redirection() {
            if hops >= MAX_REDIRECTS {
                return Err(AgentError::other(format!(
                    "WebFetch '{initial_url}' exceeded {MAX_REDIRECTS} redirect hops"
                )));
            }
            let location = match resp
                .headers()
                .get(reqwest::header::LOCATION)
                .and_then(|v| v.to_str().ok())
                .map(|s| s.to_string())
            {
                Some(loc) => loc,
                None => return Ok(resp), // 30x without Location: surface as-is.
            };
            let base = url::Url::parse(&current).map_err(|e| {
                AgentError::other(format!("WebFetch redirect base parse failed: {e}"))
            })?;
            let next = base.join(&location).map_err(|e| {
                AgentError::other(format!(
                    "WebFetch redirect '{location}' from '{current}' parse failed: {e}"
                ))
            })?;
            let scheme = next.scheme();
            if scheme != "http" && scheme != "https" {
                return Err(AgentError::other(format!(
                    "WebFetch redirect '{next}' from '{current}' uses non-http scheme"
                )));
            }
            current = next.to_string();
            hops += 1;
            continue;
        }
        return Ok(resp);
    }
}

/// `true` for IPs that should be blocked by default — loopback,
/// RFC 1918 private, IPv4 link-local (incl. cloud metadata
/// 169.254.169.254), unspecified, IPv6 unique-local / link-local,
/// and **IPv4-mapped IPv6** versions of any blocked IPv4 (e.g.
/// `::ffff:127.0.0.1` resolves to IPv4 127.0.0.1 and is blocked).
pub(crate) fn is_blocked_ip(ip: IpAddr) -> bool {
    match ip {
        IpAddr::V4(v4) => {
            v4.is_loopback()
                || v4.is_private()
                || v4.is_link_local()
                || v4.is_unspecified()
                || v4.is_broadcast()
                || v4.is_multicast()
        }
        IpAddr::V6(v6) => {
            // ::ffff:a.b.c.d → recurse so the IPv4 ranges apply.
            if let Some(v4) = v6.to_ipv4_mapped() {
                return is_blocked_ip(IpAddr::V4(v4));
            }
            v6.is_loopback()
                || v6.is_unspecified()
                || v6.is_multicast()
                // Unique local fc00::/7
                || (v6.segments()[0] & 0xfe00) == 0xfc00
                // Link-local fe80::/10
                || (v6.segments()[0] & 0xffc0) == 0xfe80
        }
    }
}

/// Cheap HTML→text. Strips `<script>` / `<style>` block contents,
/// then drops every remaining tag. Decodes the four common HTML
/// entities (`&lt; &gt; &amp; &quot;`) plus numeric entities up to
/// U+10FFFF. Not a full parser — but good enough for letting a
/// model read documentation / README / blog content without
/// dragging in a 1+ MiB HTML parser dep.
pub(crate) fn html_to_text(html: &str) -> String {
    let stripped = strip_block(html, "script");
    let stripped = strip_block(&stripped, "style");
    let stripped = strip_block(&stripped, "noscript");
    let mut out = String::with_capacity(stripped.len());
    let mut in_tag = false;
    for ch in stripped.chars() {
        match ch {
            '<' => in_tag = true,
            '>' => in_tag = false,
            _ if !in_tag => out.push(ch),
            _ => {}
        }
    }
    decode_entities(&out)
        .lines()
        .map(|l| l.trim_end().to_string())
        .filter(|l| !l.is_empty())
        .collect::<Vec<_>>()
        .join("\n")
}

fn strip_block(html: &str, tag: &str) -> String {
    let open_lower = format!("<{}", tag.to_ascii_lowercase());
    let close_lower = format!("</{}>", tag.to_ascii_lowercase());
    let lower = html.to_ascii_lowercase();
    let mut out = String::with_capacity(html.len());
    let mut cursor = 0usize;
    while let Some(rel_open) = lower[cursor..].find(&open_lower) {
        let abs_open = cursor + rel_open;
        out.push_str(&html[cursor..abs_open]);
        // Find the corresponding close tag.
        let after_open = abs_open;
        match lower[after_open..].find(&close_lower) {
            Some(rel_close) => {
                let abs_close = after_open + rel_close + close_lower.len();
                cursor = abs_close;
            }
            None => {
                // Unterminated — drop the rest.
                cursor = html.len();
            }
        }
    }
    out.push_str(&html[cursor..]);
    out
}

fn decode_entities(s: &str) -> String {
    let mut out = String::with_capacity(s.len());
    let mut chars = s.chars().peekable();
    while let Some(c) = chars.next() {
        if c != '&' {
            out.push(c);
            continue;
        }
        // Read up to ; or end of string (max 8 chars to avoid
        // pathological inputs).
        let mut buf = String::with_capacity(8);
        let mut consumed = 0usize;
        let mut closed = false;
        while let Some(&n) = chars.peek() {
            if consumed >= 8 {
                break;
            }
            chars.next();
            consumed += 1;
            if n == ';' {
                closed = true;
                break;
            }
            buf.push(n);
        }
        if !closed {
            out.push('&');
            out.push_str(&buf);
            continue;
        }
        match buf.as_str() {
            "lt" => out.push('<'),
            "gt" => out.push('>'),
            "amp" => out.push('&'),
            "quot" => out.push('"'),
            "apos" => out.push('\''),
            "nbsp" => out.push(' '),
            other if other.starts_with('#') => {
                let n = if let Some(stripped) = other.strip_prefix("#x") {
                    u32::from_str_radix(stripped, 16).ok()
                } else {
                    other[1..].parse::<u32>().ok()
                };
                if let Some(c) = n.and_then(char::from_u32) {
                    out.push(c);
                } else {
                    out.push('&');
                    out.push_str(&buf);
                    out.push(';');
                }
            }
            _ => {
                out.push('&');
                out.push_str(&buf);
                out.push(';');
            }
        }
    }
    out
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::sync::Arc;

    #[test]
    fn html_to_text_strips_tags() {
        let html = "<p>Hello <b>world</b></p>";
        assert_eq!(html_to_text(html), "Hello world");
    }

    #[test]
    fn html_to_text_strips_script_and_style() {
        let html = r#"
<html><head>
<script>alert("x")</script>
<style>body { color: red }</style>
</head><body>Real content</body></html>"#;
        let text = html_to_text(html);
        assert!(text.contains("Real content"));
        assert!(!text.contains("alert"));
        assert!(!text.contains("color"));
    }

    #[test]
    fn html_to_text_decodes_named_entities() {
        let html = "<p>&lt;hello&gt; &amp; &quot;world&quot;</p>";
        assert_eq!(html_to_text(html), "<hello> & \"world\"");
    }

    #[test]
    fn html_to_text_decodes_numeric_entities() {
        let html = "<p>&#65;&#x42;&#x4d;</p>"; // ABM
        assert_eq!(html_to_text(html), "ABM");
    }

    #[test]
    fn html_to_text_collapses_blank_lines() {
        let html = "<p>line1</p>\n\n\n<p>line2</p>";
        let out = html_to_text(html);
        assert_eq!(out, "line1\nline2");
    }

    #[test]
    fn html_to_text_handles_unterminated_script() {
        // Defensive: malformed HTML shouldn't blow up.
        let html = "<script>broken";
        let _ = html_to_text(html); // must not panic
    }

    #[tokio::test]
    async fn webfetch_rejects_empty_url() {
        let tool = WebFetchTool::new();
        let ctx = make_ctx();
        let err = tool
            .call(&ctx, json!({"url": "  "}))
            .await
            .expect_err("empty");
        assert!(err.to_string().contains("non-empty"));
    }

    #[tokio::test]
    async fn webfetch_rejects_non_http_schemes() {
        let tool = WebFetchTool::new();
        let ctx = make_ctx();
        for url in [
            "file:///etc/passwd",
            "javascript:alert(1)",
            "data:text/plain,hello",
        ] {
            let err = tool
                .call(&ctx, json!({"url": url}))
                .await
                .expect_err("scheme");
            assert!(
                err.to_string().contains("only supports http"),
                "url {url}: {err}"
            );
        }
    }

    #[tokio::test]
    async fn webfetch_aborts_on_ctx_abort_before_send() {
        let tool = WebFetchTool::new();
        let ctx = make_ctx();
        ctx.abort.abort_with_reason("user cancelled");
        let err = tool
            .call(&ctx, json!({"url": "https://example.com/"}))
            .await
            .expect_err("aborted");
        assert!(matches!(err, AgentError::Aborted(_)));
    }

    fn make_ctx() -> ToolUseContext {
        use std::num::NonZeroUsize;
        ToolUseContext {
            cwd: std::env::current_dir().unwrap(),
            abort: agent::abort::AbortController::new(),
            file_cache: Arc::new(agent::file_cache::FileStateCache::new(
                NonZeroUsize::new(8).unwrap(),
                1024 * 1024,
            )),
            permissions: Arc::new(agent::permission::PermissionManager::new()),
            hooks: Arc::new(agent::hook::HookRunner::new()),
        }
    }

    #[test]
    fn webfetch_classified_read_only() {
        let tool = WebFetchTool::new();
        assert_eq!(tool.safety_class(), SafetyClass::ReadOnly);
    }

    #[test]
    fn is_blocked_ip_blocks_loopback_and_private() {
        use std::net::Ipv4Addr;
        use std::net::Ipv6Addr;
        // IPv4 blocked
        for ip in [
            Ipv4Addr::LOCALHOST,
            Ipv4Addr::new(10, 0, 0, 1),
            Ipv4Addr::new(192, 168, 1, 1),
            Ipv4Addr::new(172, 16, 0, 1),
            Ipv4Addr::new(169, 254, 169, 254), // cloud metadata
            Ipv4Addr::UNSPECIFIED,
        ] {
            assert!(is_blocked_ip(ip.into()), "should block {ip}");
        }
        // IPv4 allowed
        for ip in [
            Ipv4Addr::new(8, 8, 8, 8),
            Ipv4Addr::new(1, 1, 1, 1),
            Ipv4Addr::new(140, 82, 121, 4), // github.com sample
        ] {
            assert!(!is_blocked_ip(ip.into()), "should allow {ip}");
        }
        // IPv6 blocked
        for ip in [
            Ipv6Addr::LOCALHOST,
            Ipv6Addr::UNSPECIFIED,
            "fc00::1".parse::<Ipv6Addr>().unwrap(),
            "fe80::1".parse::<Ipv6Addr>().unwrap(),
        ] {
            assert!(is_blocked_ip(ip.into()), "should block {ip}");
        }
        // IPv6 allowed
        let public_v6: Ipv6Addr = "2606:4700:4700::1111".parse().unwrap();
        assert!(!is_blocked_ip(public_v6.into()));
    }

    #[test]
    fn is_blocked_ip_blocks_ipv4_mapped_loopback_and_private() {
        use std::net::Ipv6Addr;
        // ::ffff:127.0.0.1 — loopback via IPv4-mapped IPv6.
        let mapped_loop: Ipv6Addr = "::ffff:127.0.0.1".parse().unwrap();
        assert!(is_blocked_ip(mapped_loop.into()));
        // ::ffff:10.0.0.1 — RFC1918 via IPv4-mapped IPv6.
        let mapped_priv: Ipv6Addr = "::ffff:10.0.0.1".parse().unwrap();
        assert!(is_blocked_ip(mapped_priv.into()));
        // ::ffff:169.254.169.254 — cloud metadata via IPv4-mapped.
        let mapped_meta: Ipv6Addr = "::ffff:169.254.169.254".parse().unwrap();
        assert!(is_blocked_ip(mapped_meta.into()));
        // ::ffff:8.8.8.8 — public IPv4 via mapping must NOT be blocked.
        let mapped_pub: Ipv6Addr = "::ffff:8.8.8.8".parse().unwrap();
        assert!(!is_blocked_ip(mapped_pub.into()));
    }

    #[tokio::test]
    async fn webfetch_blocks_loopback_url_by_default() {
        let tool = WebFetchTool::new();
        let ctx = make_ctx();
        let err = tool
            .call(&ctx, json!({"url": "http://127.0.0.1:8080/admin"}))
            .await
            .expect_err("ssrf");
        assert!(
            err.to_string().contains("private/loopback") || err.to_string().contains("loopback"),
            "got {err}"
        );
    }

    #[tokio::test]
    async fn webfetch_blocks_metadata_ip_by_default() {
        let tool = WebFetchTool::new();
        let ctx = make_ctx();
        let err = tool
            .call(
                &ctx,
                json!({"url": "http://169.254.169.254/latest/meta-data/"}),
            )
            .await
            .expect_err("ssrf");
        assert!(err.to_string().contains("private"), "got {err}");
    }
}
