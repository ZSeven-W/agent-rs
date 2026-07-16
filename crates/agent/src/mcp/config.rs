//! MCP server configuration (Tier 1 / claude-code parity).
//!
//! Mirrors `services/mcp/config.ts`. Hosts (OpenPencil, Zode) keep
//! a list of MCP server configs in their settings file; this module
//! provides the strongly-typed schema + parser.
//!
//! ## Wire shape
//!
//! ```json
//! {
//!   "servers": {
//!     "github": {
//!       "transport": "stdio",
//!       "command": "npx",
//!       "args": ["-y", "@modelcontextprotocol/server-github"],
//!       "env": { "GITHUB_TOKEN": "$GITHUB_TOKEN" }
//!     },
//!     "linear": {
//!       "transport": "sse",
//!       "url": "https://mcp.linear.app/sse",
//!       "headers": { "Authorization": "Bearer $LINEAR_TOKEN" }
//!     },
//!     "remote-tools": {
//!       "transport": "websocket",
//!       "url": "wss://api.example.com/mcp"
//!     }
//!   }
//! }
//! ```
//!
//! Environment-variable substitution (`$NAME` or `${NAME}`) happens
//! at expand time via [`McpServerConfig::expand_env`] — the host
//! decides when (typically just before connect).

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Top-level config: a name → server map.
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
pub struct McpConfig {
    #[serde(default)]
    pub servers: BTreeMap<String, McpServerConfig>,
}

/// A single MCP server's connection spec.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "transport", rename_all = "snake_case")]
#[non_exhaustive]
pub enum McpServerConfig {
    /// Subprocess — agent spawns the binary and pipes stdio.
    Stdio {
        command: String,
        #[serde(default)]
        args: Vec<String>,
        #[serde(default)]
        env: BTreeMap<String, String>,
        /// Optional working directory. Defaults to inherit.
        #[serde(default)]
        cwd: Option<String>,
        /// Whether the server is enabled. Disabled servers stay in the
        /// config so the host UI can render them, but the lifecycle
        /// manager skips connect.
        #[serde(default = "default_true")]
        enabled: bool,
    },
    /// Server-Sent Events (SSE) over HTTPS.
    Sse {
        url: String,
        #[serde(default)]
        headers: BTreeMap<String, String>,
        #[serde(default = "default_true")]
        enabled: bool,
    },
    /// WebSocket transport.
    #[serde(rename = "websocket")]
    WebSocket {
        url: String,
        #[serde(default)]
        headers: BTreeMap<String, String>,
        #[serde(default = "default_true")]
        enabled: bool,
    },
}

const fn default_true() -> bool {
    true
}

impl McpServerConfig {
    pub fn enabled(&self) -> bool {
        match self {
            Self::Stdio { enabled, .. }
            | Self::Sse { enabled, .. }
            | Self::WebSocket { enabled, .. } => *enabled,
        }
    }

    /// Expand `$NAME` / `${NAME}` references in command, args, env
    /// values, urls, and header values against the supplied env map.
    /// Unset variables become empty strings (matches claude-code's
    /// behavior for templated configs).
    pub fn expand_env(&mut self, env: &BTreeMap<String, String>) {
        match self {
            Self::Stdio {
                command,
                args,
                env: e,
                cwd,
                ..
            } => {
                expand(command, env);
                for a in args {
                    expand(a, env);
                }
                for v in e.values_mut() {
                    expand(v, env);
                }
                if let Some(c) = cwd.as_mut() {
                    expand(c, env);
                }
            }
            Self::Sse { url, headers, .. } | Self::WebSocket { url, headers, .. } => {
                expand(url, env);
                for v in headers.values_mut() {
                    expand(v, env);
                }
            }
        }
    }
}

/// Parse an MCP config from a JSON byte slice.
pub fn parse_json(bytes: &[u8]) -> Result<McpConfig, ConfigError> {
    serde_json::from_slice(bytes).map_err(|e| ConfigError::Json(e.to_string()))
}

/// Parse from a string.
pub fn parse_json_str(s: &str) -> Result<McpConfig, ConfigError> {
    parse_json(s.as_bytes())
}

#[derive(Debug, thiserror::Error)]
pub enum ConfigError {
    #[error("config json parse error: {0}")]
    Json(String),
}

/// Substitute `$VAR` and `${VAR}` references in `s` against `env`.
/// Unknown variables expand to empty. `$$` escapes to a literal `$`.
///
/// **Character-aware**: walks `char_indices` so non-ASCII text in
/// the surrounding template is preserved verbatim. Variable names
/// must start with `[A-Za-z_]` (no `$1`-style positional params),
/// continue with `[A-Za-z0-9_]`. `${NAME}` form accepts the same
/// charset between braces.
fn expand(s: &mut String, env: &BTreeMap<String, String>) {
    if !s.contains('$') {
        return;
    }
    let mut out = String::with_capacity(s.len());
    let chars: Vec<(usize, char)> = s.char_indices().collect();
    let mut i = 0;
    while i < chars.len() {
        let (_, c) = chars[i];
        if c != '$' {
            out.push(c);
            i += 1;
            continue;
        }
        // Need at least one more char to be a possible substitution.
        if i + 1 >= chars.len() {
            out.push('$');
            i += 1;
            continue;
        }
        let (_, next) = chars[i + 1];
        if next == '$' {
            // `$$` literal $.
            out.push('$');
            i += 2;
            continue;
        }
        if next == '{' {
            // `${NAME}` form. The body MUST match the same identifier
            // charset as bare `$NAME`: first char ASCII letter or
            // underscore, rest ASCII alnum or underscore, then `}`.
            // Anything else (`${FOO-BAR}`, `${FOO:-default}`,
            // `${FOO BAR}`) is treated as a literal — the entire
            // `${...}` (including the closing brace, when present)
            // is copied to output and the parser advances past it,
            // so any inner `$` does NOT get re-expanded. This
            // approximates POSIX-shell parameter-substitution
            // semantics for the unsupported subset (we don't
            // implement defaults, alternates, etc.).
            // Compute the validity + extent of the body up front so
            // every "invalid" path takes the same literal-pass-through
            // branch — including `${$BAR}` / `${1}` / `${}` (where the
            // first char fails the identifier rule).
            let mut valid = i + 2 < chars.len();
            if valid {
                let (_, first) = chars[i + 2];
                if !(first.is_ascii_alphabetic() || first == '_') {
                    valid = false;
                }
            }
            let mut end = i + 2;
            if valid {
                while end < chars.len() && chars[end].1 != '}' {
                    let ch = chars[end].1;
                    if !(ch.is_ascii_alphanumeric() || ch == '_') {
                        valid = false;
                        break;
                    }
                    end += 1;
                }
            }
            if valid && end < chars.len() && chars[end].1 == '}' {
                let name: String = chars[i + 2..end].iter().map(|(_, c)| *c).collect();
                if !name.is_empty() {
                    if let Some(v) = env.get(&name) {
                        out.push_str(v);
                    }
                }
                i = end + 1;
                continue;
            }
            // Invalid (any reason: bad first char, bad mid char,
            // unterminated). Emit the entire `${...}` (or `${...` for
            // unterminated) as literal and skip past the closing
            // brace if there is one — so any inner `$` does NOT get
            // re-expanded on the next iteration.
            let close = chars[i + 2..]
                .iter()
                .position(|(_, c)| *c == '}')
                .map(|p| i + 2 + p);
            let stop = close.map(|p| p + 1).unwrap_or(chars.len());
            for (_, c) in &chars[i..stop] {
                out.push(*c);
            }
            i = stop;
            continue;
        }
        // `$NAME` — first char must be ASCII letter or underscore.
        // `$1` etc. are NOT treated as variables; they pass through
        // literally (matches POSIX shell positional convention but
        // we don't expand them).
        if !(next.is_ascii_alphabetic() || next == '_') {
            out.push('$');
            i += 1;
            continue;
        }
        let mut end = i + 1;
        while end < chars.len() {
            let (_, ch) = chars[end];
            if ch.is_ascii_alphanumeric() || ch == '_' {
                end += 1;
            } else {
                break;
            }
        }
        let name: String = chars[i + 1..end].iter().map(|(_, c)| *c).collect();
        if let Some(v) = env.get(&name) {
            out.push_str(v);
        }
        i = end;
    }
    *s = out;
}

#[cfg(test)]
mod tests {
    use super::*;

    fn env_with<I: IntoIterator<Item = (&'static str, &'static str)>>(
        pairs: I,
    ) -> BTreeMap<String, String> {
        pairs
            .into_iter()
            .map(|(k, v)| (k.to_string(), v.to_string()))
            .collect()
    }

    #[test]
    fn parse_stdio_config() {
        let json = r#"{
            "servers": {
                "github": {
                    "transport": "stdio",
                    "command": "npx",
                    "args": ["-y", "@modelcontextprotocol/server-github"],
                    "env": { "GITHUB_TOKEN": "$GITHUB_TOKEN" }
                }
            }
        }"#;
        let cfg = parse_json_str(json).unwrap();
        assert_eq!(cfg.servers.len(), 1);
        match cfg.servers.get("github").unwrap() {
            McpServerConfig::Stdio {
                command,
                args,
                env,
                enabled,
                ..
            } => {
                assert_eq!(command, "npx");
                assert_eq!(args.len(), 2);
                assert_eq!(env.get("GITHUB_TOKEN").unwrap(), "$GITHUB_TOKEN");
                assert!(*enabled);
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn parse_sse_and_websocket() {
        let json = r#"{
            "servers": {
                "linear": { "transport": "sse", "url": "https://x" },
                "ws":     { "transport": "websocket", "url": "wss://y" }
            }
        }"#;
        let cfg = parse_json_str(json).unwrap();
        assert!(matches!(
            cfg.servers.get("linear").unwrap(),
            McpServerConfig::Sse { .. }
        ));
        assert!(matches!(
            cfg.servers.get("ws").unwrap(),
            McpServerConfig::WebSocket { .. }
        ));
    }

    #[test]
    fn enabled_defaults_true() {
        let json = r#"{ "servers": { "x": { "transport": "stdio", "command": "echo" } } }"#;
        let cfg = parse_json_str(json).unwrap();
        assert!(cfg.servers.get("x").unwrap().enabled());
    }

    #[test]
    fn enabled_can_be_false() {
        let json = r#"{
            "servers": {
                "x": { "transport": "stdio", "command": "echo", "enabled": false }
            }
        }"#;
        let cfg = parse_json_str(json).unwrap();
        assert!(!cfg.servers.get("x").unwrap().enabled());
    }

    #[test]
    fn expand_env_simple_var() {
        let env = env_with([("FOO", "bar")]);
        let mut s = "prefix-$FOO-suffix".to_string();
        expand(&mut s, &env);
        assert_eq!(s, "prefix-bar-suffix");
    }

    #[test]
    fn expand_env_braced_var() {
        let env = env_with([("FOO", "bar")]);
        let mut s = "${FOO}-${MISSING}".to_string();
        expand(&mut s, &env);
        assert_eq!(s, "bar-");
    }

    #[test]
    fn expand_env_double_dollar_escapes() {
        let env = env_with([("FOO", "bar")]);
        let mut s = "$$NOTVAR-$FOO".to_string();
        expand(&mut s, &env);
        assert_eq!(s, "$NOTVAR-bar");
    }

    #[test]
    fn expand_env_unterminated_brace_left_literal() {
        let env = env_with([("FOO", "bar")]);
        let mut s = "${FOO".to_string();
        expand(&mut s, &env);
        // Unterminated → leave the leading $ as literal, rest is just text.
        assert!(s.contains("$"));
    }

    #[test]
    fn expand_env_preserves_non_ascii_text() {
        let env = env_with([("FOO", "bar")]);
        let mut s = "你好-$FOO-世界".to_string();
        expand(&mut s, &env);
        assert_eq!(s, "你好-bar-世界");
    }

    #[test]
    fn expand_env_braced_first_char_invalid_does_not_expand_inner() {
        let env = env_with([("BAR", "should-not-leak")]);
        // First-char-invalid (e.g. `${$BAR}`, `${1abc}`, `${}`) must
        // pass the entire `${...}` as literal, so the inner `$BAR`
        // doesn't get re-expanded on the next iteration.
        for &input in &["${$BAR}", "${1abc}", "${}", "${ FOO}"] {
            let mut s = input.to_string();
            expand(&mut s, &env);
            assert_eq!(s, input, "input {input} mangled to {s}");
        }
    }

    #[test]
    fn expand_env_nested_braces_do_not_expand_inner() {
        let env = env_with([("BAR", "should-not-leak"), ("FOO", "fooval")]);
        // `${FOO${BAR}}` is invalid (contains nested `$`); the entire
        // construct must pass through literally — the inner `${BAR}`
        // must NOT expand because of re-parsing.
        let mut s = "${FOO${BAR}}".to_string();
        expand(&mut s, &env);
        assert_eq!(s, "${FOO${BAR}}");
        // Same idea with bare $BAR inside an invalid brace body.
        let mut s = "${FOO$BAR}".to_string();
        expand(&mut s, &env);
        assert_eq!(s, "${FOO$BAR}");
    }

    #[test]
    fn expand_env_braced_invalid_charset_left_literal() {
        let env = env_with([("FOO", "bar")]);
        // POSIX shell-style modifiers are NOT supported — they must
        // pass through literally rather than silently erase.
        for &input in &["${FOO-BAR}", "${FOO:-default}", "${FOO BAR}", "${FOO/x/y}"] {
            let mut s = input.to_string();
            expand(&mut s, &env);
            assert_eq!(s, input, "input {input} mangled to {s}");
        }
        // Bare `$FOO-BAR` expands the FOO part and keeps -BAR.
        let mut s = "$FOO-BAR".to_string();
        expand(&mut s, &env);
        assert_eq!(s, "bar-BAR");
    }

    #[test]
    fn expand_env_braced_dollar_digit_is_literal() {
        let env = env_with([("1", "should-not-expand")]);
        let mut s = "${1}".to_string();
        expand(&mut s, &env);
        // `${1}` must NOT be treated as a variable ref.
        assert_eq!(s, "${1}");
    }

    #[test]
    fn expand_env_dollar_digit_is_literal() {
        let env = env_with([("1", "should-not-expand")]);
        let mut s = "$1 and $FOO".to_string();
        expand(&mut s, &env);
        // $1 literal; $FOO unset → empty.
        assert_eq!(s, "$1 and ");
    }

    #[test]
    fn expand_env_unknown_var_becomes_empty() {
        let env = env_with([]);
        let mut s = "x-$NOPE-y".to_string();
        expand(&mut s, &env);
        assert_eq!(s, "x--y");
    }

    #[test]
    fn expand_server_config_stdio() {
        let env = env_with([("GITHUB_TOKEN", "ghp_xxx"), ("HOME", "/u/me")]);
        let mut cfg = McpServerConfig::Stdio {
            command: "$HOME/bin/srv".into(),
            args: vec!["--token=$GITHUB_TOKEN".into()],
            env: [("AUTH".into(), "Bearer $GITHUB_TOKEN".into())].into(),
            cwd: Some("$HOME/work".into()),
            enabled: true,
        };
        cfg.expand_env(&env);
        match cfg {
            McpServerConfig::Stdio {
                command,
                args,
                env: e,
                cwd,
                ..
            } => {
                assert_eq!(command, "/u/me/bin/srv");
                assert_eq!(args[0], "--token=ghp_xxx");
                assert_eq!(e.get("AUTH").unwrap(), "Bearer ghp_xxx");
                assert_eq!(cwd.unwrap(), "/u/me/work");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn expand_sse_config_url_and_headers() {
        let env = env_with([("API", "https://api.example"), ("TOK", "xyz")]);
        let mut cfg = McpServerConfig::Sse {
            url: "$API/sse".into(),
            headers: [("Authorization".into(), "Bearer $TOK".into())].into(),
            enabled: true,
        };
        cfg.expand_env(&env);
        match cfg {
            McpServerConfig::Sse { url, headers, .. } => {
                assert_eq!(url, "https://api.example/sse");
                assert_eq!(headers.get("Authorization").unwrap(), "Bearer xyz");
            }
            _ => panic!("wrong variant"),
        }
    }

    #[test]
    fn empty_servers_parses() {
        let cfg = parse_json_str(r#"{"servers": {}}"#).unwrap();
        assert!(cfg.servers.is_empty());
    }

    #[test]
    fn missing_servers_field_parses_to_empty() {
        let cfg = parse_json_str(r#"{}"#).unwrap();
        assert!(cfg.servers.is_empty());
    }

    #[test]
    fn invalid_json_errors() {
        assert!(matches!(
            parse_json_str("{ broken").unwrap_err(),
            ConfigError::Json(_)
        ));
    }

    #[test]
    fn roundtrip_serialize_deserialize() {
        let cfg = McpConfig {
            servers: [(
                "x".to_string(),
                McpServerConfig::Stdio {
                    command: "cmd".into(),
                    args: vec!["a".into()],
                    env: [("K".into(), "V".into())].into(),
                    cwd: None,
                    enabled: true,
                },
            )]
            .into(),
        };
        let json = serde_json::to_string(&cfg).unwrap();
        let parsed: McpConfig = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, cfg);
    }
}
