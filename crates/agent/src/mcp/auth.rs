//! MCP OAuth helper (Tier 1 / claude-code parity).
//!
//! Mirrors `services/mcp/auth.ts`. Many remote MCP servers (Linear,
//! Notion, custom enterprise) require OAuth 2.0 authorization-code
//! flow before they'll accept tool calls. This module implements the
//! generic OAuth flow:
//!
//! 1. `OauthClient::start` builds an authorization URL and returns
//!    a [`PendingAuthorization`] carrying that URL, the `state`
//!    nonce, and the PKCE verifier.
//! 2. Host displays the URL via the [`super::elicitation`] handler
//!    (typically `OpenUrl`).
//! 3. User authorizes, browser redirects to `redirect_uri` carrying
//!    `code` + `state`.
//! 4. Host calls `OauthClient::exchange_code` with the `code`; we
//!    POST to `token_url` and return [`Tokens`].
//! 5. Host stores tokens; subsequent server requests use the access
//!    token. When it expires, [`OauthClient::refresh`] swaps the
//!    refresh token for a fresh access token.
//!
//! The HTTP client is `reqwest` so OAuth lives behind the `mcp`
//! feature alongside the rest of the module. Local-redirect-server
//! capture (the part that listens for the browser callback) is
//! intentionally NOT included — the host owns the browser bridge
//! and our `exchange_code` accepts the captured `code` directly.

use std::collections::BTreeMap;

use serde::{Deserialize, Serialize};

/// Bundle of client config needed to drive an OAuth flow.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct OauthClient {
    /// Public client identifier issued by the OAuth provider.
    pub client_id: String,
    /// Optional client secret — required for some providers, omitted
    /// for PKCE-only public clients.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub client_secret: Option<String>,
    /// Authorization endpoint (the URL the user's browser visits).
    pub authorization_url: String,
    /// Token endpoint (the URL we POST `code` and `refresh_token` to).
    pub token_url: String,
    /// Local redirect URI registered with the provider.
    pub redirect_uri: String,
    /// Requested scopes — joined with spaces in the `scope` query param.
    #[serde(default)]
    pub scopes: Vec<String>,
}

/// State of an in-flight authorization. Returned from
/// [`OauthClient::start`] for the host to surface in its UI.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PendingAuthorization {
    pub authorize_url: String,
    /// CSRF guard — caller compares this against the `state` carried
    /// back on the redirect.
    pub state: String,
    /// PKCE verifier — sent to the token endpoint with the code.
    /// [`OauthClient::start`] always sets this to `Some`; callers
    /// that construct [`PendingAuthorization`] manually (e.g. for
    /// tests) may set `None` to omit PKCE.
    pub code_verifier: Option<String>,
}

/// Tokens returned by the token endpoint.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct Tokens {
    pub access_token: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub refresh_token: Option<String>,
    /// Lifetime in seconds — host should refresh before this elapses.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub expires_in: Option<u32>,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub token_type: Option<String>,
    /// Provider-supplied scope string (may differ from requested).
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub scope: Option<String>,
}

#[derive(Debug, thiserror::Error)]
pub enum AuthError {
    #[error("auth: state mismatch — possible CSRF")]
    StateMismatch,
    #[error("auth: token endpoint returned {status}: {body}")]
    TokenEndpoint { status: u16, body: String },
    #[error("auth: http: {0}")]
    Http(String),
    #[error("auth: parse: {0}")]
    Parse(String),
}

impl OauthClient {
    /// Build the authorization URL + state nonce. PKCE always on.
    pub fn start(&self) -> PendingAuthorization {
        let state = generate_token(24);
        let verifier = generate_token(43);
        let challenge = pkce_challenge(&verifier);
        let mut params: Vec<(String, String)> = vec![
            ("response_type".into(), "code".into()),
            ("client_id".into(), self.client_id.clone()),
            ("redirect_uri".into(), self.redirect_uri.clone()),
            ("state".into(), state.clone()),
            ("code_challenge".into(), challenge),
            ("code_challenge_method".into(), "S256".into()),
        ];
        if !self.scopes.is_empty() {
            params.push(("scope".into(), self.scopes.join(" ")));
        }
        let qs = encode_query(&params);
        let sep = if self.authorization_url.contains('?') {
            '&'
        } else {
            '?'
        };
        let authorize_url = format!("{}{sep}{qs}", self.authorization_url);
        PendingAuthorization {
            authorize_url,
            state,
            code_verifier: Some(verifier),
        }
    }

    /// Exchange an authorization code for tokens. Returns
    /// [`AuthError::StateMismatch`] if `received_state` differs from
    /// the pending nonce. Networking is feature-gated on `mcp` (which
    /// already implies `reqwest`).
    pub async fn exchange_code(
        &self,
        pending: &PendingAuthorization,
        received_state: &str,
        code: &str,
    ) -> Result<Tokens, AuthError> {
        if !constant_time_eq(received_state.as_bytes(), pending.state.as_bytes()) {
            return Err(AuthError::StateMismatch);
        }
        let mut form: BTreeMap<&str, String> = BTreeMap::new();
        form.insert("grant_type", "authorization_code".into());
        form.insert("code", code.to_string());
        form.insert("redirect_uri", self.redirect_uri.clone());
        form.insert("client_id", self.client_id.clone());
        if let Some(s) = &self.client_secret {
            form.insert("client_secret", s.clone());
        }
        if let Some(v) = &pending.code_verifier {
            form.insert("code_verifier", v.clone());
        }
        post_form(&self.token_url, &form).await
    }

    /// Swap a refresh token for a fresh access token. Caller persists
    /// the returned [`Tokens`] (refresh_token may rotate).
    pub async fn refresh(&self, refresh_token: &str) -> Result<Tokens, AuthError> {
        let mut form: BTreeMap<&str, String> = BTreeMap::new();
        form.insert("grant_type", "refresh_token".into());
        form.insert("refresh_token", refresh_token.to_string());
        form.insert("client_id", self.client_id.clone());
        if let Some(s) = &self.client_secret {
            form.insert("client_secret", s.clone());
        }
        post_form(&self.token_url, &form).await
    }
}

async fn post_form(url: &str, form: &BTreeMap<&str, String>) -> Result<Tokens, AuthError> {
    // reqwest is enabled by the `mcp` feature directly (see
    // crates/agent/Cargo.toml). No dependency on `anthropic`.
    let client = reqwest::Client::new();
    let resp = client
        .post(url)
        .form(form)
        .send()
        .await
        .map_err(|e| AuthError::Http(e.to_string()))?;
    let status = resp.status().as_u16();
    let body = resp
        .text()
        .await
        .map_err(|e| AuthError::Http(e.to_string()))?;
    if !(200..300).contains(&status) {
        return Err(AuthError::TokenEndpoint { status, body });
    }
    serde_json::from_str(&body).map_err(|e| AuthError::Parse(e.to_string()))
}

/// Constant-time equality. Prevents timing side-channels on the
/// position-of-first-mismatch dimension that a naive `==` would leak.
///
/// Walks `max(a.len(), b.len())` bytes regardless of where the inputs
/// differ; length difference is folded into the accumulator instead
/// of short-circuiting. Length itself still affects total runtime
/// (the loop runs N iterations), so this is NOT a defense against
/// length-leak attacks — the OAuth state nonce we compare against has
/// a fixed length set at generation time, so length is not a meaningful
/// secret in this code path.
fn constant_time_eq(a: &[u8], b: &[u8]) -> bool {
    let n = a.len().max(b.len());
    let mut acc: u32 = (a.len() ^ b.len()) as u32;
    for i in 0..n {
        let x = *a.get(i).unwrap_or(&0);
        let y = *b.get(i).unwrap_or(&0);
        acc |= (x ^ y) as u32;
    }
    acc == 0
}

/// Random URL-safe base64 token. Quality is sufficient for OAuth
/// state + PKCE verifier per RFC 7636.
fn generate_token(min_len: usize) -> String {
    use rand::Rng;
    const ALPHABET: &[u8] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut rng = rand::thread_rng();
    (0..min_len)
        .map(|_| {
            let i = rng.gen_range(0..ALPHABET.len());
            ALPHABET[i] as char
        })
        .collect()
}

/// PKCE S256 challenge — base64url(SHA-256(verifier)). Inline impl
/// avoids pulling sha2 as a dep.
fn pkce_challenge(verifier: &str) -> String {
    let hash = sha256_bytes(verifier.as_bytes());
    base64url_no_pad(&hash)
}

/// Inline SHA-256 implementation (FIPS 180-4). About 100 lines; pulls
/// no extra deps. Constants are the published H₀ + K₀..K₆₃.
fn sha256_bytes(input: &[u8]) -> [u8; 32] {
    const H0: [u32; 8] = [
        0x6a09_e667,
        0xbb67_ae85,
        0x3c6e_f372,
        0xa54f_f53a,
        0x510e_527f,
        0x9b05_688c,
        0x1f83_d9ab,
        0x5be0_cd19,
    ];
    const K: [u32; 64] = [
        0x428a_2f98,
        0x7137_4491,
        0xb5c0_fbcf,
        0xe9b5_dba5,
        0x3956_c25b,
        0x59f1_11f1,
        0x923f_82a4,
        0xab1c_5ed5,
        0xd807_aa98,
        0x1283_5b01,
        0x2431_85be,
        0x550c_7dc3,
        0x72be_5d74,
        0x80de_b1fe,
        0x9bdc_06a7,
        0xc19b_f174,
        0xe49b_69c1,
        0xefbe_4786,
        0x0fc1_9dc6,
        0x240c_a1cc,
        0x2de9_2c6f,
        0x4a74_84aa,
        0x5cb0_a9dc,
        0x76f9_88da,
        0x983e_5152,
        0xa831_c66d,
        0xb003_27c8,
        0xbf59_7fc7,
        0xc6e0_0bf3,
        0xd5a7_9147,
        0x06ca_6351,
        0x1429_2967,
        0x27b7_0a85,
        0x2e1b_2138,
        0x4d2c_6dfc,
        0x5338_0d13,
        0x650a_7354,
        0x766a_0abb,
        0x81c2_c92e,
        0x9272_2c85,
        0xa2bf_e8a1,
        0xa81a_664b,
        0xc24b_8b70,
        0xc76c_51a3,
        0xd192_e819,
        0xd699_0624,
        0xf40e_3585,
        0x106a_a070,
        0x19a4_c116,
        0x1e37_6c08,
        0x2748_774c,
        0x34b0_bcb5,
        0x391c_0cb3,
        0x4ed8_aa4a,
        0x5b9c_ca4f,
        0x682e_6ff3,
        0x748f_82ee,
        0x78a5_636f,
        0x84c8_7814,
        0x8cc7_0208,
        0x90be_fffa,
        0xa450_6ceb,
        0xbef9_a3f7,
        0xc671_78f2,
    ];

    let bit_len = (input.len() as u64).wrapping_mul(8);
    let mut padded = input.to_vec();
    padded.push(0x80);
    while padded.len() % 64 != 56 {
        padded.push(0);
    }
    padded.extend_from_slice(&bit_len.to_be_bytes());

    let mut h = H0;
    for block in padded.chunks_exact(64) {
        let mut w = [0u32; 64];
        for (i, chunk) in block.chunks_exact(4).enumerate() {
            w[i] = u32::from_be_bytes([chunk[0], chunk[1], chunk[2], chunk[3]]);
        }
        for i in 16..64 {
            let s0 = w[i - 15].rotate_right(7) ^ w[i - 15].rotate_right(18) ^ (w[i - 15] >> 3);
            let s1 = w[i - 2].rotate_right(17) ^ w[i - 2].rotate_right(19) ^ (w[i - 2] >> 10);
            w[i] = w[i - 16]
                .wrapping_add(s0)
                .wrapping_add(w[i - 7])
                .wrapping_add(s1);
        }
        let mut a = h[0];
        let mut b = h[1];
        let mut c = h[2];
        let mut d = h[3];
        let mut e = h[4];
        let mut f = h[5];
        let mut g = h[6];
        let mut hh = h[7];
        for i in 0..64 {
            let s1 = e.rotate_right(6) ^ e.rotate_right(11) ^ e.rotate_right(25);
            let ch = (e & f) ^ ((!e) & g);
            let temp1 = hh
                .wrapping_add(s1)
                .wrapping_add(ch)
                .wrapping_add(K[i])
                .wrapping_add(w[i]);
            let s0 = a.rotate_right(2) ^ a.rotate_right(13) ^ a.rotate_right(22);
            let maj = (a & b) ^ (a & c) ^ (b & c);
            let temp2 = s0.wrapping_add(maj);
            hh = g;
            g = f;
            f = e;
            e = d.wrapping_add(temp1);
            d = c;
            c = b;
            b = a;
            a = temp1.wrapping_add(temp2);
        }
        h[0] = h[0].wrapping_add(a);
        h[1] = h[1].wrapping_add(b);
        h[2] = h[2].wrapping_add(c);
        h[3] = h[3].wrapping_add(d);
        h[4] = h[4].wrapping_add(e);
        h[5] = h[5].wrapping_add(f);
        h[6] = h[6].wrapping_add(g);
        h[7] = h[7].wrapping_add(hh);
    }
    let mut out = [0u8; 32];
    for (i, word) in h.iter().enumerate() {
        out[i * 4..(i + 1) * 4].copy_from_slice(&word.to_be_bytes());
    }
    out
}

/// Base64URL encoding without padding (RFC 4648 §5). Inline because
/// we only call it on 32-byte SHA-256 outputs.
fn base64url_no_pad(bytes: &[u8]) -> String {
    const ALPHABET: &[u8; 64] = b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_";
    let mut out = String::with_capacity(bytes.len() * 4 / 3 + 4);
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
        }
        if chunk.len() > 2 {
            out.push(ALPHABET[(n & 0x3f) as usize] as char);
        }
    }
    out
}

/// Form-style URL encoding: `key=value&key2=value2`, percent-encoding
/// reserved chars per RFC 3986.
fn encode_query(params: &[(String, String)]) -> String {
    let mut out = String::new();
    for (i, (k, v)) in params.iter().enumerate() {
        if i > 0 {
            out.push('&');
        }
        percent_encode(k, &mut out);
        out.push('=');
        percent_encode(v, &mut out);
    }
    out
}

fn percent_encode(s: &str, out: &mut String) {
    for &b in s.as_bytes() {
        if b.is_ascii_alphanumeric() || matches!(b, b'-' | b'_' | b'.' | b'~') {
            out.push(b as char);
        } else {
            out.push('%');
            out.push(hex_upper(b >> 4));
            out.push(hex_upper(b & 0xf));
        }
    }
}

fn hex_upper(n: u8) -> char {
    let n = n & 0xf;
    if n < 10 {
        (b'0' + n) as char
    } else {
        (b'A' + n - 10) as char
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn client() -> OauthClient {
        OauthClient {
            client_id: "abc".into(),
            client_secret: None,
            authorization_url: "https://auth.example/authorize".into(),
            token_url: "https://auth.example/token".into(),
            redirect_uri: "http://localhost:7919/cb".into(),
            scopes: vec!["read".into(), "write".into()],
        }
    }

    #[test]
    fn start_builds_url_with_required_params() {
        let p = client().start();
        assert!(p
            .authorize_url
            .starts_with("https://auth.example/authorize?"));
        assert!(p.authorize_url.contains("response_type=code"));
        assert!(p.authorize_url.contains("client_id=abc"));
        assert!(p.authorize_url.contains("code_challenge_method=S256"));
        assert!(p.authorize_url.contains("scope=read%20write"));
        assert!(!p.state.is_empty());
        assert!(p.code_verifier.is_some());
    }

    #[test]
    fn start_handles_existing_query_in_authorization_url() {
        let mut c = client();
        c.authorization_url = "https://x/auth?tenant=foo".into();
        let p = c.start();
        // Second join uses & not ?
        let qs = p.authorize_url.split_once("auth?").unwrap().1;
        assert!(qs.starts_with("tenant=foo&"));
    }

    #[test]
    fn sha256_known_vector() {
        // SHA-256("abc") = ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad
        let h = sha256_bytes(b"abc");
        let expected = [
            0xba, 0x78, 0x16, 0xbf, 0x8f, 0x01, 0xcf, 0xea, 0x41, 0x41, 0x40, 0xde, 0x5d, 0xae,
            0x22, 0x23, 0xb0, 0x03, 0x61, 0xa3, 0x96, 0x17, 0x7a, 0x9c, 0xb4, 0x10, 0xff, 0x61,
            0xf2, 0x00, 0x15, 0xad,
        ];
        assert_eq!(h, expected);
    }

    #[test]
    fn sha256_empty_input() {
        // SHA-256("") = e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855
        let h = sha256_bytes(b"");
        let expected = [
            0xe3, 0xb0, 0xc4, 0x42, 0x98, 0xfc, 0x1c, 0x14, 0x9a, 0xfb, 0xf4, 0xc8, 0x99, 0x6f,
            0xb9, 0x24, 0x27, 0xae, 0x41, 0xe4, 0x64, 0x9b, 0x93, 0x4c, 0xa4, 0x95, 0x99, 0x1b,
            0x78, 0x52, 0xb8, 0x55,
        ];
        assert_eq!(h, expected);
    }

    #[test]
    fn pkce_challenge_matches_rfc7636_appendix_b() {
        // RFC 7636 Appendix B test vector.
        let verifier = "dBjftJeZ4CVP-mB92K27uhbUJU1p1r_wW1gFWFOEjXk";
        let chall = pkce_challenge(verifier);
        assert_eq!(chall, "E9Melhoa2OwvFrEMTJguCHaoeK1t8URWbuGJSstw-cM");
    }

    #[test]
    fn base64url_no_pad_known() {
        assert_eq!(base64url_no_pad(b""), "");
        assert_eq!(base64url_no_pad(b"f"), "Zg");
        assert_eq!(base64url_no_pad(b"fo"), "Zm8");
        assert_eq!(base64url_no_pad(b"foo"), "Zm9v");
        assert_eq!(base64url_no_pad(b"foob"), "Zm9vYg");
    }

    #[test]
    fn percent_encode_round_trip() {
        let mut out = String::new();
        percent_encode("hello world!", &mut out);
        assert_eq!(out, "hello%20world%21");
    }

    #[test]
    fn constant_time_eq_known_values() {
        assert!(constant_time_eq(b"abc", b"abc"));
        assert!(!constant_time_eq(b"abc", b"abd"));
        assert!(!constant_time_eq(b"ab", b"abc"));
        assert!(constant_time_eq(b"", b""));
        // Same length but every byte different.
        assert!(!constant_time_eq(b"\x00\x00\x00", b"\xff\xff\xff"));
    }

    #[test]
    fn constant_time_eq_length_mismatch_returns_false() {
        // Mismatched lengths produce deterministic false. We can't
        // directly assert constant-time runtime from a unit test;
        // this exercise verifies correctness across drastic length
        // skews.
        assert!(!constant_time_eq(
            b"x",
            b"this is much longer than the other side"
        ));
        assert!(!constant_time_eq(b"", b"a"));
        assert!(!constant_time_eq(b"a", b""));
    }

    #[tokio::test]
    async fn exchange_rejects_state_mismatch() {
        let p = PendingAuthorization {
            authorize_url: "x".into(),
            state: "abc".into(),
            code_verifier: Some("v".into()),
        };
        let err = client()
            .exchange_code(&p, "wrong", "code")
            .await
            .unwrap_err();
        assert!(matches!(err, AuthError::StateMismatch));
    }

    #[test]
    fn tokens_serde_roundtrip() {
        let t = Tokens {
            access_token: "AT".into(),
            refresh_token: Some("RT".into()),
            expires_in: Some(3600),
            token_type: Some("Bearer".into()),
            scope: Some("read".into()),
        };
        let json = serde_json::to_string(&t).unwrap();
        let parsed: Tokens = serde_json::from_str(&json).unwrap();
        assert_eq!(parsed, t);
    }
}
