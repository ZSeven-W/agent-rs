//! Permission decision types — shape ported verbatim from the Zig
//! `agent/src/permission.zig` (Tier A in the skeleton audit).

use serde::{Deserialize, Serialize};

/// Structured matcher for tool input. A [`PermissionRule`] uses a
/// matcher to decide whether the rule applies to a specific tool call.
///
/// `Always` is the whole-tool case — matches any input, equivalent to
/// the legacy `rule_content == None` behavior. `Field` reaches into
/// the input via an [RFC 6901 JSON pointer] and runs a string pattern
/// against the resolved string value. `AnyOf` / `AllOf` / `Not`
/// compose. `ExactJson` requires the entire input to be JSON-equal to
/// a literal value.
///
/// # Why no regex out of the box
///
/// agent-rs deliberately doesn't depend on `regex` to keep the dep
/// tree thin. Hosts that need regex matching can implement their own
/// matcher by constructing the rule with a [`StringPattern::Glob`]
/// fallback and rejecting at the tool boundary, or extend the matcher
/// in their own crate using the public enum.
///
/// [RFC 6901 JSON pointer]: https://datatracker.ietf.org/doc/html/rfc6901
#[derive(Debug, Clone, Default, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum PermissionMatcher {
    /// Whole-tool — matches any input. Default for [`PermissionRule::whole_tool`].
    #[default]
    Always,
    /// Match a string field at `pointer` against `pattern`. If the
    /// pointer doesn't resolve, or the resolved value isn't a string,
    /// the matcher returns false.
    Field {
        pointer: String,
        pattern: StringPattern,
    },
    /// Exact JSON-value equality on the entire input.
    ExactJson { value: serde_json::Value },
    /// Logical OR — match iff any sub-matcher matches. An empty list
    /// never matches.
    AnyOf { matchers: Vec<PermissionMatcher> },
    /// Logical AND — match iff every sub-matcher matches. An empty
    /// list always matches (vacuous truth).
    AllOf { matchers: Vec<PermissionMatcher> },
    /// Logical NOT — match iff the inner matcher doesn't.
    Not { matcher: Box<PermissionMatcher> },
}

impl PermissionMatcher {
    /// Convenience: match a string field by pointer + pattern.
    pub fn field(pointer: impl Into<String>, pattern: StringPattern) -> Self {
        Self::Field {
            pointer: pointer.into(),
            pattern,
        }
    }

    /// Convenience: match a string field by pointer + glob.
    pub fn field_glob(pointer: impl Into<String>, glob: impl Into<String>) -> Self {
        Self::Field {
            pointer: pointer.into(),
            pattern: StringPattern::glob(glob),
        }
    }

    /// Convenience: match a string field by pointer + prefix.
    pub fn field_prefix(pointer: impl Into<String>, prefix: impl Into<String>) -> Self {
        Self::Field {
            pointer: pointer.into(),
            pattern: StringPattern::prefix(prefix),
        }
    }

    /// Run the matcher against `input`. Returns `true` if the rule
    /// applies. Pure function with no side effects — safe to call from
    /// any thread.
    pub fn matches(&self, input: &serde_json::Value) -> bool {
        match self {
            Self::Always => true,
            Self::Field { pointer, pattern } => match input.pointer(pointer) {
                Some(serde_json::Value::String(s)) => pattern.matches(s),
                _ => false,
            },
            Self::ExactJson { value } => input == value,
            Self::AnyOf { matchers } => matchers.iter().any(|m| m.matches(input)),
            Self::AllOf { matchers } => matchers.iter().all(|m| m.matches(input)),
            Self::Not { matcher } => !matcher.matches(input),
        }
    }
}

/// String pattern for matching a single resolved field value.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum StringPattern {
    /// Exact string equality.
    Exact { value: String },
    /// Substring at the start.
    Prefix { value: String },
    /// Substring at the end.
    Suffix { value: String },
    /// Substring anywhere.
    Contains { value: String },
    /// Shell-style glob: `*` matches any sequence (including empty),
    /// `?` matches a single char. No bracket / brace expansion. Match
    /// is anchored on both ends — wrap the pattern in `*…*` for
    /// "contains".
    Glob { value: String },
}

impl StringPattern {
    /// Equivalent constructors with shorter call sites.
    pub fn exact(s: impl Into<String>) -> Self {
        Self::Exact { value: s.into() }
    }
    pub fn prefix(s: impl Into<String>) -> Self {
        Self::Prefix { value: s.into() }
    }
    pub fn suffix(s: impl Into<String>) -> Self {
        Self::Suffix { value: s.into() }
    }
    pub fn contains(s: impl Into<String>) -> Self {
        Self::Contains { value: s.into() }
    }
    pub fn glob(s: impl Into<String>) -> Self {
        Self::Glob { value: s.into() }
    }

    pub fn matches(&self, s: &str) -> bool {
        match self {
            Self::Exact { value } => s == value,
            Self::Prefix { value } => s.starts_with(value.as_str()),
            Self::Suffix { value } => s.ends_with(value.as_str()),
            Self::Contains { value } => s.contains(value.as_str()),
            Self::Glob { value } => glob_match(value, s),
        }
    }
}

/// Hand-rolled shell-glob matcher. Supports `*` (any chars) and `?`
/// (single char). Anchored on both ends. Iterative two-pointer with
/// backtracking on `*` — O(n*m) worst case but small constants and no
/// allocation. Operates on chars (not bytes) so multi-byte UTF-8
/// sequences match correctly.
fn glob_match(pattern: &str, text: &str) -> bool {
    let pat: Vec<char> = pattern.chars().collect();
    let txt: Vec<char> = text.chars().collect();
    let (mut p, mut t) = (0usize, 0usize);
    let (mut star_p, mut star_t): (Option<usize>, usize) = (None, 0);
    while t < txt.len() {
        if p < pat.len() && (pat[p] == '?' || pat[p] == txt[t]) {
            p += 1;
            t += 1;
        } else if p < pat.len() && pat[p] == '*' {
            star_p = Some(p);
            star_t = t;
            p += 1;
        } else if let Some(sp) = star_p {
            // Backtrack: extend the previous `*` match by one char.
            p = sp + 1;
            star_t += 1;
            t = star_t;
        } else {
            return false;
        }
    }
    while p < pat.len() && pat[p] == '*' {
        p += 1;
    }
    p == pat.len()
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Default, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionMode {
    #[default]
    Default,
    AcceptEdits,
    Bypass,
    Plan,
    DontAsk,
    Auto,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum PermissionBehavior {
    Allow,
    Deny,
    Ask,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum RuleSource {
    Policy,
    User,
    Project,
    Local,
    Flag,
    CliArg,
    Command,
    Session,
}

/// A permission rule applies to a specific tool plus, optionally, a
/// shape constraint on the tool's input.
///
/// - `matcher = PermissionMatcher::Always` (the default) covers the
///   **whole tool** regardless of input — equivalent to the legacy
///   `rule_content == None` form.
/// - Any other `matcher` variant inspects the tool input via JSON
///   pointer / pattern matching and only fires when the matcher
///   returns true.
///
/// The legacy `rule_content: Option<String>` field is retained for
/// wire-format compatibility with serialized rule files but is no
/// longer load-bearing for matching — use [`Self::with_matcher`] or
/// [`Self::with_input_match`] for new rules. When both `matcher` and
/// `rule_content` are present, `matcher` wins.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct PermissionRule {
    pub source: RuleSource,
    pub behavior: PermissionBehavior,
    pub tool_name: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub rule_content: Option<String>,
    /// Structured input matcher. `Always` (default) means whole-tool;
    /// see [`PermissionMatcher`] for the supported shapes.
    #[serde(default, skip_serializing_if = "matcher_is_always")]
    pub matcher: PermissionMatcher,
}

fn matcher_is_always(m: &PermissionMatcher) -> bool {
    matches!(m, PermissionMatcher::Always)
}

impl PermissionRule {
    /// Whole-tool rule — applies regardless of input shape.
    pub fn whole_tool(
        source: RuleSource,
        behavior: PermissionBehavior,
        tool_name: impl Into<String>,
    ) -> Self {
        Self {
            source,
            behavior,
            tool_name: tool_name.into(),
            rule_content: None,
            matcher: PermissionMatcher::Always,
        }
    }

    /// Rule that fires only when the input matches `matcher`. Use
    /// [`PermissionMatcher::field_glob`] / [`PermissionMatcher::field_prefix`]
    /// for the common Bash-command-glob and file-path-prefix cases.
    pub fn with_matcher(
        source: RuleSource,
        behavior: PermissionBehavior,
        tool_name: impl Into<String>,
        matcher: PermissionMatcher,
    ) -> Self {
        Self {
            source,
            behavior,
            tool_name: tool_name.into(),
            rule_content: None,
            matcher,
        }
    }

    /// Convenience: build a rule that matches a string field at
    /// `pointer` against `pattern`.
    pub fn with_input_match(
        source: RuleSource,
        behavior: PermissionBehavior,
        tool_name: impl Into<String>,
        pointer: impl Into<String>,
        pattern: StringPattern,
    ) -> Self {
        Self::with_matcher(
            source,
            behavior,
            tool_name,
            PermissionMatcher::field(pointer, pattern),
        )
    }

    /// Test whether this rule applies to a specific tool invocation.
    /// Returns `true` if the tool name matches AND the matcher accepts
    /// the input.
    ///
    /// **Legacy compatibility**: a rule with `rule_content = Some(_)`
    /// and the default `matcher = Always` is treated as an *ineffective
    /// rule* (no match) so old serialized rule files that carried a
    /// pattern string don't suddenly start matching every tool input
    /// when this code is upgraded. Migrate such rules to
    /// [`Self::with_input_match`] / [`Self::with_matcher`] to opt into
    /// structured matching.
    pub fn applies_to(&self, tool_name: &str, input: &serde_json::Value) -> bool {
        if self.tool_name != tool_name {
            return false;
        }
        if matches!(self.matcher, PermissionMatcher::Always) && self.rule_content.is_some() {
            return false;
        }
        self.matcher.matches(input)
    }
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
#[serde(tag = "kind", rename_all = "snake_case")]
pub enum DecisionReason {
    Rule {
        rule: PermissionRule,
    },
    Mode {
        mode: PermissionMode,
    },
    SafetyCheck {
        reason: String,
        classifier_approvable: bool,
    },
    Other {
        message: String,
    },
}

impl DecisionReason {
    pub fn rule(rule: PermissionRule) -> Self {
        Self::Rule { rule }
    }
    pub fn mode(mode: PermissionMode) -> Self {
        Self::Mode { mode }
    }
    pub fn other(message: impl Into<String>) -> Self {
        Self::Other {
            message: message.into(),
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct AllowDecision {
    /// Optionally an updated/sanitised input to pass on to the tool.
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub updated_input: Option<serde_json::Value>,
    pub reason: DecisionReason,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct AskDecision {
    pub message_text: String,
    #[serde(default, skip_serializing_if = "Option::is_none")]
    pub reason: Option<DecisionReason>,
}

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub struct DenyDecision {
    pub message_text: String,
    pub reason: DecisionReason,
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
#[serde(tag = "decision", rename_all = "snake_case")]
pub enum PermissionDecision {
    Allow(AllowDecision),
    Ask(AskDecision),
    Deny(DenyDecision),
}

impl PermissionDecision {
    pub fn is_allow(&self) -> bool {
        matches!(self, Self::Allow(_))
    }
    pub fn is_ask(&self) -> bool {
        matches!(self, Self::Ask(_))
    }
    pub fn is_deny(&self) -> bool {
        matches!(self, Self::Deny(_))
    }
}

/// Bundle of rules + mode flags that drive the 7-step chain. Build via
/// `PermissionContext::default()` + field assignments, or via the
/// [`crate::permission::PermissionManager`] builder.
#[derive(Debug, Clone, Default)]
pub struct PermissionContext {
    pub mode: PermissionMode,
    pub always_allow_rules: Vec<PermissionRule>,
    pub always_deny_rules: Vec<PermissionRule>,
    pub always_ask_rules: Vec<PermissionRule>,
    /// Set by the host to indicate the user has bypass available
    /// (some UIs hide the Bypass mode if the user lacks the role).
    pub is_bypass_available: bool,
    /// Set by the host to indicate auto-approve is available.
    pub is_auto_available: bool,
    /// Hint that the host is in a mode where prompting is undesirable
    /// (e.g., headless CI). Currently informational; see also `DontAsk` mode.
    pub should_avoid_prompts: bool,
}

#[cfg(test)]
mod matcher_tests {
    use super::*;

    #[test]
    fn glob_basic() {
        assert!(glob_match("*", ""));
        assert!(glob_match("*", "anything"));
        assert!(glob_match("rm -rf*", "rm -rf /home"));
        assert!(!glob_match("rm -rf*", "ls -la"));
        assert!(glob_match("foo?bar", "foo!bar"));
        assert!(!glob_match("foo?bar", "fooXXbar"));
        assert!(glob_match("a*b*c", "axxxbyyyc"));
        assert!(glob_match("", ""));
        assert!(!glob_match("", "x"));
    }

    #[test]
    fn glob_anchored_on_both_ends() {
        assert!(!glob_match("foo", "foobar"));
        assert!(!glob_match("foo", "barfoo"));
        assert!(glob_match("*foo*", "barfoobaz"));
    }

    #[test]
    fn glob_handles_multibyte() {
        assert!(glob_match("你好*", "你好世界"));
        assert!(glob_match("*界", "你好世界"));
        assert!(!glob_match("你?", "你好世"));
    }

    #[test]
    fn string_pattern_variants() {
        assert!(StringPattern::exact("rm").matches("rm"));
        assert!(!StringPattern::exact("rm").matches("rm -rf"));
        assert!(StringPattern::prefix("rm ").matches("rm -rf /"));
        assert!(StringPattern::suffix(".sh").matches("script.sh"));
        assert!(StringPattern::contains("etc").matches("/etc/passwd"));
        assert!(StringPattern::glob("/etc/*").matches("/etc/passwd"));
    }

    #[test]
    fn matcher_always_accepts_anything() {
        let m = PermissionMatcher::Always;
        assert!(m.matches(&serde_json::json!({})));
        assert!(m.matches(&serde_json::json!({"x": 1})));
        assert!(m.matches(&serde_json::Value::Null));
    }

    #[test]
    fn matcher_field_resolves_pointer() {
        let m = PermissionMatcher::field_glob("/command", "rm *");
        let danger = serde_json::json!({"command": "rm /tmp/x"});
        let safe = serde_json::json!({"command": "ls /tmp"});
        let nested = serde_json::json!({"args": {"command": "rm /tmp/x"}});
        assert!(m.matches(&danger));
        assert!(!m.matches(&safe));
        assert!(!m.matches(&nested), "wrong pointer should not match");
    }

    #[test]
    fn matcher_field_returns_false_when_field_missing_or_wrong_type() {
        let m = PermissionMatcher::field_glob("/command", "rm *");
        // Missing field.
        assert!(!m.matches(&serde_json::json!({})));
        // Field is not a string.
        assert!(!m.matches(&serde_json::json!({"command": 42})));
        assert!(!m.matches(&serde_json::json!({"command": null})));
    }

    #[test]
    fn matcher_exact_json_equals() {
        let m = PermissionMatcher::ExactJson {
            value: serde_json::json!({"a": 1}),
        };
        assert!(m.matches(&serde_json::json!({"a": 1})));
        assert!(!m.matches(&serde_json::json!({"a": 2})));
        assert!(!m.matches(&serde_json::json!({"a": 1, "b": 2})));
    }

    #[test]
    fn matcher_any_of_logical_or() {
        let m = PermissionMatcher::AnyOf {
            matchers: vec![
                PermissionMatcher::field_glob("/command", "rm *"),
                PermissionMatcher::field_glob("/command", "dd *"),
            ],
        };
        assert!(m.matches(&serde_json::json!({"command": "rm -rf"})));
        assert!(m.matches(&serde_json::json!({"command": "dd if=/x"})));
        assert!(!m.matches(&serde_json::json!({"command": "ls"})));
    }

    #[test]
    fn matcher_any_of_empty_never_matches() {
        let m = PermissionMatcher::AnyOf { matchers: vec![] };
        assert!(!m.matches(&serde_json::json!({})));
    }

    #[test]
    fn matcher_all_of_logical_and() {
        let m = PermissionMatcher::AllOf {
            matchers: vec![
                PermissionMatcher::field_prefix("/path", "/etc/"),
                PermissionMatcher::field_glob("/path", "*passwd*"),
            ],
        };
        assert!(m.matches(&serde_json::json!({"path": "/etc/passwd"})));
        assert!(!m.matches(&serde_json::json!({"path": "/etc/hosts"})));
        assert!(!m.matches(&serde_json::json!({"path": "/var/passwd"})));
    }

    #[test]
    fn matcher_all_of_empty_always_matches() {
        let m = PermissionMatcher::AllOf { matchers: vec![] };
        // Vacuous truth — treat empty AllOf as Always.
        assert!(m.matches(&serde_json::json!({})));
    }

    #[test]
    fn matcher_not_inverts() {
        let inner = PermissionMatcher::field_glob("/command", "rm *");
        let m = PermissionMatcher::Not {
            matcher: Box::new(inner),
        };
        assert!(!m.matches(&serde_json::json!({"command": "rm -rf"})));
        assert!(m.matches(&serde_json::json!({"command": "ls"})));
    }

    #[test]
    fn rule_applies_to_combines_tool_name_and_matcher() {
        let r = PermissionRule::with_input_match(
            RuleSource::Project,
            PermissionBehavior::Deny,
            "Bash",
            "/command",
            StringPattern::glob("rm -rf *"),
        );
        assert!(r.applies_to("Bash", &serde_json::json!({"command": "rm -rf /"})));
        assert!(!r.applies_to("Bash", &serde_json::json!({"command": "ls"})));
        assert!(!r.applies_to("FileEdit", &serde_json::json!({"command": "rm -rf /"})));
    }

    #[test]
    fn rule_serde_roundtrip_omits_default_matcher() {
        // Whole-tool rules should serialize without a matcher field
        // for backward compatibility with existing rule files.
        let r = PermissionRule::whole_tool(RuleSource::Project, PermissionBehavior::Allow, "Bash");
        let json = serde_json::to_string(&r).unwrap();
        assert!(!json.contains("matcher"), "got {json}");
        let back: PermissionRule = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }

    #[test]
    fn rule_serde_roundtrip_preserves_matcher() {
        let r = PermissionRule::with_input_match(
            RuleSource::Project,
            PermissionBehavior::Deny,
            "Bash",
            "/command",
            StringPattern::glob("rm -rf *"),
        );
        let json = serde_json::to_string(&r).unwrap();
        let back: PermissionRule = serde_json::from_str(&json).unwrap();
        assert_eq!(r, back);
    }
}
