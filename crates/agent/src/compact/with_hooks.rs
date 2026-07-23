//! Compaction wrapper that fires PreCompact + PostCompact hooks.
//!
//! Mirror of `services/compact/compactWarningHook.ts` + the hook-firing
//! surrounding `compactConversation` in `services/compact/compact.ts`.
//!
//! Run [`compact_with_hooks`] instead of [`super::compact_conversation`]
//! when a [`HookRunner`] is available. Pre/Post events fire even on
//! provider failure (the post hook carries `replaced_count = 0` if
//! the model never responded usefully).

use super::summarize::{compact_conversation, CompactError, CompactionResult};
use super::PartialCompactDirection;
use crate::abort::AbortController;
use crate::hook::{HookEvent, HookOutcome, HookRunner};
use crate::message::Message;
use crate::provider::Provider;

/// What kicked off this compaction call.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum CompactTrigger {
    /// Fired by the QueryLoop's auto-compact threshold.
    Auto,
    /// Fired by an explicit user / API request.
    Manual,
}

impl CompactTrigger {
    pub fn as_str(self) -> &'static str {
        match self {
            Self::Auto => "auto",
            Self::Manual => "manual",
        }
    }
}

/// Bundle of inputs for a hook-instrumented compaction call. Held as
/// a builder so the entry point doesn't violate clippy's
/// `too_many_arguments` lint (8 separate params became opaque).
#[derive(Debug)]
pub struct CompactWithHooksRequest<'a> {
    pub messages: &'a [Message],
    pub model: String,
    pub custom_instructions: Option<&'a str>,
    pub direction: PartialCompactDirection,
    pub trigger: CompactTrigger,
    pub abort: AbortController,
}

impl<'a> CompactWithHooksRequest<'a> {
    pub fn new(messages: &'a [Message], model: impl Into<String>) -> Self {
        Self {
            messages,
            model: model.into(),
            custom_instructions: None,
            direction: PartialCompactDirection::default(),
            trigger: CompactTrigger::Auto,
            abort: AbortController::new(),
        }
    }

    pub fn with_custom_instructions(mut self, ci: &'a str) -> Self {
        self.custom_instructions = Some(ci);
        self
    }

    pub fn with_direction(mut self, d: PartialCompactDirection) -> Self {
        self.direction = d;
        self
    }

    pub fn with_trigger(mut self, t: CompactTrigger) -> Self {
        self.trigger = t;
        self
    }

    pub fn with_abort(mut self, a: AbortController) -> Self {
        self.abort = a;
        self
    }
}

/// Run a compaction request through the hook lifecycle:
///
/// 1. Fire [`HookEvent::PreCompact`] — if any handler returns
///    [`HookOutcome::Block`], abort with [`CompactError::Aborted`]
///    (the hook system treats Block as "do not perform the action").
/// 2. Run [`compact_conversation`].
/// 3. Fire [`HookEvent::PostCompact`] regardless of success or
///    failure (with `replaced_count = 0` on failure).
/// 4. Return the original result / error.
pub async fn compact_with_hooks(
    hooks: &HookRunner,
    provider: &dyn Provider,
    request: CompactWithHooksRequest<'_>,
) -> Result<CompactionResult, CompactError> {
    let CompactWithHooksRequest {
        messages,
        model,
        custom_instructions,
        direction,
        trigger,
        abort,
    } = request;
    let pre_outcome = hooks
        .run_with_abort(
            &HookEvent::PreCompact {
                trigger: trigger.as_str().into(),
                custom_instructions: custom_instructions.map(String::from),
            },
            &abort,
        )
        .await;
    if matches!(pre_outcome, HookOutcome::Block) {
        return Err(CompactError::Aborted);
    }

    let result = compact_conversation(
        messages,
        provider,
        model.clone(),
        custom_instructions,
        direction,
        abort.clone(),
    )
    .await;

    match &result {
        Ok(r) => {
            let _ = hooks
                .run_with_abort(
                    &HookEvent::PostCompact {
                        pre_tokens: r.pre_compact_tokens,
                        post_tokens: r.post_compact_tokens,
                        replaced_count: r.replaced_uuids.len() as u32,
                    },
                    &abort,
                )
                .await;
        }
        Err(_) => {
            let _ = hooks
                .run_with_abort(
                    &HookEvent::PostCompact {
                        pre_tokens: messages
                            .iter()
                            .map(super::estimate_tokens)
                            .fold(0u32, u32::saturating_add),
                        post_tokens: 0,
                        replaced_count: 0,
                    },
                    &abort,
                )
                .await;
        }
    }

    result
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicUsize, Ordering};
    use std::sync::{Arc, Mutex};

    use async_trait::async_trait;
    use futures::stream;

    use super::*;
    use crate::error::AgentError;
    use crate::hook::{HookHandler, RustHookHandler};
    use crate::message::{ContentBlock, Header};
    use crate::provider::{ProviderCapabilities, StreamRequest};
    use crate::stream::{Event, EventStream};

    fn happy_response() -> &'static str {
        "<analysis>- did X</analysis><summary>Did X.</summary>"
    }

    #[derive(Debug)]
    struct ScriptedProvider(Mutex<Vec<Event>>);

    impl ScriptedProvider {
        fn from_text(t: &str) -> Self {
            Self(Mutex::new(vec![
                Event::TextDelta { delta: t.into() },
                Event::Result {
                    data: Default::default(),
                },
            ]))
        }
    }

    #[async_trait]
    impl Provider for ScriptedProvider {
        fn id(&self) -> &str {
            "scripted"
        }
        fn capabilities(&self) -> ProviderCapabilities {
            ProviderCapabilities::default()
        }
        async fn stream(
            &self,
            _req: StreamRequest,
            _abort: AbortController,
        ) -> Result<Box<dyn EventStream>, AgentError> {
            let evs: Vec<Event> = self.0.lock().map(|mut g| std::mem::take(&mut *g)).unwrap();
            Ok(Box::new(stream::iter(evs.into_iter().map(Ok))))
        }
    }

    fn user(text: &str) -> Message {
        Message::User {
            header: Header::new(),
            content: vec![ContentBlock::Text { text: text.into() }],
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fires_pre_then_post_on_success() {
        let counter = Arc::new(AtomicUsize::new(0));
        let pre_seen = counter.clone();
        let post_seen = counter.clone();
        let pre_handler = Arc::new(RustHookHandler::new("pre", move |e| {
            if matches!(e, HookEvent::PreCompact { .. }) {
                pre_seen.fetch_add(1, Ordering::SeqCst);
            }
            HookOutcome::Ok
        }));
        let post_handler = Arc::new(RustHookHandler::new("post", move |e| {
            if matches!(e, HookEvent::PostCompact { .. }) {
                post_seen.fetch_add(1, Ordering::SeqCst);
            }
            HookOutcome::Ok
        }));
        let mut runner = HookRunner::new();
        runner.register(pre_handler);
        runner.register(post_handler);
        let provider = ScriptedProvider::from_text(happy_response());
        let messages = vec![user("a"), user("b")];
        let result = compact_with_hooks(
            &runner,
            &provider,
            CompactWithHooksRequest::new(&messages, "m").with_trigger(CompactTrigger::Auto),
        )
        .await
        .unwrap();
        assert_eq!(counter.load(Ordering::SeqCst), 2);
        assert_eq!(result.replaced_uuids.len(), 2);
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn pre_compact_block_short_circuits_with_aborted() {
        let blocker: Arc<dyn HookHandler> = Arc::new(RustHookHandler::new("blocker", |e| {
            if matches!(e, HookEvent::PreCompact { .. }) {
                HookOutcome::Block
            } else {
                HookOutcome::Ok
            }
        }));
        let mut runner = HookRunner::new();
        runner.register(blocker);
        let provider = ScriptedProvider::from_text(happy_response());
        let messages = vec![user("a"), user("b")];
        match compact_with_hooks(
            &runner,
            &provider,
            CompactWithHooksRequest::new(&messages, "m").with_trigger(CompactTrigger::Manual),
        )
        .await
        {
            Err(CompactError::Aborted) => {}
            other => panic!("expected Aborted, got {other:?}"),
        }
    }

    #[tokio::test(flavor = "multi_thread", worker_threads = 4)]
    async fn fires_post_with_zero_replaced_on_failure() {
        let post_seen = Arc::new(AtomicUsize::new(0));
        let saw_zero = Arc::new(AtomicUsize::new(0));
        let post_seen_clone = post_seen.clone();
        let saw_zero_clone = saw_zero.clone();
        let post: Arc<dyn HookHandler> = Arc::new(RustHookHandler::new("post-watch", move |e| {
            if let HookEvent::PostCompact { replaced_count, .. } = e {
                post_seen_clone.fetch_add(1, Ordering::SeqCst);
                if *replaced_count == 0 {
                    saw_zero_clone.fetch_add(1, Ordering::SeqCst);
                }
            }
            HookOutcome::Ok
        }));
        let mut runner = HookRunner::new();
        runner.register(post);
        // Empty-response provider triggers compact failure.
        let provider = ScriptedProvider::from_text("");
        let messages = vec![user("a"), user("b")];
        let result = compact_with_hooks(
            &runner,
            &provider,
            CompactWithHooksRequest::new(&messages, "m").with_trigger(CompactTrigger::Auto),
        )
        .await;
        assert!(result.is_err());
        assert_eq!(post_seen.load(Ordering::SeqCst), 1);
        assert_eq!(saw_zero.load(Ordering::SeqCst), 1);
    }
}
