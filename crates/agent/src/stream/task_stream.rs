//! Event stream backed by one owned Tokio task.

use std::future::Future;
use std::pin::Pin;
use std::task::{Context, Poll};

use futures::channel::mpsc;
use futures::Stream;

use crate::error::AgentError;

use super::Event;

/// Owns a producer task so stream EOF is not mistaken for success when that
/// task panics or is cancelled. Dropping the stream hard-cancels the producer.
#[derive(Debug)]
pub(crate) struct TaskEventStream {
    rx: mpsc::UnboundedReceiver<Result<Event, AgentError>>,
    task: Option<tokio::task::JoinHandle<()>>,
    label: &'static str,
    terminal_error: Option<AgentError>,
}

impl TaskEventStream {
    pub(crate) fn new(
        rx: mpsc::UnboundedReceiver<Result<Event, AgentError>>,
        task: tokio::task::JoinHandle<()>,
        label: &'static str,
    ) -> Self {
        Self {
            rx,
            task: Some(task),
            label,
            terminal_error: None,
        }
    }
}

impl Stream for TaskEventStream {
    type Item = Result<Event, AgentError>;

    fn poll_next(mut self: Pin<&mut Self>, cx: &mut Context<'_>) -> Poll<Option<Self::Item>> {
        match Pin::new(&mut self.rx).poll_next(cx) {
            Poll::Ready(Some(Err(error))) => {
                if self.terminal_error.is_none() {
                    self.terminal_error = Some(error);
                }
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(Some(Ok(_))) if self.terminal_error.is_some() => {
                cx.waker().wake_by_ref();
                Poll::Pending
            }
            Poll::Ready(Some(item)) => Poll::Ready(Some(item)),
            Poll::Pending => Poll::Pending,
            Poll::Ready(None) => {
                let Some(task) = self.task.as_mut() else {
                    return Poll::Ready(None);
                };
                match Pin::new(task).poll(cx) {
                    Poll::Pending => Poll::Pending,
                    Poll::Ready(Ok(())) => {
                        self.task.take();
                        match self.terminal_error.take() {
                            Some(error) => Poll::Ready(Some(Err(error))),
                            None => Poll::Ready(None),
                        }
                    }
                    Poll::Ready(Err(error)) => {
                        self.task.take();
                        if self.terminal_error.is_none() {
                            self.terminal_error = Some(AgentError::other(format!(
                                "{} task failed: {error}",
                                self.label
                            )));
                        }
                        Poll::Ready(self.terminal_error.take().map(Err))
                    }
                }
            }
        }
    }
}

impl Drop for TaskEventStream {
    fn drop(&mut self) {
        if let Some(task) = self.task.take() {
            task.abort();
        }
    }
}

#[cfg(test)]
mod tests {
    use std::sync::atomic::{AtomicBool, Ordering};
    use std::sync::Arc;

    use futures::StreamExt;

    use super::*;

    #[tokio::test]
    async fn producer_panic_is_an_error_not_clean_eof() {
        let (tx, rx) = mpsc::unbounded();
        let task = tokio::spawn(async move {
            drop(tx);
            panic!("parser panic probe");
        });
        let mut stream = TaskEventStream::new(rx, task, "parser");

        let error = stream.next().await.unwrap().unwrap_err();
        assert!(error.to_string().contains("parser task failed"));
        assert!(stream.next().await.is_none());
    }

    #[tokio::test]
    async fn drop_keeps_quiescence_fence_until_producer_is_destroyed() {
        struct DropProbe(Arc<AtomicBool>);
        impl Drop for DropProbe {
            fn drop(&mut self) {
                self.0.store(true, Ordering::SeqCst);
            }
        }

        let abort = crate::abort::AbortController::new();
        let activity = abort.activity();
        let work = activity.track_worker();
        let dropped = Arc::new(AtomicBool::new(false));
        let task_dropped = dropped.clone();
        let started = Arc::new(tokio::sync::Notify::new());
        let task_started = started.clone();
        let (tx, rx) = mpsc::unbounded();
        let task = tokio::spawn(async move {
            let _work = work;
            let _probe = DropProbe(task_dropped);
            let _tx = tx;
            task_started.notify_one();
            futures::future::pending::<()>().await;
        });
        started.notified().await;
        let stream = TaskEventStream::new(rx, task, "drop probe");
        assert_eq!(activity.active_workers(), 1);

        drop(stream);

        tokio::time::timeout(
            std::time::Duration::from_secs(1),
            activity.wait_for_quiescence(),
        )
        .await
        .expect("producer drop should acknowledge quiescence");
        assert!(dropped.load(Ordering::SeqCst));
        assert_eq!(activity.active_workers(), 0);
    }
}
