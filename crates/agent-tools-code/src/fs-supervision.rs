use std::io;
use std::path::Path;

use agent::abort::AbortController;

async fn run_tracked<F>(abort: &AbortController, mutation: F) -> io::Result<()>
where
    F: FnOnce() -> io::Result<()> + Send + 'static,
{
    if abort.is_aborted() {
        return Err(io::Error::new(
            io::ErrorKind::Interrupted,
            abort
                .reason()
                .unwrap_or_else(|| "filesystem mutation cancelled".into()),
        ));
    }

    let work = abort.activity().track_worker();
    tokio::task::spawn_blocking(move || {
        let _work = work;
        mutation()
    })
    .await
    .map_err(|error| io::Error::other(format!("filesystem mutation worker failed: {error}")))?
}

pub(crate) async fn write_file(
    path: &Path,
    bytes: &[u8],
    abort: &AbortController,
) -> io::Result<()> {
    let path = path.to_path_buf();
    let bytes = bytes.to_vec();
    run_tracked(abort, move || std::fs::write(path, bytes)).await
}

pub(crate) async fn create_dir(
    path: &Path,
    recursive: bool,
    abort: &AbortController,
) -> io::Result<()> {
    let path = path.to_path_buf();
    run_tracked(abort, move || {
        if recursive {
            std::fs::create_dir_all(path)
        } else {
            std::fs::create_dir(path)
        }
    })
    .await
}

pub(crate) async fn rename(from: &Path, to: &Path, abort: &AbortController) -> io::Result<()> {
    let from = from.to_path_buf();
    let to = to.to_path_buf();
    run_tracked(abort, move || std::fs::rename(from, to)).await
}

pub(crate) async fn remove(
    path: &Path,
    recursive: bool,
    is_dir: bool,
    abort: &AbortController,
) -> io::Result<()> {
    let path = path.to_path_buf();
    run_tracked(abort, move || {
        if is_dir {
            if recursive {
                std::fs::remove_dir_all(path)
            } else {
                std::fs::remove_dir(path)
            }
        } else {
            std::fs::remove_file(path)
        }
    })
    .await
}

#[cfg(test)]
mod tests {
    use std::time::Duration;

    use super::*;

    #[tokio::test]
    async fn caller_drop_does_not_hide_running_mutation() {
        let abort = AbortController::new();
        let (started_tx, started_rx) = tokio::sync::oneshot::channel();
        let (release_tx, release_rx) = std::sync::mpsc::channel();
        let worker_abort = abort.clone();
        let caller = tokio::spawn(async move {
            run_tracked(&worker_abort, move || {
                let _ = started_tx.send(());
                release_rx.recv().unwrap();
                Ok(())
            })
            .await
        });

        tokio::time::timeout(Duration::from_secs(1), started_rx)
            .await
            .unwrap()
            .unwrap();
        caller.abort();
        let _ = caller.await;
        assert_eq!(abort.activity().active_workers(), 1);
        assert!(tokio::time::timeout(
            Duration::from_millis(25),
            abort.activity().wait_for_quiescence()
        )
        .await
        .is_err());

        let _ = release_tx.send(());
        tokio::time::timeout(
            Duration::from_secs(1),
            abort.activity().wait_for_quiescence(),
        )
        .await
        .unwrap();
    }
}
