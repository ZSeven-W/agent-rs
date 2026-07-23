//! Small process-spawn helpers shared by shell tools.

#[cfg(unix)]
#[allow(unsafe_code)]
pub(crate) fn detach_from_controlling_tty(cmd: &mut tokio::process::Command) {
    // `sudo` and similar programs can bypass stdin/stdout/stderr and prompt via
    // /dev/tty. Run shell tools in a new session so they have no controlling
    // terminal; stdout/stderr pipes still work, and the child pid remains the
    // process-group id for best-effort group kills.
    unsafe {
        cmd.pre_exec(|| {
            if libc::setsid() == -1 {
                Err(std::io::Error::last_os_error())
            } else {
                Ok(())
            }
        });
    }
}

#[cfg(not(unix))]
pub(crate) fn detach_from_controlling_tty(_cmd: &mut tokio::process::Command) {}

#[cfg(all(unix, feature = "shell"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProcessGroupState {
    Alive,
    Gone,
}

#[cfg(all(unix, feature = "shell"))]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub(crate) enum ProcessGroupSignal {
    Delivered,
    AlreadyGone,
}

#[cfg(all(unix, feature = "shell"))]
#[allow(unsafe_code)]
pub(crate) fn kill_process_group(pid: u32) -> std::io::Result<ProcessGroupSignal> {
    let pid = i32::try_from(pid)
        .map_err(|_| std::io::Error::other("process group id exceeds platform range"))?;
    if unsafe { libc::killpg(pid, libc::SIGKILL) } == 0 {
        return Ok(ProcessGroupSignal::Delivered);
    }
    let error = std::io::Error::last_os_error();
    if error.raw_os_error() == Some(libc::ESRCH) {
        Ok(ProcessGroupSignal::AlreadyGone)
    } else {
        Err(error)
    }
}

#[cfg(all(unix, feature = "shell"))]
#[allow(unsafe_code)]
pub(crate) fn process_group_state(pid: u32) -> std::io::Result<ProcessGroupState> {
    let pid = i32::try_from(pid)
        .map_err(|_| std::io::Error::other("process group id exceeds platform range"))?;
    if unsafe { libc::killpg(pid, 0) } == 0 {
        return Ok(ProcessGroupState::Alive);
    }
    let error = std::io::Error::last_os_error();
    match error.raw_os_error() {
        Some(libc::ESRCH) => Ok(ProcessGroupState::Gone),
        Some(libc::EPERM) => Ok(ProcessGroupState::Alive),
        _ => Err(error),
    }
}
