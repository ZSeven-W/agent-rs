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
