//! Lifecycle for the Ormah server daemon.
//!
//! The server runs as an independent daemon (`ormah server start -d`), not
//! as a child of this process — closing the app leaves it running. The app
//! can start/stop it on demand via start_daemon() / stop_daemon().
//!
//! On first launch the bundled `uv` sidecar installs the `ormah` Python
//! package from PyPI. Subsequent launches skip the install and start the
//! server directly. Falls back to a system `ormah` on PATH when the `uv`
//! sidecar is absent (dev builds).

use crate::diagnostics;
use std::path::PathBuf;
use std::process::Stdio;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Runtime};
use tauri_plugin_shell::process::{Command, CommandEvent};
use tauri_plugin_shell::ShellExt;

// Injected by build.rs from the repo's pyproject.toml, so the pinned Python
// package version can never drift from the released one.
const ORMAH_VERSION: &str = env!("ORMAH_PY_VERSION");
const PYTHON_VERSION: &str = "3.11";
const RETRY_DELAYS: [u64; 4] = [10, 20, 30, 45];
const INSTALL_ATTEMPTS: u8 = RETRY_DELAYS.len() as u8 + 1;
const INSTALL_TIMEOUT: Duration = Duration::from_secs(600);
// `server start -d` gives up after 300s, or 600s on first run, while the server
// may still be booting. A failure this slow gets extra time to become healthy.
const SLOW_DAEMON_START: Duration = Duration::from_secs(250);
const SLOW_DAEMON_GRACE: Duration = Duration::from_secs(300);
// Above the CLI's own 600s wait, so only a truly hung command gets killed.
const DAEMON_START_TIMEOUT: Duration = Duration::from_secs(660);
const DAEMON_STOP_TIMEOUT: Duration = Duration::from_secs(60);
const VERSION_TIMEOUT: Duration = Duration::from_secs(30);
static STATUS: Mutex<Phase> = Mutex::new(Phase::Starting);
static SETUP_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Setup state, polled by the splash screen. Only Ready means the server is up.
#[derive(Clone, serde::Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum Phase {
    Installing {
        version: &'static str,
        attempt: u8,
        attempts: u8,
    },
    Retrying {
        attempt: u8,
        attempts: u8,
    },
    Starting,
    Ready,
    Failed {
        reason: String,
        /// A server from before a failed upgrade is still healthy and usable.
        can_open: bool,
    },
}

pub fn status() -> Phase {
    STATUS.lock().map(|s| s.clone()).unwrap_or(Phase::Starting)
}

fn publish(phase: Phase) {
    diagnostics::log(&format!(
        "status {}",
        serde_json::to_string(&phase).unwrap_or_default()
    ));
    if let Ok(mut status) = STATUS.lock() {
        *status = phase;
    }
}

/// Start runtime setup once. Returns false when another attempt is still active.
pub fn start<R: Runtime>(app: AppHandle<R>) -> bool {
    if SETUP_IN_PROGRESS
        .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
        .is_err()
    {
        return false;
    }
    // Publish synchronously so an accepted retry replaces the previous Failed.
    publish(Phase::Starting);
    tauri::async_runtime::spawn(async move {
        let phase = match ensure_running(&app).await {
            Ok(()) => Phase::Ready,
            Err(reason) => Phase::Failed {
                reason,
                can_open: is_running().await,
            },
        };
        // Release before publishing: Try again only appears once Failed is visible.
        SETUP_IN_PROGRESS.store(false, Ordering::Release);
        publish(phase);
    });
    true
}

async fn ensure_running<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    log_platform(app).await;
    let installed = find_ormah().is_some();
    let runtime_action = choose_runtime_action(installed, installed && needs_upgrade());
    diagnostics::log(&format!("runtime action={runtime_action:?}"));
    let runtime_changed = match runtime_action {
        RuntimeAction::Install | RuntimeAction::Upgrade => {
            if let Err(failure) = install_with_retry(app).await {
                // A failed upgrade keeps the old runtime: start it so memories stay reachable.
                if runtime_action == RuntimeAction::Upgrade {
                    let _ = start_daemon().await;
                }
                return Err(failure.public_reason().to_string());
            }
            true
        }
        RuntimeAction::Reuse => false,
    };

    // Reinstall replaces disk files; restart the daemon to load the new package.
    if runtime_changed {
        let _ = tauri::async_runtime::spawn_blocking(stop_daemon).await;
    }
    publish(Phase::Starting);
    start_daemon().await
}

async fn log_platform<R: Runtime>(app: &AppHandle<R>) {
    let os_version = std::process::Command::new("uname")
        .arg("-srv")
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    // Names only: values can contain proxy credentials.
    let mut environment_names: Vec<_> = std::env::vars_os()
        .map(|(key, _)| key.to_string_lossy().into_owned())
        .filter(|key| {
            key.starts_with("UV_")
                || ["http_proxy", "https_proxy", "all_proxy", "no_proxy"]
                    .contains(&key.to_ascii_lowercase().as_str())
        })
        .collect();
    environment_names.sort();
    diagnostics::log(&format!(
        "app={} pinned_ormah={ORMAH_VERSION} os={} version={os_version} arch={} environment_names={environment_names:?}",
        env!("CARGO_PKG_VERSION"),
        std::env::consts::OS,
        std::env::consts::ARCH
    ));
    if let Ok(uv) = app.shell().sidecar("uv") {
        let _ = run_uv(uv.arg("--version"), Duration::from_secs(10)).await;
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq)]
enum RuntimeAction {
    Install,
    Upgrade,
    Reuse,
}

fn choose_runtime_action(installed: bool, upgrade_needed: bool) -> RuntimeAction {
    if !installed {
        RuntimeAction::Install
    } else if upgrade_needed {
        RuntimeAction::Upgrade
    } else {
        RuntimeAction::Reuse
    }
}

/// Find the ormah binary. Checks uv tool install locations first (GUI apps
/// don't inherit the user's shell PATH, so ~/.local/bin is often missing),
/// then falls back to a PATH search.
pub fn find_ormah() -> Option<PathBuf> {
    let home = std::env::var("HOME").ok()?;

    let candidates = [
        format!("{home}/.local/bin/ormah"),
        format!("{home}/.local/share/uv/tools/ormah/bin/ormah"),
        format!("{home}/.local/share/mise/shims/ormah"),
        "/usr/local/bin/ormah".to_string(),
        "/opt/homebrew/bin/ormah".to_string(),
    ];

    for path in &candidates {
        let p = PathBuf::from(path);
        if p.exists() {
            return Some(p);
        }
    }

    std::process::Command::new("which")
        .arg("ormah")
        .output()
        .ok()
        .and_then(|o| {
            if o.status.success() {
                let s = String::from_utf8_lossy(&o.stdout).trim().to_string();
                if s.is_empty() {
                    None
                } else {
                    Some(PathBuf::from(s))
                }
            } else {
                None
            }
        })
}

/// Start the ormah server as an independent background daemon.
/// Safe to call when already running — ormah server start is idempotent.
pub async fn start_daemon() -> Result<(), String> {
    let started = Instant::now();
    let result = tauri::async_runtime::spawn_blocking(run_daemon_command)
        .await
        .map_err(|e| e.to_string())
        .and_then(|r| r);
    let Err(error) = result else { return Ok(()) };
    let elapsed = started.elapsed();
    diagnostics::log(&format!(
        "daemon start failed after {}s: {error}",
        elapsed.as_secs()
    ));
    let deadline = Instant::now()
        + if elapsed >= SLOW_DAEMON_START {
            SLOW_DAEMON_GRACE
        } else {
            Duration::ZERO
        };
    loop {
        if is_running().await {
            diagnostics::log("daemon command failed but the health check passed");
            return Ok(());
        }
        if Instant::now() >= deadline {
            return Err("Ormah could not start its server. Try again, or copy the error log for help.".into());
        }
        tokio::time::sleep(Duration::from_secs(2)).await;
    }
}

fn run_daemon_command() -> Result<(), String> {
    let bin = find_ormah().ok_or("Ormah runtime was not found")?;
    let mut command = clean_python_env(std::process::Command::new(&bin));
    command.args(["server", "start", "-d"]);
    // Files, not pipes: a fallback daemon inherits these and outlives this command.
    match diagnostics::daemon_output() {
        Some((stdout, stderr)) => command.stdout(stdout).stderr(stderr),
        None => command.stdout(Stdio::null()).stderr(Stdio::null()),
    };
    let status = wait_bounded(&mut command, DAEMON_START_TIMEOUT).map_err(|e| e.to_string())?;
    diagnostics::log(&format!("daemon command={bin:?} status={status}"));
    if status.success() {
        Ok(())
    } else {
        Err(format!("exit status {status}"))
    }
}

/// Run a command, killing it at the deadline so setup can always finish.
fn wait_bounded(
    command: &mut std::process::Command,
    timeout: Duration,
) -> std::io::Result<std::process::ExitStatus> {
    let mut child = command.spawn()?;
    let started = Instant::now();
    loop {
        if let Some(status) = child.try_wait()? {
            return Ok(status);
        }
        if started.elapsed() >= timeout {
            let _ = child.kill();
            let _ = child.wait();
            return Err(std::io::Error::new(
                std::io::ErrorKind::TimedOut,
                format!("killed after {}s", timeout.as_secs()),
            ));
        }
        std::thread::sleep(Duration::from_millis(100));
    }
}

/// Stop the ormah daemon. No-op if not running.
pub fn stop_daemon() {
    let Some(bin) = find_ormah() else { return };
    let mut command = clean_python_env(std::process::Command::new(&bin));
    command.args(["server", "stop"]);
    if let Err(error) = wait_bounded(&mut command, DAEMON_STOP_TIMEOUT) {
        diagnostics::log(&format!("daemon stop failed: {error}"));
    }
}

/// Ping /admin/health — true means the server is up and responding.
pub async fn is_running() -> bool {
    let url = format!("{}/admin/health", crate::commands::base_url());
    let Ok(client) = crate::commands::local_client(Duration::from_secs(2)) else {
        return false;
    };
    client
        .get(&url)
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn needs_upgrade() -> bool {
    let Some(bin) = find_ormah() else {
        return false;
    };
    let mut command = clean_python_env(std::process::Command::new(bin));
    command
        .arg("--version")
        .stderr(Stdio::null())
        .stdout(Stdio::piped());
    // A version line fits in the pipe buffer, so waiting before reading can't deadlock.
    match spawn_output_bounded(&mut command) {
        Some(stdout) => should_upgrade(&stdout, ORMAH_VERSION),
        None => true,
    }
}

fn spawn_output_bounded(command: &mut std::process::Command) -> Option<Vec<u8>> {
    let mut child = command.spawn().ok()?;
    let started = Instant::now();
    loop {
        match child.try_wait().ok()? {
            Some(status) if status.success() => break,
            Some(_) => return None,
            None if started.elapsed() >= VERSION_TIMEOUT => {
                let _ = child.kill();
                let _ = child.wait();
                return None;
            }
            None => std::thread::sleep(Duration::from_millis(100)),
        }
    }
    let mut stdout = Vec::new();
    std::io::Read::read_to_end(&mut child.stdout.take()?, &mut stdout).ok()?;
    Some(stdout)
}

fn should_upgrade(output: &[u8], required: &str) -> bool {
    let Ok(required) = semver::Version::parse(required) else {
        return true;
    };
    let installed = String::from_utf8_lossy(output)
        .split_whitespace()
        .find_map(|word| semver::Version::parse(word).ok());
    match installed {
        Some(installed) => installed < required,
        None => true,
    }
}

/// Remove Python env vars that AppImage sets and that corrupt child Python runtimes.
/// AppImage mounts itself and sets PYTHONHOME/PYTHONPATH to paths inside the
/// mount — any subprocess that spawns Python inherits them and fails to find
/// the stdlib with "No module named 'encodings'".
pub(crate) fn clean_python_env(mut cmd: std::process::Command) -> std::process::Command {
    for var in &[
        "PYTHONHOME",
        "PYTHONPATH",
        "PYTHONSTARTUP",
        "PYTHONEXECUTABLE",
    ] {
        cmd.env_remove(var);
    }
    cmd
}

#[derive(Debug, PartialEq)]
enum InstallFailure {
    Network,
    PythonDownload,
    TimedOut,
    Other,
}

impl InstallFailure {
    fn from_stderr(stderr: &str) -> Self {
        let stderr = stderr.to_ascii_lowercase();
        if !transient_install_error(&stderr) {
            Self::Other
        } else if stderr.contains("cpython-") {
            Self::PythonDownload
        } else {
            Self::Network
        }
    }

    fn public_reason(&self) -> &'static str {
        match self {
            Self::Network => {
                "Ormah could not reach the package service. Check your connection, then try again."
            }
            Self::PythonDownload => {
                "Ormah could not download Python. Check your connection, then try again."
            }
            Self::TimedOut => {
                "Installing Ormah took too long. Try again, or copy the error log for help."
            }
            Self::Other => {
                "Ormah could not install its runtime. Try again, or copy the error log for help."
            }
        }
    }
}

fn transient_install_error(details: &str) -> bool {
    let details = details.to_ascii_lowercase();
    [
        "dns error",
        "failed to lookup address",
        "temporary failure in name resolution",
        "timed out",
        "network timeout",
        "connection reset",
        "connection refused",
        "network is unreachable",
    ]
    .iter()
    .any(|pattern| details.contains(pattern))
}

fn install_args(version: &str) -> Vec<String> {
    let spec = format!("ormah[litellm]=={version}");
    [
        "tool", "install", "--reinstall", "--force", "--python", PYTHON_VERSION, &spec,
    ]
    .map(String::from)
    .to_vec()
}

async fn install_with_retry<R: Runtime>(app: &AppHandle<R>) -> Result<(), InstallFailure> {
    for attempt in 1..=INSTALL_ATTEMPTS {
        publish(Phase::Installing {
            version: ORMAH_VERSION,
            attempt,
            attempts: INSTALL_ATTEMPTS,
        });
        let failure = match install_via_uv(app).await {
            Ok(()) => return Ok(()),
            Err(failure) => failure,
        };
        let delay = RETRY_DELAYS.get(attempt as usize - 1);
        match delay {
            Some(delay) if failure != InstallFailure::Other => {
                publish(Phase::Retrying {
                    attempt: attempt + 1,
                    attempts: INSTALL_ATTEMPTS,
                });
                tokio::time::sleep(Duration::from_secs(*delay)).await;
            }
            _ => return Err(failure),
        }
    }
    unreachable!("the bounded install loop always returns")
}

async fn install_via_uv<R: Runtime>(app: &AppHandle<R>) -> Result<(), InstallFailure> {
    let uv = app.shell().sidecar("uv").map_err(|e| {
        diagnostics::log(&format!("uv sidecar not available: {e}"));
        InstallFailure::Other
    })?;
    let args = install_args(ORMAH_VERSION);
    diagnostics::log(&format!("uv arguments: {args:?}"));
    run_uv(uv.args(args), INSTALL_TIMEOUT).await
}

/// Run uv, logging each line as it arrives so a hung or killed run leaves evidence.
async fn run_uv(command: Command, timeout: Duration) -> Result<(), InstallFailure> {
    let started = Instant::now();
    let (mut rx, child) = command.spawn().map_err(|e| {
        diagnostics::log(&format!("uv spawn failed: {e}"));
        InstallFailure::from_stderr(&e.to_string())
    })?;
    let mut stderr = String::new();
    let result = tokio::time::timeout(timeout, async {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stderr(line) | CommandEvent::Stdout(line) => {
                    let line = String::from_utf8_lossy(&line);
                    diagnostics::log(&format!("uv: {}", line.trim_end()));
                    stderr.push_str(&line);
                }
                CommandEvent::Error(error) => {
                    diagnostics::log(&format!("uv error: {error}"));
                    stderr.push_str(&error);
                }
                CommandEvent::Terminated(status) => {
                    diagnostics::log(&format!(
                        "uv exit={:?} signal={:?} after {}s",
                        status.code,
                        status.signal,
                        started.elapsed().as_secs()
                    ));
                    return status.code == Some(0);
                }
                _ => {}
            }
        }
        false
    })
    .await;
    match result {
        Ok(true) => Ok(()),
        Ok(false) => Err(InstallFailure::from_stderr(&stderr)),
        Err(_) => {
            diagnostics::log(&format!(
                "uv killed after {}s timeout: {:?}",
                timeout.as_secs(),
                child.kill()
            ));
            Err(InstallFailure::TimedOut)
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn pins_python_and_the_exact_package() {
        assert_eq!(
            install_args("1.2.3"),
            [
                "tool",
                "install",
                "--reinstall",
                "--force",
                "--python",
                "3.11",
                "ormah[litellm]==1.2.3"
            ]
        );
        assert_eq!(INSTALL_ATTEMPTS, 5);
    }

    #[test]
    fn classifies_real_uv_failures() {
        for (stderr, expected) in [
            (include_str!("fixtures/uv-pypi-down.txt"), InstallFailure::Network),
            (include_str!("fixtures/uv-python-down.txt"), InstallFailure::PythonDownload),
            (include_str!("fixtures/uv-no-solution.txt"), InstallFailure::Other),
            (
                "Failed to download distribution due to network timeout. Try increasing UV_HTTP_TIMEOUT",
                InstallFailure::Network,
            ),
            ("error: Connection refused via mirror.internal", InstallFailure::Network),
            ("No solution for timeout-helper==1.0", InstallFailure::Other),
        ] {
            assert_eq!(InstallFailure::from_stderr(stderr), expected, "{stderr}");
        }
    }

    #[tokio::test]
    async fn kills_a_hung_uv() {
        let app = tauri::test::mock_builder()
            .plugin(tauri_plugin_shell::init())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        let failure = run_uv(
            app.shell().command("/bin/sh").args(["-c", "exec sleep 30"]),
            Duration::from_millis(100),
        )
        .await
        .unwrap_err();
        assert_eq!(failure, InstallFailure::TimedOut);
    }

    #[test]
    fn kills_a_hung_blocking_command() {
        let mut command = std::process::Command::new("/bin/sh");
        command.args(["-c", "exec sleep 30"]);
        let started = Instant::now();
        assert!(wait_bounded(&mut command, Duration::from_millis(100)).is_err());
        assert!(started.elapsed() < Duration::from_secs(5));
    }

    #[test]
    fn installs_when_runtime_is_missing() {
        assert_eq!(choose_runtime_action(false, false), RuntimeAction::Install);
    }

    #[test]
    fn upgrades_when_installed_runtime_is_old() {
        assert_eq!(choose_runtime_action(true, true), RuntimeAction::Upgrade);
    }

    #[test]
    fn reuses_matching_or_newer_runtime() {
        assert_eq!(choose_runtime_action(true, false), RuntimeAction::Reuse);
    }

    #[test]
    fn upgrades_an_older_cli() {
        assert!(should_upgrade(b"ormah 0.13.6\n", "0.14.0"));
    }

    #[test]
    fn leaves_the_pinned_cli_unchanged() {
        assert!(!should_upgrade(b"ormah 0.14.0\n", "0.14.0"));
    }

    #[test]
    fn never_downgrades_a_newer_cli() {
        assert!(!should_upgrade(b"ormah 0.15.0\n", "0.14.0"));
    }

    #[test]
    fn repairs_unparseable_version_output() {
        assert!(should_upgrade(b"unknown\n", "0.14.0"));
    }

    #[test]
    fn retries_only_transient_install_failures() {
        assert!(transient_install_error(
            "Temporary failure in name resolution while fetching pypi.org"
        ));
        assert!(transient_install_error("operation timed out"));
        assert!(!transient_install_error(
            "No solution found when resolving dependencies"
        ));
        assert!(!transient_install_error("permission denied"));
    }
}
