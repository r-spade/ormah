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

use crate::diagnostics::{self, OutputRedactor};
use std::path::PathBuf;
use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::Mutex;
use std::time::{Duration, Instant};
use tauri::{AppHandle, Emitter, Runtime};
use tauri_plugin_shell::process::{Command, CommandEvent};
use tauri_plugin_shell::ShellExt;

// Injected by build.rs from the repo's pyproject.toml, so the pinned Python
// package version can never drift from the released one.
const ORMAH_VERSION: &str = env!("ORMAH_PY_VERSION");
const INSTALL_ARGS: &[&str] = &[
    "tool",
    "install",
    "--reinstall",
    "--force",
    "--python",
    "3.11",
];
const RETRY_DELAYS: &[u64] = &[10, 20, 30, 45];
const INSTALL_ATTEMPTS: u8 = RETRY_DELAYS.len() as u8 + 1;
const INSTALL_TIMEOUT: Duration = Duration::from_secs(600);
static LATEST_STATUS: Mutex<Option<SetupStatus>> = Mutex::new(None);
static SETUP_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

/// Phase emitted on the "ormah://status" event so the UI can show progress.
#[derive(Clone, serde::Serialize)]
#[serde(tag = "phase", rename_all = "snake_case")]
pub enum Phase {
    Installing {
        version: &'static str,
        attempt: u8,
        attempts: u8,
    },
    Retrying {
        version: &'static str,
        attempt: u8,
        attempts: u8,
        delay_seconds: u64,
    },
    Starting,
    Failed {
        reason: String,
        can_retry: bool,
    },
}

#[derive(Clone, serde::Serialize)]
pub struct SetupStatus {
    pub sequence: u64,
    pub updated_at_ms: u64,
    #[serde(flatten)]
    pub phase: Phase,
}

pub fn latest_status() -> Option<SetupStatus> {
    LATEST_STATUS.lock().ok()?.clone()
}

fn publish<R: Runtime>(app: &AppHandle<R>, phase: Phase) {
    update_status(app, phase, false);
}

fn update_status<R: Runtime>(app: &AppHandle<R>, phase: Phase, completed: bool) {
    if let Ok(mut latest) = LATEST_STATUS.lock() {
        if completed {
            SETUP_IN_PROGRESS.store(false, Ordering::Release);
        }
        let sequence = latest.as_ref().map_or(1, |s| s.sequence + 1);
        let status = SetupStatus {
            sequence,
            updated_at_ms: diagnostics::now_ms(),
            phase,
        };
        diagnostics::log(&format!(
            "status {}",
            serde_json::to_string(&status).unwrap_or_default()
        ));
        *latest = Some(status.clone());
        let _ = app.emit("ormah://status", status);
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
    // Publish synchronously so an accepted retry cannot leave the previous Failed snapshot.
    publish(
        &app,
        Phase::Installing {
            version: ORMAH_VERSION,
            attempt: 1,
            attempts: INSTALL_ATTEMPTS,
        },
    );
    tauri::async_runtime::spawn(async move {
        let result = ensure_running(&app).await;
        // Release before Failed is visible so a quick click really can retry.
        if let Err(reason) = result {
            update_status(
                &app,
                Phase::Failed {
                    reason,
                    can_retry: true,
                },
                true,
            );
        } else {
            SETUP_IN_PROGRESS.store(false, Ordering::Release);
        }
    });
    true
}

async fn ensure_running<R: Runtime>(app: &AppHandle<R>) -> Result<(), String> {
    log_platform();
    match app.shell().sidecar("uv") {
        Ok(command) => {
            match run_uv_command(command.arg("--version"), Duration::from_secs(10)).await {
                Ok(out) => diagnostics::log(&format!("bundled uv version: {}", out.trim())),
                Err(e) => {
                    diagnostics::log(&format!("bundled uv version unavailable: {}", e.details))
                }
            }
        }
        Err(e) => diagnostics::log(&format!("bundled uv unavailable: {e}")),
    }
    let installed = find_ormah().is_some();
    let runtime_action = choose_runtime_action(installed, installed && needs_upgrade());
    diagnostics::log(&format!("runtime action={runtime_action:?}"));
    let runtime_changed = match runtime_action {
        RuntimeAction::Install | RuntimeAction::Upgrade => {
            install_with_retry(app)
                .await
                .map_err(|failure| failure.public_reason())?;
            true
        }
        RuntimeAction::Reuse => false,
    };

    // Reinstall replaces disk files; restart the daemon to load the new package.
    if runtime_changed {
        let _ = tauri::async_runtime::spawn_blocking(stop_daemon).await;
    }
    publish(app, Phase::Starting);
    start_daemon().await
}

fn log_platform() {
    let os_version = std::process::Command::new("uname")
        .args(["-srv"])
        .output()
        .map(|o| String::from_utf8_lossy(&o.stdout).trim().to_string())
        .unwrap_or_default();
    let mut names: Vec<_> = std::env::vars_os()
        .filter_map(|(key, _)| {
            let key = key.to_string_lossy().into_owned();
            let lower = key.to_ascii_lowercase();
            (key.starts_with("UV_")
                || ["http_proxy", "https_proxy", "all_proxy", "no_proxy"].contains(&lower.as_str()))
            .then_some(key)
        })
        .collect();
    names.sort();
    diagnostics::log(&format!(
        "app={} pinned_ormah={} os={} version={} arch={} environment_names={names:?}",
        env!("CARGO_PKG_VERSION"),
        ORMAH_VERSION,
        std::env::consts::OS,
        os_version,
        std::env::consts::ARCH
    ));
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
    let result = tauri::async_runtime::spawn_blocking(|| {
        let bin = find_ormah().ok_or_else(|| "Ormah runtime was not found.".to_string())?;
        diagnostics::log(&format!("daemon command={bin:?} args=[server, start, -d]"));
        let started = Instant::now();
        let output = clean_python_env(std::process::Command::new(&bin))
            .args(["server", "start", "-d"])
            .output()
            .map_err(|e| e.to_string())?;
        diagnostics::log(&format!(
            "daemon duration_ms={} status={} stdout:\n{}\nstderr:\n{}",
            started.elapsed().as_millis(),
            output.status,
            String::from_utf8_lossy(&output.stdout),
            String::from_utf8_lossy(&output.stderr)
        ));
        if output.status.success() {
            Ok(())
        } else {
            Err(format!("daemon exited {}", output.status))
        }
    })
    .await
    .map_err(|e| e.to_string())
    .and_then(|r| r);
    if let Err(error) = result {
        diagnostics::log(&format!("daemon start error: {error}"));
        if is_running().await {
            diagnostics::log("daemon command failed but health check confirms a server is running");
            return Ok(());
        }
        return Err(
            "Ormah could not start its server. Try again, or copy the details for help.".into(),
        );
    }
    Ok(())
}

/// Stop the ormah daemon. No-op if not running.
pub fn stop_daemon() {
    let Some(bin) = find_ormah() else { return };
    let _ = clean_python_env(std::process::Command::new(&bin))
        .args(["server", "stop"])
        .status();
}

/// Ping /admin/health — true means the server is up and responding.
pub async fn is_running() -> bool {
    let url = format!("{}/admin/health", crate::commands::base_url());
    reqwest::Client::new()
        .get(&url)
        .timeout(std::time::Duration::from_secs(2))
        .send()
        .await
        .map(|r| r.status().is_success())
        .unwrap_or(false)
}

fn needs_upgrade() -> bool {
    let Some(bin) = find_ormah() else {
        return false;
    };
    let Ok(out) = clean_python_env(std::process::Command::new(bin))
        .arg("--version")
        .output()
    else {
        return true;
    };
    if !out.status.success() {
        return true;
    }
    if should_upgrade(&out.stdout, ORMAH_VERSION) {
        return true;
    }
    let installed = String::from_utf8_lossy(&out.stdout)
        .split_whitespace()
        .find_map(|word| semver::Version::parse(word).ok());
    if installed != semver::Version::parse(ORMAH_VERSION).ok() {
        return false;
    }
    let Some(python) = find_ormah()
        .and_then(|bin| bin.canonicalize().ok())
        .and_then(|bin| bin.parent().map(|dir| dir.join("python")))
        .filter(|p| p.exists())
    else {
        return false;
    };
    let output = clean_python_env(std::process::Command::new(python))
        .args([
            "-c",
            "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')",
        ])
        .output();
    match output {
        Ok(out) => interpreter_needs_rebuild(out.status.success(), &out.stdout),
        Err(_) => true,
    }
}

fn interpreter_needs_rebuild(success: bool, output: &[u8]) -> bool {
    !success || String::from_utf8_lossy(output).trim() != "3.11"
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
enum FailureKind {
    PackageService,
    PythonDownload,
    TimedOut,
    Other,
}

#[derive(Debug)]
struct InstallFailure {
    details: String,
    transient: bool,
    kind: FailureKind,
}

impl InstallFailure {
    fn from_details(details: String) -> Self {
        let transient = transient_install_error(&details);
        let lower = details.to_ascii_lowercase();
        let kind = if lower.contains("failed to download")
            && (lower.contains("cpython-") || lower.contains("python-build-standalone"))
        {
            FailureKind::PythonDownload
        } else if transient
            && (lower.contains("pypi.org/") || lower.contains("files.pythonhosted.org/"))
        {
            FailureKind::PackageService
        } else {
            FailureKind::Other
        };
        Self {
            details,
            transient,
            kind,
        }
    }
    fn public_reason(&self) -> String {
        match self.kind {
            FailureKind::PythonDownload => {
                "Ormah could not download Python 3.11. Check your connection, then try again."
            }
            FailureKind::PackageService => {
                "Ormah could not reach the package service. Check your connection, then try again."
            }
            FailureKind::TimedOut => {
                "Installing Ormah took too long. Try again, or copy the details for help."
            }
            FailureKind::Other => {
                "Ormah could not install its runtime. Try again, or copy the details for help."
            }
        }
        .into()
    }
}

fn transient_install_error(details: &str) -> bool {
    let details = details.to_ascii_lowercase();
    [
        "dns error",
        "failed to lookup address",
        "temporary failure in name resolution",
        "timed out",
        "connection reset",
        "connection refused",
        "network is unreachable",
    ]
    .iter()
    .any(|pattern| details.contains(pattern))
}

fn install_args(version: &str) -> Vec<String> {
    INSTALL_ARGS
        .iter()
        .map(|arg| arg.to_string())
        .chain([format!("ormah=={version}")])
        .collect()
}

fn retry_delay(attempt: u8) -> Option<u64> {
    attempt
        .checked_sub(1)
        .and_then(|i| RETRY_DELAYS.get(i as usize).copied())
}

async fn install_with_retry<R: Runtime>(app: &AppHandle<R>) -> Result<(), InstallFailure> {
    for attempt in 1..=INSTALL_ATTEMPTS {
        publish(
            app,
            Phase::Installing {
                version: ORMAH_VERSION,
                attempt,
                attempts: INSTALL_ATTEMPTS,
            },
        );
        let started = Instant::now();
        let result = install_via_uv(app).await;
        diagnostics::log(&format!(
            "uv attempt={attempt}/{INSTALL_ATTEMPTS} duration_ms={} result={:?}",
            started.elapsed().as_millis(),
            result
                .as_ref()
                .map(|_| "success")
                .map_err(|e| (&e.kind, e.transient))
        ));
        match result {
            Ok(()) => return Ok(()),
            Err(failure) if failure.transient && retry_delay(attempt).is_some() => {
                let delay_seconds = retry_delay(attempt).unwrap();
                publish(
                    app,
                    Phase::Retrying {
                        version: ORMAH_VERSION,
                        attempt: attempt + 1,
                        attempts: INSTALL_ATTEMPTS,
                        delay_seconds,
                    },
                );
                tokio::time::sleep(Duration::from_secs(delay_seconds)).await;
            }
            Err(failure) => return Err(failure),
        }
    }
    unreachable!("the bounded install loop always returns")
}

async fn install_via_uv<R: Runtime>(app: &AppHandle<R>) -> Result<(), InstallFailure> {
    let args = install_args(ORMAH_VERSION);
    diagnostics::log(&format!(
        "uv arguments={args:?} timeout_seconds={}",
        INSTALL_TIMEOUT.as_secs()
    ));
    let uv = app.shell().sidecar("uv").map_err(|e| {
        diagnostics::log(&format!("uv sidecar not available ({e})"));
        InstallFailure::from_details(format!("uv sidecar not available ({e})"))
    })?;
    run_uv_command(uv.args(args), INSTALL_TIMEOUT)
        .await
        .map(|_| ())
}

async fn run_uv_command(command: Command, timeout: Duration) -> Result<String, InstallFailure> {
    let (mut rx, child) = command.set_raw_out(true).spawn().map_err(|e| {
        diagnostics::log(&format!("uv spawn failed: {e}"));
        InstallFailure::from_details(e.to_string())
    })?;
    diagnostics::log(&format!("uv spawned pid={}", child.pid()));
    let mut stderr = String::new();
    let mut stdout = String::new();
    let mut stderr_redactor = OutputRedactor::default();
    let mut stdout_redactor = OutputRedactor::default();
    let result = tokio::time::timeout(timeout, async {
        while let Some(event) = rx.recv().await {
            match event {
                CommandEvent::Stderr(bytes) => {
                    stderr.push_str(&String::from_utf8_lossy(&bytes));
                    let safe = stderr_redactor.push(&bytes);
                    if !safe.is_empty() {
                        diagnostics::log(&format!("uv stderr: {safe}"));
                    }
                }
                CommandEvent::Stdout(bytes) => {
                    stdout.push_str(&String::from_utf8_lossy(&bytes));
                    let safe = stdout_redactor.push(&bytes);
                    if !safe.is_empty() {
                        diagnostics::log(&format!("uv stdout: {safe}"));
                    }
                }
                CommandEvent::Terminated(status) => {
                    diagnostics::log(&format!(
                        "uv exit_code={:?} signal={:?}",
                        status.code, status.signal
                    ));
                    return status.code == Some(0);
                }
                CommandEvent::Error(error) => {
                    stderr.push_str(&error);
                    diagnostics::log(&format!("uv event error: {error}"));
                }
                _ => {}
            }
            // Logs retain the complete stream subject to rotation; bound in-memory diagnostics.
            for output in [&mut stderr, &mut stdout] {
                if output.len() > 1024 * 1024 {
                    let mut cut = output.len() - 1024 * 1024;
                    while !output.is_char_boundary(cut) {
                        cut += 1;
                    }
                    output.drain(..cut);
                }
            }
        }
        false
    })
    .await;
    if result.is_err() {
        diagnostics::log(&format!(
            "uv hard timeout after {} seconds; kill={:?}",
            timeout.as_secs(),
            child.kill()
        ));
    }
    for (name, safe) in [
        ("stderr", stderr_redactor.finish()),
        ("stdout", stdout_redactor.finish()),
    ] {
        if !safe.is_empty() {
            diagnostics::log(&format!("uv {name}: {safe}"));
        }
    }
    match result {
        Ok(true) => Ok(stdout),
        Ok(false) => Err(InstallFailure::from_details(stderr)),
        Err(_) => Err(InstallFailure {
            details: format!(
                "uv hard timeout after {} seconds\n{stderr}",
                timeout.as_secs()
            ),
            transient: true,
            kind: FailureKind::TimedOut,
        }),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn rebuilds_a_matching_package_on_another_or_broken_python() {
        assert!(interpreter_needs_rebuild(true, b"3.12\n"));
        assert!(interpreter_needs_rebuild(false, b""));
        assert!(!interpreter_needs_rebuild(true, b"3.11\n"));
    }

    #[test]
    fn pinned_install_arguments_and_retry_schedule() {
        assert_eq!(
            install_args("1.2.3"),
            [
                "tool",
                "install",
                "--reinstall",
                "--force",
                "--python",
                "3.11",
                "ormah==1.2.3"
            ]
        );
        assert_eq!(INSTALL_ATTEMPTS, 5);
        assert_eq!((1..=5).filter_map(retry_delay).sum::<u64>(), 105);
        assert_eq!(retry_delay(0), None);
        assert_eq!(retry_delay(5), None);
    }

    #[test]
    fn classifies_real_uv_failures() {
        let pypi = InstallFailure::from_details(include_str!("fixtures/uv-pypi-down.txt").into());
        assert!(pypi.transient);
        assert_eq!(pypi.kind, FailureKind::PackageService);
        let python =
            InstallFailure::from_details(include_str!("fixtures/uv-python-down.txt").into());
        assert!(python.transient);
        assert_eq!(python.kind, FailureKind::PythonDownload);
        let resolution =
            InstallFailure::from_details(include_str!("fixtures/uv-no-solution.txt").into());
        assert!(!resolution.transient);
        assert_eq!(resolution.kind, FailureKind::Other);
        assert!(!transient_install_error(
            "No solution for timeout-helper==1.0"
        ));
    }

    #[tokio::test]
    async fn kills_a_hung_child_and_keeps_partial_stderr() {
        let app = tauri::test::mock_builder()
            .plugin(tauri_plugin_shell::init())
            .build(tauri::test::mock_context(tauri::test::noop_assets()))
            .unwrap();
        let failure = run_uv_command(
            app.shell()
                .command("/bin/sh")
                .args(["-c", "printf 'download pending' >&2; exec sleep 30"]),
            Duration::from_millis(100),
        )
        .await
        .unwrap_err();
        assert!(failure.transient);
        assert_eq!(failure.kind, FailureKind::TimedOut);
        assert!(failure.details.contains("download pending"));
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
