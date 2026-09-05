//! Tauri commands + small helpers shared by the tray and desktop webview.

use serde_json::Value;
use tauri::{AppHandle, Manager, Runtime};
use tauri_plugin_notification::NotificationExt;

use crate::sidecar::find_ormah;
use crate::stats::{self, Stats};

/// Base URL of the bundled server. Honors `ORMAH_PORT`, defaults to 8787.
pub fn base_url() -> String {
    let port = std::env::var("ORMAH_PORT").unwrap_or_else(|_| "8787".to_string());
    format!("http://127.0.0.1:{port}")
}

// ---- graph -----------------------------------------------------------------

/// The URL the main window navigates to once the server is up — the existing
/// web UI, served by the local server. No reimplementation.
#[tauri::command]
pub fn graph_url() -> String {
    base_url()
}

/// Bring the main window forward (the graph lives inside it).
pub fn open_graph<R: Runtime>(app: &AppHandle<R>) {
    if let Some(win) = app.get_webview_window("main") {
        let _ = win.show();
        let _ = win.unminimize();
        let _ = win.set_focus();
    }
}

#[tauri::command]
pub fn open_graph_cmd<R: Runtime>(app: AppHandle<R>) {
    open_graph(&app);
}

// ---- stats -----------------------------------------------------------------

#[tauri::command]
pub async fn fetch_stats() -> Result<Stats, String> {
    stats::fetch().await.map_err(|e| e.to_string())
}

/// Which supported AI tools are installed (for the detection list).
#[tauri::command]
pub async fn detect_agents() -> Result<Value, String> {
    let url = format!("{}/agent/clients", base_url());
    reqwest::Client::new()
        .get(url)
        .timeout(std::time::Duration::from_secs(5))
        .send()
        .await
        .map_err(|e| e.to_string())?
        .json::<Value>()
        .await
        .map_err(|e| e.to_string())
}

// ---- agent setup -----------------------------------------------------------

/// Run `ormah setup --json` and return its parsed result.
/// Uses find_ormah() so it works even when ~/.local/bin is not on PATH.
#[tauri::command]
pub async fn setup_agents<R: Runtime>(_app: AppHandle<R>) -> Result<Value, String> {
    let bin = find_ormah().ok_or_else(|| "ormah not found — install may have failed".to_string())?;
    let out = std::process::Command::new(bin)
        .args(["setup", "--json"])
        .output()
        .map_err(|e| format!("ormah setup failed: {e}"))?;
    let stdout = String::from_utf8_lossy(&out.stdout);
    serde_json::from_str::<Value>(&stdout).map_err(|e| {
        format!(
            "could not parse setup output: {e}\nstderr: {}",
            String::from_utf8_lossy(&out.stderr)
        )
    })
}

/// Tray-triggered variant: run setup off-thread and surface a notification.
pub fn run_setup_notify<R: Runtime>(app: AppHandle<R>) {
    tauri::async_runtime::spawn(async move {
        let (title, body) = match setup_agents(app.clone()).await {
            Ok(v) => {
                let wired = v
                    .get("wired")
                    .and_then(|w| w.as_array())
                    .map(|a| {
                        a.iter()
                            .filter_map(|x| x.as_str())
                            .collect::<Vec<_>>()
                            .join(", ")
                    })
                    .unwrap_or_default();
                if wired.is_empty() {
                    (
                        "No agents found",
                        "Install Claude Code, Claude Desktop, or Codex, then try again."
                            .to_string(),
                    )
                } else {
                    ("Agents wired up", format!("Connected: {wired}"))
                }
            }
            Err(e) => ("Setup failed", e),
        };
        let _ = app
            .notification()
            .builder()
            .title(title)
            .body(body)
            .show();
    });
}

// ---- server control --------------------------------------------------------

/// Start the ormah daemon. No-op if already running.
#[tauri::command]
pub async fn start_server() -> Result<(), String> {
    crate::sidecar::start_daemon();
    Ok(())
}

/// Stop the ormah daemon.
#[tauri::command]
pub async fn stop_server() -> Result<(), String> {
    crate::sidecar::stop_daemon();
    Ok(())
}

/// Returns true if the ormah server is currently reachable.
#[tauri::command]
pub async fn server_status() -> bool {
    crate::sidecar::is_running().await
}

/// Retry a failed bundled-runtime setup without starting concurrent installers.
#[tauri::command]
pub fn retry_runtime_setup<R: Runtime>(app: AppHandle<R>) -> bool {
    crate::sidecar::start(app)
}

#[cfg(test)]
mod tests {
    #[test]
    fn bootstrap_commands_are_allowed_by_the_desktop_capability() {
        let permissions = include_str!("../permissions/desktop.toml");
        let bootstrap = permissions
            .split_once("identifier = \"desktop-bootstrap\"")
            .expect("desktop bootstrap permission set is missing")
            .1
            .split_once("[[permission]]")
            .map(|(bootstrap, _)| bootstrap)
            .unwrap_or(permissions);

        for command in ["fetch_stats", "graph_url", "retry_runtime_setup"] {
            assert!(
                bootstrap.contains(&format!("\"{command}\"")),
                "bootstrap command `{command}` is not allowed by the desktop capability"
            );
        }
    }
}
