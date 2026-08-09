//! Signed desktop updates with explicit user-controlled installation.

use std::sync::atomic::{AtomicBool, Ordering};
use std::sync::{Mutex, OnceLock};

use tauri::{AppHandle, Emitter, Runtime, WebviewWindow};
use tauri_plugin_updater::{Update, UpdaterExt};

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
#[serde(rename_all = "snake_case")]
pub enum UpdatePhase {
    Checking,
    Current,
    Available,
    Downloading,
    Installing,
    Failed,
}

#[derive(Clone, Debug, serde::Serialize, serde::Deserialize, PartialEq, Eq)]
pub struct UpdateStatus {
    pub phase: UpdatePhase,
    pub version: Option<String>,
    pub notes: String,
    pub progress_percent: Option<u8>,
    pub message: Option<String>,
}

impl UpdateStatus {
    fn checking() -> Self {
        Self {
            phase: UpdatePhase::Checking,
            version: None,
            notes: String::new(),
            progress_percent: None,
            message: None,
        }
    }

    fn current() -> Self {
        Self {
            phase: UpdatePhase::Current,
            ..Self::checking()
        }
    }

    fn available(version: String, notes: String) -> Self {
        Self {
            phase: UpdatePhase::Available,
            version: Some(version),
            notes,
            progress_percent: None,
            message: None,
        }
    }

    fn failed(message: &str) -> Self {
        Self {
            phase: UpdatePhase::Failed,
            version: None,
            notes: String::new(),
            progress_percent: None,
            message: Some(message.to_string()),
        }
    }
}

/// Kept for the existing tray event contract.
#[derive(Clone, serde::Serialize, serde::Deserialize)]
pub struct UpdateAvailable {
    pub version: String,
    pub notes: String,
}

static STATUS: OnceLock<Mutex<UpdateStatus>> = OnceLock::new();
static INSTALL_IN_PROGRESS: AtomicBool = AtomicBool::new(false);

struct InstallGuard;

impl InstallGuard {
    fn acquire() -> Option<Self> {
        INSTALL_IN_PROGRESS
            .compare_exchange(false, true, Ordering::AcqRel, Ordering::Acquire)
            .ok()
            .map(|_| Self)
    }
}

impl Drop for InstallGuard {
    fn drop(&mut self) {
        INSTALL_IN_PROGRESS.store(false, Ordering::Release);
    }
}

fn status_store() -> &'static Mutex<UpdateStatus> {
    STATUS.get_or_init(|| Mutex::new(UpdateStatus::checking()))
}

fn current_status() -> UpdateStatus {
    status_store()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner())
        .clone()
}

fn set_status(status: UpdateStatus) {
    *status_store()
        .lock()
        .unwrap_or_else(|poisoned| poisoned.into_inner()) = status;
}

fn concise_notes(body: Option<&str>) -> String {
    let notes = body
        .unwrap_or_default()
        .lines()
        .map(str::trim)
        .filter(|line| !line.is_empty())
        .take(3)
        .collect::<Vec<_>>()
        .join(" ");
    notes.chars().take(320).collect()
}

/// Check once at startup. This only discovers updates; it never installs one.
pub fn check<R: Runtime>(app: AppHandle<R>) {
    tauri::async_runtime::spawn(async move {
        if let Err(error) = check_now(&app, true).await {
            eprintln!("updater check failed: {error}");
            // Startup checks are best-effort. Being offline is not an app
            // failure and should not interrupt the user with a warning.
            set_status(UpdateStatus::current());
        }
    });
}

async fn available_update<R: Runtime>(
    app: &AppHandle<R>,
) -> tauri_plugin_updater::Result<Option<Update>> {
    app.updater()?.check().await
}

async fn check_now<R: Runtime>(
    app: &AppHandle<R>,
    notify: bool,
) -> tauri_plugin_updater::Result<UpdateStatus> {
    set_status(UpdateStatus::checking());
    let Some(update) = available_update(app).await? else {
        let status = UpdateStatus::current();
        set_status(status.clone());
        return Ok(status);
    };

    let status = UpdateStatus::available(
        update.version.clone(),
        concise_notes(update.body.as_deref()),
    );
    set_status(status.clone());
    let _ = app.emit(
        "ormah://update-available",
        UpdateAvailable {
            version: update.version.clone(),
            notes: status.notes.clone(),
        },
    );

    if notify {
        use tauri_plugin_notification::NotificationExt;
        let body = format!(
            "Ormah Desktop {} is ready. Open Ormah to review and install it.",
            update.version
        );
        let _ = app
            .notification()
            .builder()
            .title("Update available")
            .body(body)
            .show();
    }
    Ok(status)
}

#[tauri::command]
pub fn desktop_update_status(window: WebviewWindow) -> Result<UpdateStatus, String> {
    crate::product_bridge::require_product_origin(&window)?;
    Ok(current_status())
}

#[tauri::command]
pub async fn check_desktop_update<R: Runtime>(
    app: AppHandle<R>,
    window: WebviewWindow<R>,
) -> Result<UpdateStatus, String> {
    crate::product_bridge::require_product_origin(&window)?;
    check_now(&app, false).await.map_err(|error| {
        eprintln!("manual updater check failed: {error}");
        let status = UpdateStatus::failed(
            "Ormah could not check for updates. Check your connection and try again.",
        );
        set_status(status);
        "Ormah could not check for updates. Check your connection and try again.".to_string()
    })
}

#[tauri::command]
pub async fn install_desktop_update<R: Runtime>(
    app: AppHandle<R>,
    window: WebviewWindow<R>,
) -> Result<UpdateStatus, String> {
    crate::product_bridge::require_product_origin(&window)?;
    let _guard = InstallGuard::acquire()
        .ok_or_else(|| "An Ormah update is already in progress.".to_string())?;
    install_now(app).await.map_err(|error| {
        eprintln!("update install failed: {error}");
        let message = "The update could not be installed. Your current Ormah version is unchanged.";
        set_status(UpdateStatus::failed(message));
        message.to_string()
    })?;
    Ok(current_status())
}

/// Tray entry point. Installation still happens only after an explicit click.
pub fn install<R: Runtime>(app: AppHandle<R>) {
    let Some(guard) = InstallGuard::acquire() else {
        return;
    };
    tauri::async_runtime::spawn(async move {
        let _guard = guard;
        if let Err(error) = install_now(app).await {
            eprintln!("update install failed: {error}");
            set_status(UpdateStatus::failed(
                "The update could not be installed. Your current Ormah version is unchanged.",
            ));
        }
    });
}

async fn install_now<R: Runtime>(app: AppHandle<R>) -> tauri_plugin_updater::Result<()> {
    let Some(update) = available_update(&app).await? else {
        set_status(UpdateStatus::current());
        return Ok(());
    };

    let version = update.version.clone();
    let notes = concise_notes(update.body.as_deref());
    let mut downloaded = 0_u64;
    set_status(UpdateStatus {
        phase: UpdatePhase::Downloading,
        version: Some(version.clone()),
        notes: notes.clone(),
        progress_percent: Some(0),
        message: None,
    });
    update
        .download_and_install(
            move |chunk, total| {
                downloaded = downloaded.saturating_add(chunk as u64);
                let progress = total
                    .filter(|total| *total > 0)
                    .map(|total| ((downloaded.saturating_mul(100) / total).min(100)) as u8);
                set_status(UpdateStatus {
                    phase: UpdatePhase::Downloading,
                    version: Some(version.clone()),
                    notes: notes.clone(),
                    progress_percent: progress,
                    message: None,
                });
            },
            || {
                let previous = current_status();
                set_status(UpdateStatus {
                    phase: UpdatePhase::Installing,
                    progress_percent: Some(100),
                    ..previous
                });
            },
        )
        .await?;

    // download_and_install verifies the configured signature before it returns.
    // Restart only after the signed artifact has been installed successfully.
    app.restart();
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn release_notes_are_short_plain_text() {
        let input = "\n First improvement \n\nSecond improvement\nThird improvement\nIgnored";
        assert_eq!(
            concise_notes(Some(input)),
            "First improvement Second improvement Third improvement"
        );
        assert!(concise_notes(Some(&"x".repeat(400))).chars().count() <= 320);
    }

    #[test]
    fn update_state_never_implies_automatic_installation() {
        let available = UpdateStatus::available("0.4.0".into(), "Safer restore".into());
        assert_eq!(available.phase, UpdatePhase::Available);
        assert_eq!(available.progress_percent, None);
        assert_eq!(UpdateStatus::current().phase, UpdatePhase::Current);
    }
}
