//! Ormah menubar app (Tauri v2).
//!
//! Ruthlessly minimal per F10: a tray icon whose title is the weekly
//! whispers-used count (the felt-value counter, F09), a dropdown with the
//! stats + actions, a bundled server sidecar, and one-click agent setup.
//! No global hotkey, no native graph, no settings beyond start-at-login.

mod commands;
mod product_bridge;
mod sidecar;
mod stats;
mod tray;
mod updater;

use tauri::{WebviewUrl, WebviewWindowBuilder};
#[cfg(target_os = "linux")]
use tauri::Manager;
use tauri_plugin_autostart::{MacosLauncher, ManagerExt};

// Hide scrollbar chrome across every page the webview loads (the install shell
// *and* the graph it navigates to). Scrolling still works; only the visual bar
// is removed — kills the spurious horizontal scrollbar on the graph view.
const HIDE_SCROLLBARS: &str = r#"
(function () {
  var css = '::-webkit-scrollbar{width:0!important;height:0!important;background:transparent!important}'
          + 'html{scrollbar-width:none!important;-ms-overflow-style:none!important}';
  var apply = function () {
    if (document.getElementById('__ormah_noscroll')) return;
    var s = document.createElement('style');
    s.id = '__ormah_noscroll';
    s.appendChild(document.createTextNode(css));
    (document.head || document.documentElement).appendChild(s);
  };
  apply();
  document.addEventListener('DOMContentLoaded', apply);
})();
"#;

// Marks the webview as running inside the desktop app on every page load —
// including after navigating to http://localhost:8787 where window.__TAURI__
// is not injected (remote URL). The UI checks this to apply in-app styles.
const MARK_IN_APP: &str = "window.__ORMAH_DESKTOP__ = true;";

/// Linux single-instance guard.
///
/// macOS uses `tauri-plugin-single-instance` (a Unix socket). On Linux that
/// plugin instead claims a D-Bus well-known name, which fails silently in the
/// AppImage/deb runtime and lets duplicate instances — and duplicate tray
/// icons — through. Here we reserve a per-user *abstract* Unix socket: the
/// kernel guarantees a single owner and auto-releases it when the process
/// dies, so there are no stale lock files.
///
/// Returns `Ok(Some(listener))` when we are the first instance (the caller
/// must keep the listener alive for the process lifetime), `Ok(None)` for a
/// duplicate that should exit, and `Err` when the mechanism is unavailable —
/// in which case the caller boots anyway, since failing open must never block
/// the app from starting.
#[cfg(target_os = "linux")]
fn single_instance_listener() -> std::io::Result<Option<std::os::unix::net::UnixListener>> {
    use std::io::Write;
    use std::os::linux::net::SocketAddrExt;
    use std::os::unix::net::{SocketAddr, UnixListener, UnixStream};

    // XDG_RUNTIME_DIR is already per-user (/run/user/<uid>), so embedding it
    // scopes the lock to this user without having to look up the uid.
    let scope = std::env::var("XDG_RUNTIME_DIR").unwrap_or_else(|_| "shared".into());
    let name = format!("ormah-desktop.{}", scope.replace('/', "_"));
    let addr = SocketAddr::from_abstract_name(name.as_bytes())?;
    match UnixListener::bind_addr(&addr) {
        Ok(listener) => Ok(Some(listener)),
        Err(e) if e.kind() == std::io::ErrorKind::AddrInUse => {
            // Signal the first instance to show and focus its window, then
            // let the caller exit this duplicate process.
            if let Ok(mut stream) = UnixStream::connect_addr(&addr) {
                let _ = stream.write_all(b"f");
            }
            Ok(None)
        }
        Err(e) => Err(e),
    }
}

/// Desktop-menu integration for AppImage runs.
///
/// Raw AppImages don't register with the app menu, so the app is invisible to
/// search/dock until the user wires a .desktop entry by hand. The `.deb` ships
/// a system-wide entry via dpkg, and dev builds shouldn't pollute the menu, so
/// this only runs when the AppImage runtime is detected (it sets $APPIMAGE to
/// the image's path). Idempotent: rewritten only when the AppImage path moves.
/// Best-effort by design — a failure here must never block startup.
#[cfg(target_os = "linux")]
fn integrate_appimage() {
    let Ok(appimage) = std::env::var("APPIMAGE") else { return };
    let Some(home) = std::env::var_os("HOME") else { return };
    let share = std::path::Path::new(&home).join(".local/share");

    // Icon name matches the .deb packaging (Icon=ormah-desktop) so both
    // install paths resolve the same themed icon.
    let icons: [(&str, &[u8]); 2] = [
        ("128x128", include_bytes!("../icons/128x128.png")),
        ("256x256", include_bytes!("../icons/128x128@2x.png")),
    ];
    for (size, bytes) in icons {
        let dir = share.join(format!("icons/hicolor/{size}/apps"));
        let path = dir.join("ormah-desktop.png");
        if !path.exists() {
            let _ = std::fs::create_dir_all(&dir);
            let _ = std::fs::write(&path, bytes);
        }
    }

    let entry = format!(
        "[Desktop Entry]\n\
         Type=Application\n\
         Name=Ormah\n\
         Comment=Local-first memory for your AI tools\n\
         Exec=\"{appimage}\"\n\
         Icon=ormah-desktop\n\
         Terminal=false\n\
         Categories=Office;\n\
         StartupWMClass=ormah-desktop\n"
    );
    let apps_dir = share.join("applications");
    let entry_path = apps_dir.join("ormah.desktop");
    if std::fs::read_to_string(&entry_path).is_ok_and(|cur| cur == entry) {
        return;
    }
    let _ = std::fs::create_dir_all(&apps_dir);
    if std::fs::write(&entry_path, entry).is_ok() {
        // Refresh the menu cache; desktops that watch the dir pick it up anyway.
        let _ = std::process::Command::new("update-desktop-database")
            .arg(apps_dir)
            .status();
    }
}

pub fn run() {
    #[allow(unused_mut)]
    let mut builder = tauri::Builder::default();

    // macOS/other: the plugin handles single-instance and focuses the existing
    // window on a second launch. Linux uses the abstract-socket guard in
    // `setup` instead (see `single_instance_listener`).
    #[cfg(not(target_os = "linux"))]
    {
        builder = builder.plugin(tauri_plugin_single_instance::init(|app, _args, _cwd| {
            commands::open_graph(app);
        }));
    }

    builder
        .plugin(tauri_plugin_shell::init())
        .plugin(tauri_plugin_dialog::init())
        .plugin(tauri_plugin_notification::init())
        .plugin(tauri_plugin_updater::Builder::new().build())
        .plugin(tauri_plugin_autostart::init(
            MacosLauncher::LaunchAgent,
            None,
        ))
        .invoke_handler(tauri::generate_handler![
            commands::setup_agents,
            commands::fetch_stats,
            commands::detect_agents,
            commands::open_graph_cmd,
            commands::graph_url,
            commands::mark_onboarded,
            commands::is_onboarded,
            commands::start_server,
            commands::stop_server,
            commands::server_status,
            commands::retry_runtime_setup,
            product_bridge::desktop_bridge_info,
            product_bridge::account_status,
            product_bridge::request_account_code,
            product_bridge::verify_account_code,
            product_bridge::logout_account,
            product_bridge::billing_offer,
            product_bridge::protection_status,
            product_bridge::create_protection_intent,
            product_bridge::bind_protection_intent,
            product_bridge::cancel_protection_intent,
            product_bridge::enable_protection,
            product_bridge::disable_protection,
            product_bridge::backup_now,
            product_bridge::verify_now,
            product_bridge::operation_status,
            product_bridge::open_checkout,
            product_bridge::open_billing_portal,
            product_bridge::save_recovery_kit,
            product_bridge::open_printable_recovery_kit,
        ])
        .setup(|app| {
            // Linux: refuse to start a second instance so we never spawn a
            // duplicate tray icon. macOS handles this via the plugin above.
            #[cfg(target_os = "linux")]
            match single_instance_listener() {
                // Another instance already owns the lock — it was signalled to
                // show its window; exit before building anything.
                Ok(None) => std::process::exit(0),
                // First instance: accept focus signals from future duplicate launches.
                // The thread owns the listener (keeping the socket alive), and
                // shows/focuses the main window each time a duplicate connects.
                // By the time any signal arrives the window is already built.
                Ok(Some(listener)) => {
                    let app_handle = app.handle().clone();
                    std::thread::spawn(move || {
                        for _ in listener.incoming() {
                            if let Some(win) = app_handle.get_webview_window("main") {
                                let _ = win.show();
                                let _ = win.set_focus();
                            }
                        }
                    });
                }
                // Mechanism unavailable — boot anyway rather than block startup.
                Err(e) => eprintln!("single-instance guard unavailable, continuing: {e}"),
            }

            // Register with the app menu when running as an AppImage.
            #[cfg(target_os = "linux")]
            integrate_appimage();

            // Start the bundled server as a daemon (survives app closing).
            sidecar::start(app.handle().clone());

            // Always enable autostart — ormah should run at login by default.
            let _ = app.handle().autolaunch().enable();

            // Check for a desktop app update in the background — user is
            // notified and must explicitly click to install.
            updater::check(app.handle().clone());

            // Main window: shows the install/boot flow, then navigates to the
            // graph. Built in Rust so we can attach the scrollbar-hiding init
            // script that also applies after navigating to the graph.
            let window =
                WebviewWindowBuilder::new(app, "main", WebviewUrl::App("index.html".into()))
                    .title("Ormah")
                    .inner_size(1180.0, 820.0)
                    .min_inner_size(860.0, 600.0)
                    .center()
                    .decorations(false) // custom dark title bar drawn in the UI
                    .initialization_script(HIDE_SCROLLBARS)
                    .initialization_script(MARK_IN_APP)
                    .build()?;

            // Hide the window on close instead of destroying it — the tray and
            // daemon keep running. User reopens via the tray icon.
            let win = window.clone();
            window.on_window_event(move |event| {
                if let tauri::WindowEvent::CloseRequested { api, .. } = event {
                    api.prevent_close();
                    let _ = win.hide();
                }
            });

            // Build the tray; it owns the stats poller and server controls.
            tray::build(app)?;

            Ok(())
        })
        .build(tauri::generate_context!())
        .expect("error while building the Ormah desktop app")
        .run(|_app_handle, _event| {
            // macOS fires Reopen when the Dock icon is clicked while the app
            // has no visible windows (e.g. after closing to tray) — without
            // this, only the tray menu's "Open Ormah" item can bring the
            // window back.
            #[cfg(target_os = "macos")]
            if let tauri::RunEvent::Reopen { .. } = _event {
                commands::open_graph(_app_handle);
            }

            // Server is a daemon — it outlives the app process intentionally.
        });
}
