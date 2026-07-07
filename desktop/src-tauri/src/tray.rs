//! The menubar tray: icon + title (weekly counter) + dropdown menu.

use tauri::{
    menu::{MenuBuilder, MenuItemBuilder},
    tray::{MouseButton, MouseButtonState, TrayIconBuilder, TrayIconEvent},
    App, Listener,
};
use crate::{commands, sidecar, stats, updater};

fn server_status_label(running: bool) -> &'static str {
    if running { "Server: running  ●" } else { "Server: stopped  ○" }
}

fn server_toggle_label(running: bool) -> &'static str {
    if running { "Stop server" } else { "Start server" }
}

pub fn build(app: &App) -> tauri::Result<()> {
    let handle = app.handle();

    // Stats lines (disabled = display-only). Updated live by the poller.
    let stats_week = MenuItemBuilder::with_id("stats_week", "…")
        .enabled(false)
        .build(app)?;
    let stats_total = MenuItemBuilder::with_id("stats_total", " ")
        .enabled(false)
        .build(app)?;

    // Server status (display-only) + start/stop action.
    let server_status = MenuItemBuilder::with_id("server_status", server_status_label(true))
        .enabled(false)
        .build(app)?;
    let server_toggle =
        MenuItemBuilder::with_id("server_toggle", server_toggle_label(true)).build(app)?;

    // Shown only when a new version is available; enabled on that event.
    let update_item = MenuItemBuilder::with_id("install_update", "Update available…")
        .enabled(false)
        .build(app)?;

    let open_graph = MenuItemBuilder::with_id("open_graph", "Open Ormah").build(app)?;

    let quit = MenuItemBuilder::with_id("quit", "Quit Ormah").build(app)?;

    let menu = MenuBuilder::new(app)
        .item(&stats_week)
        .item(&stats_total)
        .separator()
        .item(&server_status)
        .item(&server_toggle)
        .separator()
        .item(&update_item)
        .item(&open_graph)
        .separator()
        .item(&quit)
        .build()?;

    // Activate the update item with version label when notified.
    let update_item_for_event = update_item.clone();
    handle.listen("ormah://update-available", move |event| {
        if let Ok(payload) = serde_json::from_str::<updater::UpdateAvailable>(event.payload()) {
            let label = format!("Install update {}…", payload.version);
            let _ = update_item_for_event.set_text(label);
            let _ = update_item_for_event.set_enabled(true);
        }
    });

    let server_toggle_for_event = server_toggle.clone();
    let server_status_for_event = server_status.clone();

    let tray = TrayIconBuilder::with_id("ormah-tray")
        .icon(tauri::include_image!("icons/icon.png"))
        .icon_as_template(false)
        .title("…")
        .tooltip("Ormah — click to open")
        .menu(&menu)
        .on_menu_event(move |app, event| match event.id().as_ref() {
            "install_update" => updater::install(app.clone()),
            "open_graph" => commands::open_graph(app),
            "server_toggle" => {
                let toggle = server_toggle_for_event.clone();
                let status = server_status_for_event.clone();
                let currently_running =
                    toggle.text().map(|t| t.contains("Stop")).unwrap_or(true);
                if currently_running {
                    sidecar::stop_daemon();
                } else {
                    sidecar::start_daemon();
                }
                // Optimistically flip labels; the poller will correct on next tick.
                let now_running = !currently_running;
                let _ = status.set_text(server_status_label(now_running));
                let _ = toggle.set_text(server_toggle_label(now_running));
            }
            "quit" => {
                sidecar::stop_daemon();
                app.exit(0);
            }
            _ => {}
        })
        // Left-click opens the graph; right-click (or platform default) shows menu.
        .on_tray_icon_event(|tray, event| {
            if let TrayIconEvent::Click {
                button: MouseButton::Left,
                button_state: MouseButtonState::Up,
                ..
            } = event
            {
                commands::open_graph(tray.app_handle());
            }
        })
        .build(app)?;

    // Poll /stats + health every 60s; updates tray title, menu counters,
    // and server status items.
    stats::spawn_poller(
        handle.clone(),
        tray,
        stats_week,
        stats_total,
        server_status,
        server_toggle,
    );

    Ok(())
}
