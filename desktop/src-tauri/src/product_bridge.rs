//! Narrow desktop bridge for sensitive local account and protection operations.
//!
//! The product UI is served from the loopback Python service. It must not receive
//! the owner-only local API capability, cloud bearer tokens, presigned URLs, or
//! Stripe-hosted URLs. Each command below maps to one fixed local endpoint; there
//! is deliberately no generic HTTP proxy.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use sha2::{Digest, Sha256};
use std::fs::{Metadata, OpenOptions};
use std::io::{Read, Write};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tauri::{AppHandle, Runtime, WebviewWindow};
use tauri_plugin_dialog::DialogExt;
use tauri_plugin_shell::ShellExt;
use url::Url;

use crate::commands::base_url;

const BRIDGE_VERSION: u16 = 1;
const LOCAL_ADMIN_HEADER: &str = "X-Ormah-Local-Token";
// Synchronous intent routes may wait for the canonical store lock for up to
// 30 seconds. Keep the native request alive long enough to receive their
// typed store_busy outcome instead of abandoning a request that may proceed.
const REQUEST_TIMEOUT: Duration = Duration::from_secs(35);
const MAX_HOSTED_URL_CHARS: usize = 2048;
const CHECKOUT_HOST: &str = "checkout.stripe.com";
const PORTAL_HOST: &str = "billing.stripe.com";
const RECOVERY_KIT_FILENAME: &str = "ormah-recovery-kit.md";
const MAX_RECOVERY_KIT_BYTES: u64 = 256 * 1024;
const RECOVERY_PREPARE_PATH: &str = "/admin/cloud/protection/recovery-kit/prepare";
const RECOVERY_CONFIRM_PATH: &str = "/admin/cloud/protection/recovery-kit/confirm";

#[derive(Debug, Serialize)]
pub struct DesktopBridgeInfo {
    version: u16,
    platform: &'static str,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct AccountStatus {
    signed_in: bool,
    email: Option<String>,
    device_name: Option<String>,
    entitlement: Option<String>,
    plan_status: Option<String>,
    cache_age_seconds: Option<i64>,
    entitlement_available: bool,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct RequestCodeResult {
    status: String,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct LogoutResult {
    signed_in: bool,
    revoked_remotely: bool,
    warning: Option<String>,
}

#[derive(Debug, Deserialize, Serialize)]
pub struct BillingOffer {
    name: String,
    unit_amount: u64,
    currency: String,
    interval: String,
    interval_count: u64,
}

#[derive(Debug, Deserialize)]
struct CheckoutHandoff {
    status: String,
    url: Option<String>,
    expires_at: Option<i64>,
}

#[derive(Debug, Serialize)]
pub struct CheckoutResult {
    status: String,
    expires_at: Option<i64>,
    opened: bool,
}

#[derive(Debug, Deserialize)]
struct PortalHandoff {
    url: String,
}

#[derive(Debug, Serialize)]
pub struct OpenedResult {
    opened: bool,
}

#[derive(Debug, Deserialize, Serialize)]
#[serde(deny_unknown_fields)]
pub struct RecoveryKitReadiness {
    device_loss_recovery_ready: bool,
    recovery_kit_verified_at: String,
}

#[derive(Debug, Serialize)]
pub struct RecoveryKitActionResult {
    status: &'static str,
    device_loss_recovery_ready: Option<bool>,
    recovery_kit_verified_at: Option<String>,
}

#[derive(Debug, Serialize)]
pub struct PrintableRecoveryKitResult {
    opened: bool,
}

#[derive(Debug, PartialEq, Eq)]
enum RecoveryKitSaveOutcome {
    Canceled,
    Saved { digest: String },
}

#[tauri::command]
pub fn desktop_bridge_info() -> DesktopBridgeInfo {
    DesktopBridgeInfo {
        version: BRIDGE_VERSION,
        platform: std::env::consts::OS,
    }
}

#[tauri::command]
pub async fn account_status(window: WebviewWindow) -> Result<AccountStatus, String> {
    require_product_origin(&window)?;
    request_json("GET", "/admin/account/status", None).await
}

#[tauri::command]
pub async fn request_account_code(
    window: WebviewWindow,
    email: String,
) -> Result<RequestCodeResult, String> {
    require_product_origin(&window)?;
    let email = validate_email_input(&email)?;
    request_json(
        "POST",
        "/admin/account/request-code",
        Some(json!({ "email": email })),
    )
    .await
}

#[tauri::command]
pub async fn verify_account_code(
    window: WebviewWindow,
    email: String,
    code: String,
) -> Result<AccountStatus, String> {
    require_product_origin(&window)?;
    let email = validate_email_input(&email)?;
    let code = validate_otp_input(&code)?;
    request_json(
        "POST",
        "/admin/account/verify",
        Some(json!({ "email": email, "code": code })),
    )
    .await
}

#[tauri::command]
pub async fn logout_account(window: WebviewWindow) -> Result<LogoutResult, String> {
    require_product_origin(&window)?;
    request_json("POST", "/admin/account/logout", Some(json!({}))).await
}

#[tauri::command]
pub async fn billing_offer(window: WebviewWindow) -> Result<BillingOffer, String> {
    require_product_origin(&window)?;
    request_json("GET", "/admin/account/offer", None).await
}

#[tauri::command]
pub async fn protection_status(window: WebviewWindow) -> Result<Value, String> {
    require_product_origin(&window)?;
    request_sanitized("GET", "/admin/cloud/protection", None).await
}

#[tauri::command]
pub async fn create_protection_intent(window: WebviewWindow) -> Result<Value, String> {
    require_product_origin(&window)?;
    request_sanitized("POST", "/admin/cloud/protection/intents", Some(json!({}))).await
}

#[tauri::command]
pub async fn bind_protection_intent(
    window: WebviewWindow,
    intent_id: String,
) -> Result<Value, String> {
    require_product_origin(&window)?;
    let intent_id = validate_uuid4(&intent_id, "protection intent")?;
    let path = format!("/admin/cloud/protection/intents/{intent_id}/bind");
    request_sanitized("POST", &path, Some(json!({}))).await
}

#[tauri::command]
pub async fn cancel_protection_intent(
    window: WebviewWindow,
    intent_id: String,
) -> Result<Value, String> {
    require_product_origin(&window)?;
    let intent_id = validate_uuid4(&intent_id, "protection intent")?;
    let path = format!("/admin/cloud/protection/intents/{intent_id}/cancel");
    request_sanitized("POST", &path, Some(json!({}))).await
}

#[tauri::command]
pub async fn enable_protection(window: WebviewWindow, intent_id: String) -> Result<Value, String> {
    require_product_origin(&window)?;
    let intent_id = validate_uuid4(&intent_id, "protection intent")?;
    let path = format!("/admin/cloud/protection/intents/{intent_id}/enable");
    request_sanitized("POST", &path, Some(json!({}))).await
}

#[tauri::command]
pub async fn disable_protection(window: WebviewWindow) -> Result<Value, String> {
    require_product_origin(&window)?;
    request_sanitized("POST", "/admin/cloud/protection/disable", Some(json!({}))).await
}

#[tauri::command]
pub async fn backup_now(window: WebviewWindow) -> Result<Value, String> {
    require_product_origin(&window)?;
    request_sanitized("POST", "/admin/cloud/protection/backup", Some(json!({}))).await
}

#[tauri::command]
pub async fn verify_now(window: WebviewWindow) -> Result<Value, String> {
    require_product_origin(&window)?;
    request_sanitized("POST", "/admin/cloud/protection/verify", Some(json!({}))).await
}

#[tauri::command]
pub async fn operation_status(
    window: WebviewWindow,
    operation_id: String,
) -> Result<Value, String> {
    require_product_origin(&window)?;
    let operation_id = validate_uuid4(&operation_id, "operation")?;
    let path = format!("/admin/cloud/protection/operations/{operation_id}");
    request_sanitized("GET", &path, None).await
}

#[tauri::command]
pub async fn save_recovery_kit<R: Runtime>(
    app: AppHandle<R>,
    window: WebviewWindow<R>,
) -> Result<RecoveryKitActionResult, String> {
    require_product_origin(&window)?;
    let _: Value = request_json("POST", RECOVERY_PREPARE_PATH, Some(json!({}))).await?;
    let selected = app
        .dialog()
        .file()
        .set_title("Save sensitive Ormah recovery kit")
        .set_file_name(RECOVERY_KIT_FILENAME)
        .add_filter("Markdown document", &["md"])
        .blocking_save_file();
    let destination = match selected {
        Some(selection) => Some(
            selection
                .into_path()
                .map_err(|_| "Could not use the selected save location.".to_string())?,
        ),
        None => None,
    };
    let canonical = recovery_kit_path()?;
    let digest = match save_selected_recovery_kit(&canonical, destination.as_deref())? {
        RecoveryKitSaveOutcome::Canceled => {
            return Ok(RecoveryKitActionResult {
                status: "canceled",
                device_loss_recovery_ready: None,
                recovery_kit_verified_at: None,
            });
        }
        RecoveryKitSaveOutcome::Saved { digest } => digest,
    };
    let readiness: RecoveryKitReadiness = request_json(
        "POST",
        RECOVERY_CONFIRM_PATH,
        Some(json!({ "sha256_digest": digest })),
    )
    .await?;
    if readiness.recovery_kit_verified_at.is_empty()
        || readiness.recovery_kit_verified_at.len() > 64
    {
        return Err("The saved recovery kit could not be verified.".to_string());
    }
    Ok(RecoveryKitActionResult {
        status: "saved",
        device_loss_recovery_ready: Some(readiness.device_loss_recovery_ready),
        recovery_kit_verified_at: Some(readiness.recovery_kit_verified_at),
    })
}

#[tauri::command]
pub async fn open_printable_recovery_kit<R: Runtime>(
    app: AppHandle<R>,
    window: WebviewWindow<R>,
) -> Result<PrintableRecoveryKitResult, String> {
    require_product_origin(&window)?;
    let canonical = recovery_kit_path()?;
    read_bounded_recovery_kit(&canonical)?;
    let printable = canonical
        .to_str()
        .ok_or_else(|| "The printable recovery kit is unavailable.".to_string())?;
    open_external(&app, printable)
        .map_err(|_| "Could not open the printable recovery kit.".to_string())?;
    Ok(PrintableRecoveryKitResult { opened: true })
}

#[tauri::command]
pub async fn open_checkout<R: Runtime>(
    app: AppHandle<R>,
    window: WebviewWindow<R>,
    intent_id: String,
) -> Result<CheckoutResult, String> {
    require_product_origin(&window)?;
    let intent_id = validate_uuid4(&intent_id, "protection intent")?;
    let handoff: CheckoutHandoff = request_json(
        "POST",
        "/admin/account/checkout",
        Some(json!({ "protection_intent_id": intent_id })),
    )
    .await?;

    let hosted_url = validated_checkout_handoff(&handoff)?;
    let opened = if let Some(hosted_url) = hosted_url {
        open_external(&app, hosted_url.as_str())?;
        true
    } else {
        false
    };

    Ok(CheckoutResult {
        status: handoff.status,
        expires_at: handoff.expires_at,
        opened,
    })
}

fn validated_checkout_handoff(handoff: &CheckoutHandoff) -> Result<Option<Url>, String> {
    match handoff.status.as_str() {
        "checkout_required" => {
            if handoff.expires_at.is_none() {
                return Err("Ormah Cloud returned an incomplete Checkout handoff.".to_string());
            }
            let raw_url = handoff.url.as_deref().ok_or_else(|| {
                "Ormah Cloud returned an incomplete Checkout handoff.".to_string()
            })?;
            validate_hosted_url(raw_url, CHECKOUT_HOST).map(Some)
        }
        "already_subscribed" | "subscription_pending" => {
            if handoff.url.is_some() || handoff.expires_at.is_some() {
                return Err("Ormah Cloud returned an unexpected Checkout handoff.".to_string());
            }
            Ok(None)
        }
        _ => Err("Ormah Cloud returned an invalid Checkout handoff.".to_string()),
    }
}

#[tauri::command]
pub async fn open_billing_portal<R: Runtime>(
    app: AppHandle<R>,
    window: WebviewWindow<R>,
) -> Result<OpenedResult, String> {
    require_product_origin(&window)?;
    let handoff: PortalHandoff =
        request_json("POST", "/admin/account/portal", Some(json!({}))).await?;
    let hosted_url = validate_hosted_url(&handoff.url, PORTAL_HOST)?;
    open_external(&app, hosted_url.as_str())?;
    Ok(OpenedResult { opened: true })
}

fn open_external<R: Runtime>(app: &AppHandle<R>, url: &str) -> Result<(), String> {
    // This Rust-side API is not exposed to the frontend. The remote graph
    // capability intentionally has no generic `shell:allow-open` permission.
    #[allow(deprecated)]
    app.shell()
        .open(url, None)
        .map_err(|_| "Could not open the system browser.".to_string())
}

fn recovery_kit_path() -> Result<PathBuf, String> {
    Ok(user_home()?
        .join(".config/ormah")
        .join(RECOVERY_KIT_FILENAME))
}

fn user_home() -> Result<PathBuf, String> {
    std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| "The local Ormah configuration is unavailable.".to_string())
}

fn read_bounded_recovery_kit(path: &Path) -> Result<Vec<u8>, String> {
    read_bounded_recovery_kit_with_metadata(path).map(|(bytes, _)| bytes)
}

fn read_bounded_recovery_kit_with_metadata(path: &Path) -> Result<(Vec<u8>, Metadata), String> {
    let mut options = OpenOptions::new();
    options.read(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let mut file = options
        .open(path)
        .map_err(|_| "The recovery kit is unavailable.".to_string())?;
    let metadata = file
        .metadata()
        .map_err(|_| "The recovery kit is unavailable.".to_string())?;
    if !metadata.is_file() || metadata.len() > MAX_RECOVERY_KIT_BYTES {
        return Err("The recovery kit is invalid.".to_string());
    }
    let mut bytes = Vec::with_capacity(metadata.len() as usize);
    Read::by_ref(&mut file)
        .take(MAX_RECOVERY_KIT_BYTES + 1)
        .read_to_end(&mut bytes)
        .map_err(|_| "The recovery kit is unavailable.".to_string())?;
    if bytes.len() as u64 > MAX_RECOVERY_KIT_BYTES {
        return Err("The recovery kit is invalid.".to_string());
    }
    Ok((bytes, metadata))
}

fn save_selected_recovery_kit(
    canonical: &Path,
    destination: Option<&Path>,
) -> Result<RecoveryKitSaveOutcome, String> {
    let Some(destination) = destination else {
        return Ok(RecoveryKitSaveOutcome::Canceled);
    };
    let (canonical_bytes, canonical_metadata) = read_bounded_recovery_kit_with_metadata(canonical)?;
    if paths_resolve_to_same_file(canonical, destination)? {
        return Err("Choose a separate location for the recovery kit.".to_string());
    }

    let mut options = OpenOptions::new();
    options.create(true).write(true);
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600).custom_flags(libc::O_NOFOLLOW);
    }
    #[cfg(windows)]
    {
        use std::os::windows::fs::OpenOptionsExt;
        const FILE_FLAG_OPEN_REPARSE_POINT: u32 = 0x0020_0000;
        options.custom_flags(FILE_FLAG_OPEN_REPARSE_POINT);
    }
    let mut output = options
        .open(destination)
        .map_err(|_| "Could not save the recovery kit.".to_string())?;
    let metadata = output
        .metadata()
        .map_err(|_| "Could not save the recovery kit.".to_string())?;
    if !metadata.is_file() || metadata.file_type().is_symlink() {
        return Err("Could not save the recovery kit.".to_string());
    }
    if metadata_is_same_file(&canonical_metadata, &metadata) {
        return Err("Choose a separate location for the recovery kit.".to_string());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        output
            .set_permissions(std::fs::Permissions::from_mode(0o600))
            .map_err(|_| "Could not secure the saved recovery kit.".to_string())?;
    }
    output
        .set_len(0)
        .and_then(|_| output.write_all(&canonical_bytes))
        .and_then(|_| output.sync_all())
        .map_err(|_| "Could not save the recovery kit.".to_string())?;
    drop(output);
    sync_parent_directory(destination)?;

    let reopened = read_bounded_recovery_kit(destination)
        .map_err(|_| "The saved recovery kit could not be reopened.".to_string())?;
    if reopened != canonical_bytes {
        return Err("The saved recovery kit did not match the current kit.".to_string());
    }
    let digest = format!("{:x}", Sha256::digest(&reopened));
    Ok(RecoveryKitSaveOutcome::Saved { digest })
}

#[cfg(unix)]
fn sync_parent_directory(path: &Path) -> Result<(), String> {
    let parent = path
        .parent()
        .ok_or_else(|| "Could not secure the saved recovery kit.".to_string())?;
    std::fs::File::open(parent)
        .and_then(|directory| directory.sync_all())
        .map_err(|_| "Could not secure the saved recovery kit.".to_string())
}

#[cfg(not(unix))]
fn sync_parent_directory(_path: &Path) -> Result<(), String> {
    Ok(())
}

#[cfg(unix)]
fn metadata_is_same_file(left: &Metadata, right: &Metadata) -> bool {
    use std::os::unix::fs::MetadataExt;
    left.dev() == right.dev() && left.ino() == right.ino()
}

#[cfg(windows)]
fn metadata_is_same_file(left: &Metadata, right: &Metadata) -> bool {
    use std::os::windows::fs::MetadataExt;
    match (
        (left.volume_serial_number(), left.file_index()),
        (right.volume_serial_number(), right.file_index()),
    ) {
        ((Some(left_volume), Some(left_index)), (Some(right_volume), Some(right_index))) => {
            left_volume == right_volume && left_index == right_index
        }
        _ => false,
    }
}

#[cfg(not(any(unix, windows)))]
fn metadata_is_same_file(_left: &Metadata, _right: &Metadata) -> bool {
    false
}

fn paths_resolve_to_same_file(canonical: &Path, destination: &Path) -> Result<bool, String> {
    let source = std::fs::canonicalize(canonical)
        .map_err(|_| "The recovery kit is unavailable.".to_string())?;
    if destination.exists() {
        let resolved_destination = std::fs::canonicalize(destination)
            .map_err(|_| "Could not inspect the save location.".to_string())?;
        if resolved_destination == source {
            return Ok(true);
        }
        #[cfg(unix)]
        {
            use std::os::unix::fs::MetadataExt;
            let source_metadata = std::fs::metadata(&source)
                .map_err(|_| "The recovery kit is unavailable.".to_string())?;
            let destination_metadata = std::fs::metadata(&resolved_destination)
                .map_err(|_| "Could not inspect the save location.".to_string())?;
            return Ok(source_metadata.dev() == destination_metadata.dev()
                && source_metadata.ino() == destination_metadata.ino());
        }
        #[cfg(not(unix))]
        return Ok(false);
    }
    let parent = destination
        .parent()
        .ok_or_else(|| "Could not inspect the save location.".to_string())?;
    let filename = destination
        .file_name()
        .ok_or_else(|| "Could not inspect the save location.".to_string())?;
    let resolved_parent = std::fs::canonicalize(parent)
        .map_err(|_| "Could not inspect the save location.".to_string())?;
    Ok(resolved_parent.join(filename) == source)
}

fn require_product_origin<R: Runtime>(window: &WebviewWindow<R>) -> Result<(), String> {
    if window.label() != "main" {
        return Err("Desktop product commands are unavailable in this window.".to_string());
    }
    let current = window
        .url()
        .map_err(|_| "Could not verify the desktop product origin.".to_string())?;
    validate_product_origin(&current, &base_url())
}

fn validate_product_origin(current: &Url, expected_base: &str) -> Result<(), String> {
    let expected = Url::parse(expected_base)
        .map_err(|_| "The local Ormah service address is invalid.".to_string())?;
    let matches = current.scheme() == "http"
        && current.username().is_empty()
        && current.password().is_none()
        && current.host_str() == expected.host_str()
        && current.port_or_known_default() == expected.port_or_known_default();
    if !matches {
        return Err("Desktop product commands require the local Ormah UI.".to_string());
    }
    Ok(())
}

async fn request_sanitized(method: &str, path: &str, body: Option<Value>) -> Result<Value, String> {
    let value: Value = request_json(method, path, body).await?;
    reject_forbidden_response_fields(&value)?;
    Ok(value)
}

async fn request_json<T: for<'de> Deserialize<'de>>(
    method: &str,
    path: &str,
    body: Option<Value>,
) -> Result<T, String> {
    let token = load_local_admin_token(&local_admin_token_path()?)?;
    let url = format!("{}{path}", base_url());
    let client = reqwest::Client::builder()
        .timeout(REQUEST_TIMEOUT)
        .build()
        .map_err(|_| "Could not initialize the local Ormah connection.".to_string())?;
    let mut request = match method {
        "GET" => client.get(url),
        "POST" => client.post(url),
        _ => return Err("Unsupported local operation.".to_string()),
    }
    .header(LOCAL_ADMIN_HEADER, token);
    if let Some(body) = body {
        request = request.json(&body);
    }
    let response = request
        .send()
        .await
        .map_err(|_| "Could not reach the local Ormah service.".to_string())?;
    if !response.status().is_success() {
        let message = local_error_message(response.status().as_u16(), path);
        return Err(message.to_string());
    }
    response
        .json::<T>()
        .await
        .map_err(|_| "The local Ormah service returned an invalid response.".to_string())
}

fn local_error_message(status: u16, path: &str) -> &'static str {
    match status {
        401 if path == "/admin/account/verify" => {
            "That code is wrong, expired, or already used."
        }
        401 => "Sign in to Ormah before continuing.",
        403 => "This local Ormah operation is not allowed.",
        404 => "This Ormah Desktop feature requires a newer Ormah runtime.",
        409 if path == RECOVERY_CONFIRM_PATH => {
            "The saved recovery kit no longer matches current recovery material. Run `ormah cloud kit`, then save it again."
        }
        409 => "The protection state changed; refresh and try again.",
        429 => "Too many requests; wait briefly and try again.",
        _ => "The local Ormah operation could not be completed.",
    }
}

fn local_admin_token_path() -> Result<PathBuf, String> {
    Ok(user_home()?.join(".local/share/ormah/local_api_token"))
}

fn load_local_admin_token(path: &Path) -> Result<String, String> {
    let metadata = std::fs::symlink_metadata(path)
        .map_err(|_| "The local Ormah capability is unavailable.".to_string())?;
    if metadata.file_type().is_symlink() || !metadata.file_type().is_file() {
        return Err("The local Ormah capability is unavailable.".to_string());
    }
    if metadata.len() > 128 {
        return Err("The local Ormah capability is invalid.".to_string());
    }
    #[cfg(unix)]
    {
        use std::os::unix::fs::PermissionsExt;
        if metadata.permissions().mode() & 0o777 != 0o600 {
            return Err("The local Ormah capability is not secured.".to_string());
        }
    }
    let token = std::fs::read_to_string(path)
        .map_err(|_| "The local Ormah capability is unavailable.".to_string())?;
    let token = token.trim();
    if token.len() != 64
        || !token
            .bytes()
            .all(|byte| byte.is_ascii_digit() || (b'a'..=b'f').contains(&byte))
    {
        return Err("The local Ormah capability is invalid.".to_string());
    }
    Ok(token.to_owned())
}

fn validate_hosted_url(raw: &str, expected_host: &str) -> Result<Url, String> {
    if raw.len() > MAX_HOSTED_URL_CHARS {
        return Err("Ormah Cloud returned an invalid hosted billing URL.".to_string());
    }
    // `url::Url` normalizes an explicit `:443` away. Check the raw authority
    // first so userinfo and every explicit port remain rejected.
    let authority = raw
        .strip_prefix("https://")
        .and_then(|rest| rest.split(['/', '?', '#']).next());
    if authority != Some(expected_host) {
        return Err("Ormah Cloud returned an invalid hosted billing URL.".to_string());
    }
    let url = Url::parse(raw)
        .map_err(|_| "Ormah Cloud returned an invalid hosted billing URL.".to_string())?;
    if url.scheme() != "https"
        || url.host_str() != Some(expected_host)
        || !url.username().is_empty()
        || url.password().is_some()
        || url.port().is_some()
    {
        return Err("Ormah Cloud returned an invalid hosted billing URL.".to_string());
    }
    Ok(url)
}

fn validate_email_input(value: &str) -> Result<&str, String> {
    let value = value.trim();
    if value.is_empty() || value.len() > 254 || value.contains(['\r', '\n']) {
        return Err("Enter a valid email address.".to_string());
    }
    Ok(value)
}

fn validate_otp_input(value: &str) -> Result<&str, String> {
    let value = value.trim();
    if value.len() != 6 || !value.bytes().all(|byte| byte.is_ascii_digit()) {
        return Err("Enter the six-digit code from your email.".to_string());
    }
    Ok(value)
}

fn validate_uuid4(value: &str, label: &str) -> Result<String, String> {
    let parsed = uuid::Uuid::parse_str(value).map_err(|_| format!("Invalid {label} ID."))?;
    if parsed.get_version_num() != 4 || parsed.to_string() != value {
        return Err(format!("Invalid {label} ID."));
    }
    Ok(value.to_owned())
}

fn reject_forbidden_response_fields(value: &Value) -> Result<(), String> {
    const FORBIDDEN: &[&str] = &[
        "token",
        "accounttoken",
        "accountid",
        "accesstoken",
        "puturl",
        "geturl",
        "presignedurl",
        "url",
        "key",
        "privatekey",
        "remotekey",
        "path",
        "filepath",
        "identity",
        "identities",
        "recoverykit",
    ];
    match value {
        Value::Object(values) => {
            for (key, nested) in values {
                let normalized = key
                    .chars()
                    .filter(|character| character.is_ascii_alphanumeric())
                    .flat_map(char::to_lowercase)
                    .collect::<String>();
                if FORBIDDEN.contains(&normalized.as_str()) {
                    return Err("The local Ormah service returned forbidden data.".to_string());
                }
                reject_forbidden_response_fields(nested)?;
            }
        }
        Value::Array(values) => {
            for nested in values {
                reject_forbidden_response_fields(nested)?;
            }
        }
        Value::String(value) => {
            let lower = value.to_ascii_lowercase();
            if lower.contains("age-secret-key-")
                || lower.contains("bearer ")
                || lower.contains("x-amz-signature=")
                || lower.contains("x-amz-credential=")
                || lower.contains("x-amz-security-token=")
            {
                return Err("The local Ormah service returned forbidden data.".to_string());
            }
        }
        _ => {}
    }
    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::fs;

    #[test]
    fn product_origin_requires_exact_loopback_host_and_port() {
        let expected = "http://127.0.0.1:8787";
        assert!(validate_product_origin(
            &Url::parse("http://127.0.0.1:8787/graph?x=1").unwrap(),
            expected
        )
        .is_ok());
        for raw in [
            "http://localhost:8787/",
            "http://127.0.0.1:8788/",
            "https://127.0.0.1:8787/",
            "http://127.0.0.1.evil.example:8787/",
            "http://user@127.0.0.1:8787/",
        ] {
            assert!(validate_product_origin(&Url::parse(raw).unwrap(), expected).is_err());
        }
    }

    #[test]
    fn remote_graph_has_no_generic_dialog_file_or_shell_permission() {
        let capability: Value =
            serde_json::from_str(include_str!("../capabilities/graph.json")).unwrap();
        let permissions = capability["permissions"].as_array().unwrap();
        assert!(permissions
            .iter()
            .any(|permission| permission == "desktop-product-bridge"));
        for permission in permissions.iter().filter_map(Value::as_str) {
            assert!(!permission.starts_with("dialog:"), "{permission}");
            assert!(!permission.starts_with("fs:"), "{permission}");
            assert!(!permission.starts_with("shell:"), "{permission}");
        }
    }

    #[test]
    fn hosted_urls_require_exact_stripe_origin() {
        assert!(validate_hosted_url(
            "https://checkout.stripe.com/c/pay/cs_test_123?x=1",
            CHECKOUT_HOST
        )
        .is_ok());
        for raw in [
            "http://checkout.stripe.com/c/pay/x",
            "https://checkout.stripe.com.evil.example/c/pay/x",
            "https://evil-checkout.stripe.com/c/pay/x",
            "https://user@checkout.stripe.com/c/pay/x",
            "https://checkout.stripe.com:443/c/pay/x",
            "javascript:alert(1)",
        ] {
            assert!(validate_hosted_url(raw, CHECKOUT_HOST).is_err(), "{raw}");
        }
        assert!(
            validate_hosted_url("https://billing.stripe.com/p/session-123", PORTAL_HOST).is_ok()
        );
    }

    #[test]
    fn hosted_url_length_is_bounded() {
        let raw = format!(
            "https://checkout.stripe.com/{}",
            "a".repeat(MAX_HOSTED_URL_CHARS)
        );
        assert!(validate_hosted_url(&raw, CHECKOUT_HOST).is_err());
    }

    #[test]
    fn checkout_handoff_is_purpose_bound() {
        let required = CheckoutHandoff {
            status: "checkout_required".to_string(),
            url: Some("https://checkout.stripe.com/c/pay/cs_test_123".to_string()),
            expires_at: Some(4_000_000_000),
        };
        assert!(validated_checkout_handoff(&required).unwrap().is_some());

        let subscribed = CheckoutHandoff {
            status: "already_subscribed".to_string(),
            url: None,
            expires_at: None,
        };
        assert!(validated_checkout_handoff(&subscribed).unwrap().is_none());

        for invalid in [
            CheckoutHandoff {
                status: "checkout_required".to_string(),
                url: None,
                expires_at: Some(4_000_000_000),
            },
            CheckoutHandoff {
                status: "already_subscribed".to_string(),
                url: Some("https://checkout.stripe.com/c/pay/x".to_string()),
                expires_at: None,
            },
            CheckoutHandoff {
                status: "future_status".to_string(),
                url: None,
                expires_at: None,
            },
        ] {
            assert!(validated_checkout_handoff(&invalid).is_err());
        }
    }

    #[test]
    fn local_capability_is_validated_without_returning_file_details() {
        let root = std::env::temp_dir().join(format!("ormah-bridge-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("capability");
        fs::write(&path, format!("{}\n", "a".repeat(64))).unwrap();
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        }
        assert_eq!(load_local_admin_token(&path).unwrap(), "a".repeat(64));

        fs::write(&path, format!("{}\n", "A".repeat(64))).unwrap();
        assert!(load_local_admin_token(&path).is_err());
        fs::write(&path, "not-a-capability\n").unwrap();
        assert_eq!(
            load_local_admin_token(&path).unwrap_err(),
            "The local Ormah capability is invalid."
        );
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn local_capability_rejects_broad_permissions_and_symlinks() {
        use std::os::unix::fs::{symlink, PermissionsExt};

        let root = std::env::temp_dir().join(format!("ormah-bridge-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let path = root.join("capability");
        fs::write(&path, "a".repeat(64)).unwrap();
        fs::set_permissions(&path, fs::Permissions::from_mode(0o644)).unwrap();
        assert!(load_local_admin_token(&path).is_err());
        fs::set_permissions(&path, fs::Permissions::from_mode(0o600)).unwrap();
        let link = root.join("link");
        symlink(&path, &link).unwrap();
        assert!(load_local_admin_token(&link).is_err());
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn identifiers_and_otp_are_canonical() {
        let id = uuid::Uuid::new_v4().to_string();
        assert_eq!(validate_uuid4(&id, "operation").unwrap(), id);
        assert!(validate_uuid4(&id.to_ascii_uppercase(), "operation").is_err());
        assert!(validate_uuid4(&uuid::Uuid::new_v4().simple().to_string(), "operation").is_err());
        assert!(validate_otp_input("123456").is_ok());
        assert!(validate_otp_input("12345x").is_err());
    }

    #[test]
    fn protection_payload_rejects_secret_bearing_fields() {
        assert!(reject_forbidden_response_fields(&json!({
            "state": "protected",
            "operation": { "id": "safe" }
        }))
        .is_ok());
        for key in [
            "token",
            "accountToken",
            "account_id",
            "put_url",
            "remote_key",
            "path",
            "recovery-kit",
            "identity",
        ] {
            let mut nested = serde_json::Map::new();
            nested.insert(key.to_string(), Value::String("secret".to_string()));
            assert!(reject_forbidden_response_fields(&Value::Object(nested)).is_err());
        }
        for value in [
            "Bearer opaque-cloud-token",
            "AGE-SECRET-KEY-1EXAMPLE",
            "https://r2.example/object?X-Amz-Signature=secret",
            "https://r2.example/object?X-Amz-Credential=secret",
            "https://r2.example/object?X-Amz-Security-Token=secret",
        ] {
            assert!(reject_forbidden_response_fields(&json!({
                "message": value
            }))
            .is_err());
        }
    }

    #[test]
    fn recovery_kit_save_reopens_exact_bytes_and_hashes_them() {
        let root = std::env::temp_dir().join(format!("ormah-recovery-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let canonical = root.join("canonical.md");
        let destination = root.join("saved.md");
        let contents = b"# Ormah Recovery Kit\nAGE-SECRET-KEY-TEST-MATERIAL\n";
        fs::write(&canonical, contents).unwrap();

        let outcome = save_selected_recovery_kit(&canonical, Some(&destination)).unwrap();

        let expected = format!("{:x}", Sha256::digest(contents));
        assert_eq!(outcome, RecoveryKitSaveOutcome::Saved { digest: expected });
        assert_eq!(fs::read(&destination).unwrap(), contents);
        #[cfg(unix)]
        {
            use std::os::unix::fs::PermissionsExt;
            assert_eq!(
                fs::metadata(&destination).unwrap().permissions().mode() & 0o777,
                0o600
            );
        }
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn canceled_recovery_kit_save_does_no_file_work() {
        let missing = PathBuf::from("/definitely/not/a/recovery-kit");
        assert_eq!(
            save_selected_recovery_kit(&missing, None).unwrap(),
            RecoveryKitSaveOutcome::Canceled
        );
    }

    #[test]
    fn recovery_kit_reads_are_bounded_and_save_must_be_separate() {
        let root = std::env::temp_dir().join(format!("ormah-recovery-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let canonical = root.join("canonical.md");
        fs::write(&canonical, vec![b'x'; MAX_RECOVERY_KIT_BYTES as usize + 1]).unwrap();
        assert!(read_bounded_recovery_kit(&canonical).is_err());

        fs::write(&canonical, b"current kit").unwrap();
        assert!(save_selected_recovery_kit(&canonical, Some(&canonical)).is_err());
        assert_eq!(fs::read(&canonical).unwrap(), b"current kit");
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn recovery_kit_save_refuses_symlink_destinations() {
        use std::os::unix::fs::symlink;

        let root = std::env::temp_dir().join(format!("ormah-recovery-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let canonical = root.join("canonical.md");
        let target = root.join("target.md");
        let destination = root.join("destination.md");
        fs::write(&canonical, b"current kit").unwrap();
        fs::write(&target, b"do not overwrite").unwrap();
        symlink(&target, &destination).unwrap();

        assert!(save_selected_recovery_kit(&canonical, Some(&destination)).is_err());
        assert_eq!(fs::read(&target).unwrap(), b"do not overwrite");
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn recovery_kit_save_refuses_hard_links_to_the_canonical_file() {
        let root = std::env::temp_dir().join(format!("ormah-recovery-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let canonical = root.join("canonical.md");
        let destination = root.join("destination.md");
        fs::write(&canonical, b"current kit").unwrap();
        fs::hard_link(&canonical, &destination).unwrap();

        assert!(save_selected_recovery_kit(&canonical, Some(&destination)).is_err());
        assert_eq!(fs::read(&canonical).unwrap(), b"current kit");
        let _ = fs::remove_dir_all(root);
    }

    #[cfg(unix)]
    #[test]
    fn recovery_kit_file_identity_is_checked_from_open_handles() {
        let root = std::env::temp_dir().join(format!("ormah-recovery-{}", uuid::Uuid::new_v4()));
        fs::create_dir_all(&root).unwrap();
        let canonical = root.join("canonical.md");
        let alias = root.join("alias.md");
        fs::write(&canonical, b"current kit").unwrap();
        fs::hard_link(&canonical, &alias).unwrap();

        assert!(metadata_is_same_file(
            &fs::File::open(&canonical).unwrap().metadata().unwrap(),
            &fs::File::open(&alias).unwrap().metadata().unwrap(),
        ));
        let _ = fs::remove_dir_all(root);
    }

    #[test]
    fn recovery_action_results_are_purpose_bound_and_secret_free() {
        let canceled = serde_json::to_value(RecoveryKitActionResult {
            status: "canceled",
            device_loss_recovery_ready: None,
            recovery_kit_verified_at: None,
        })
        .unwrap();
        let saved = serde_json::to_value(RecoveryKitActionResult {
            status: "saved",
            device_loss_recovery_ready: Some(false),
            recovery_kit_verified_at: Some("2026-07-31T12:00:00+00:00".to_string()),
        })
        .unwrap();

        assert_eq!(
            canceled,
            json!({
                "status": "canceled",
                "device_loss_recovery_ready": null,
                "recovery_kit_verified_at": null
            })
        );
        assert_eq!(
            saved,
            json!({
                "status": "saved",
                "device_loss_recovery_ready": false,
                "recovery_kit_verified_at": "2026-07-31T12:00:00+00:00"
            })
        );
        let serialized = serde_json::to_string(&saved).unwrap().to_ascii_lowercase();
        for forbidden in [
            "age-secret-key",
            "sha256",
            "digest",
            "filepath",
            "presigned",
            "accounttoken",
        ] {
            assert!(!serialized.contains(forbidden));
        }
    }

    #[test]
    fn recovery_confirmation_conflict_has_a_specific_repair_message() {
        let recovery = local_error_message(409, RECOVERY_CONFIRM_PATH);
        let generic = local_error_message(409, "/admin/cloud/protection/backup");

        assert!(recovery.contains("ormah cloud kit"));
        assert!(recovery.contains("saved recovery kit"));
        assert_eq!(
            generic,
            "The protection state changed; refresh and try again."
        );
    }
}
