//! Narrow desktop bridge for sensitive local account and protection operations.
//!
//! The product UI is served from the loopback Python service. It must not receive
//! the owner-only local API capability, cloud bearer tokens, presigned URLs, or
//! Stripe-hosted URLs. Each command below maps to one fixed local endpoint; there
//! is deliberately no generic HTTP proxy.

use serde::{Deserialize, Serialize};
use serde_json::{json, Value};
use std::path::{Path, PathBuf};
use std::time::Duration;
use tauri::{AppHandle, Runtime, WebviewWindow};
use tauri_plugin_shell::ShellExt;
use url::Url;

use crate::commands::base_url;

const BRIDGE_VERSION: u16 = 1;
const LOCAL_ADMIN_HEADER: &str = "X-Ormah-Local-Token";
const REQUEST_TIMEOUT: Duration = Duration::from_secs(15);
const MAX_HOSTED_URL_CHARS: usize = 2048;
const CHECKOUT_HOST: &str = "checkout.stripe.com";
const PORTAL_HOST: &str = "billing.stripe.com";

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
        let message = match response.status().as_u16() {
            401 if path == "/admin/account/verify" => {
                "That code is wrong, expired, or already used."
            }
            401 => "Sign in to Ormah before continuing.",
            403 => "This local Ormah operation is not allowed.",
            404 => "This Ormah Desktop feature requires a newer Ormah runtime.",
            409 => "The protection state changed; refresh and try again.",
            429 => "Too many requests; wait briefly and try again.",
            _ => "The local Ormah operation could not be completed.",
        };
        return Err(message.to_string());
    }
    response
        .json::<T>()
        .await
        .map_err(|_| "The local Ormah service returned an invalid response.".to_string())
}

fn local_admin_token_path() -> Result<PathBuf, String> {
    let home = std::env::var_os("HOME")
        .or_else(|| std::env::var_os("USERPROFILE"))
        .filter(|value| !value.is_empty())
        .map(PathBuf::from)
        .ok_or_else(|| "The local Ormah capability is unavailable.".to_string())?;
    Ok(home.join(".local/share/ormah/local_api_token"))
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
}
