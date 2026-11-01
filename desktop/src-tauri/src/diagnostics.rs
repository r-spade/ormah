//! Best-effort, bounded setup logs. A logging failure must never break setup.
use once_cell::sync::Lazy;
use regex::Regex;
use std::fs::{File, OpenOptions};
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::{AppHandle, Manager, Runtime};

const MAX_LOG_BYTES: u64 = 1024 * 1024;
const COPIED_BYTES: u64 = 24 * 1024;
static LOG_DIR: Mutex<Option<PathBuf>> = Mutex::new(None);

pub fn init<R: Runtime>(app: &AppHandle<R>) {
    if let (Ok(dir), Ok(mut log_dir)) = (app.path().app_log_dir(), LOG_DIR.lock()) {
        *log_dir = Some(dir);
    }
}

fn log_dir() -> Option<PathBuf> {
    let dir = LOG_DIR.lock().ok()?.clone()?;
    std::fs::create_dir_all(&dir).ok()?;
    Some(dir)
}

fn private_file(options: &mut OpenOptions, path: &Path) -> std::io::Result<File> {
    #[cfg(unix)]
    {
        use std::os::unix::fs::OpenOptionsExt;
        options.mode(0o600);
    }
    options.open(path)
}

pub fn log(message: &str) {
    let Some(dir) = log_dir() else { return };
    let timestamp = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_secs();
    let _ = append(
        &dir.join("setup.log"),
        &format!("[{timestamp}] {}\n", redact(message)),
        MAX_LOG_BYTES,
    );
}

fn append(path: &Path, line: &str, limit: u64) -> std::io::Result<()> {
    if std::fs::metadata(path).is_ok_and(|m| m.len() >= limit) {
        std::fs::rename(path, path.with_extension("log.1"))?;
    }
    private_file(OpenOptions::new().create(true).append(true), path)?.write_all(line.as_bytes())
}

/// Stdout and stderr for `ormah server start -d`. Appends, because a running
/// fallback daemon may still hold this file; old output rotates at each start.
pub fn daemon_output() -> Option<(File, File)> {
    let path = log_dir()?.join("daemon.log");
    if std::fs::metadata(&path).is_ok_and(|m| m.len() >= MAX_LOG_BYTES) {
        std::fs::rename(&path, path.with_extension("log.1")).ok()?;
    }
    let stdout = private_file(OpenOptions::new().create(true).append(true), &path).ok()?;
    let stderr = stdout.try_clone().ok()?;
    Some((stdout, stderr))
}

/// Redacted log tails for the failure screen's Copy error log.
pub fn recent() -> String {
    let Some(dir) = log_dir() else {
        return "Setup log is unavailable.".into();
    };
    let mut details = format!("Logs: {}\n", dir.display());
    for name in ["setup.log.1", "setup.log", "daemon.log"] {
        if let Ok(tail) = read_tail(&dir.join(name)) {
            details.push_str(&format!("\n--- {name} ---\n{tail}"));
        }
    }
    redact(&details)
}

fn read_tail(path: &Path) -> std::io::Result<String> {
    let mut file = File::open(path)?;
    let len = file.metadata()?.len();
    file.seek(SeekFrom::Start(len.saturating_sub(COPIED_BYTES)))?;
    let mut bytes = Vec::new();
    file.read_to_end(&mut bytes)?;
    let tail = String::from_utf8_lossy(&bytes).into_owned();
    // Start on a whole line, so a secret's name can't be cut off from its value.
    Ok(match (len > COPIED_BYTES, tail.split_once('\n')) {
        (true, Some((_, whole_lines))) => whole_lines.to_string(),
        _ => tail,
    })
}

/// Proxy URLs in uv errors are the main risk, so redact URL credentials and named secrets.
pub fn redact(text: &str) -> String {
    static USERINFO: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?i)([a-z][a-z0-9+.-]*://)[^\s/@]+@").unwrap());
    static NAMED: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r#"(?i)(authorization\\?["']?\s*[:=]\s*|bearer\s+|[a-z0-9_-]*(?:token|secret|password|api[_-]?key)[a-z0-9_-]*\\?["']?\s*[:=]\s*)(?:basic\s+|bearer\s+)?(\\?"[^"]*"|'[^']*'|[^\s"'&,;]+)"#).unwrap()
    });
    static KNOWN_TOKEN: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"\b(?:gh[pousr]_[A-Za-z0-9]+|github_pat_\w+|glpat-[\w-]+|sk-[\w-]{8,}|(?:AKIA|ASIA)[A-Z0-9]{16}|eyJ[\w-]+\.[\w-]+\.[\w-]+)\b").unwrap()
    });
    let text = USERINFO.replace_all(text, "${1}[REDACTED]@");
    let text = NAMED.replace_all(&text, "${1}[REDACTED]");
    KNOWN_TOKEN.replace_all(&text, "[REDACTED]").into_owned()
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn redacts_credentials_and_keeps_useful_errors() {
        for text in [
            "http://user:secret1@proxy:9/p",
            "url?token=secret1&x=1",
            "Authorization: Bearer secret1",
            "Authorization: Basic secret1",
            r#"{"Authorization": "secret1"}"#,
            "password=\"secret1 with spaces\"",
            "api_key: secret1",
            r#"{\"token\":\"secret1\"}"#,
        ] {
            let safe = redact(text);
            assert!(!safe.contains("secret1"), "{safe}");
            assert!(safe.contains("[REDACTED]"), "{safe}");
        }
        for token in [
            "https://host/ghp_abc123/x",
            "gho_abc123",
            "github_pat_abc_123",
            "glpat-abc123",
            "sk-abcdefgh123",
            "AKIA1234567890123456",
            "eyJhbGci.eyJzdWIi.signature",
        ] {
            assert!(redact(token).contains("[REDACTED]"), "{token}");
        }
        assert!(redact("http://u:p@proxy:9/p").contains("proxy:9"));
        let error = "Failed to download cpython-3.11.17 (os error 61) Connection refused";
        assert_eq!(redact(error), error);
    }

    #[test]
    fn rotates_at_the_limit_and_keeps_one_old_file() {
        let dir = std::env::temp_dir().join(format!("ormah-log-test-{}", uuid::Uuid::new_v4()));
        std::fs::create_dir_all(&dir).unwrap();
        let path = dir.join("setup.log");
        for line in ["aaaa", "bbbb", "cccc"] {
            append(&path, line, 4).unwrap();
        }
        assert_eq!(std::fs::read_to_string(&path).unwrap(), "cccc");
        assert_eq!(
            std::fs::read_to_string(path.with_extension("log.1")).unwrap(),
            "bbbb"
        );
        std::fs::remove_dir_all(dir).unwrap();
    }
}
