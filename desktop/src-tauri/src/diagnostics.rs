//! Best-effort, bounded setup diagnostics. Never persist environment values.
use once_cell::sync::Lazy;
use regex::Regex;
use std::io::{Read, Seek, SeekFrom, Write};
use std::path::{Path, PathBuf};
use std::sync::Mutex;
use tauri::{AppHandle, Manager, Runtime};

const MAX_LOG_BYTES: u64 = 1024 * 1024;
static LOG: Mutex<Option<PathBuf>> = Mutex::new(None);

pub fn now_ms() -> u64 {
    std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as u64
}

pub fn init<R: Runtime>(app: &AppHandle<R>) {
    if let Ok(path) = app.path().app_log_dir() {
        if let Ok(mut log) = LOG.lock() {
            *log = Some(path.join("setup.log"));
        }
    }
}

pub fn path() -> Option<PathBuf> {
    LOG.lock().ok()?.clone()
}

pub fn log(message: &str) {
    // Hold the lock across rotation and append so concurrent streams cannot race.
    if let Ok(guard) = LOG.lock() {
        if let Some(path) = guard.as_ref() {
            let _ = append(
                path,
                &format!("[{} unix_ms] {}\n", now_ms(), redact(message)),
                MAX_LOG_BYTES,
            );
        }
    }
}

fn append(path: &Path, text: &str, limit: u64) -> std::io::Result<()> {
    if let Some(parent) = path.parent() {
        std::fs::create_dir_all(parent)?;
    }
    // The Linux daemon output collector can outlive the app. Lock a stable file
    // across rename as well as append so both processes share one rotation.
    #[cfg(unix)]
    let _lock = lock_log(path)?;
    // Split unusually large events as well, maintaining a hard bound on both files.
    let mut remaining = text;
    while !remaining.is_empty() {
        let size = std::fs::metadata(path).map(|m| m.len()).unwrap_or(0);
        if size >= limit {
            let rotated = path.with_extension("log.1");
            if rotated.exists() {
                std::fs::remove_file(&rotated)?;
            }
            std::fs::rename(path, rotated)?;
            continue;
        }
        let mut count = remaining.len().min((limit - size) as usize);
        while !remaining.is_char_boundary(count) {
            count -= 1;
        }
        if count == 0 {
            // A multi-byte character does not fit in the remaining space.
            let mut file = std::fs::OpenOptions::new().append(true).open(path)?;
            file.write_all(&vec![b' '; (limit - size) as usize])?;
            continue;
        }
        let mut options = std::fs::OpenOptions::new();
        options.create(true).append(true);
        #[cfg(unix)]
        {
            use std::os::unix::fs::OpenOptionsExt;
            options.mode(0o600);
        }
        options
            .open(path)?
            .write_all(&remaining.as_bytes()[..count])?;
        remaining = &remaining[count..];
    }
    Ok(())
}

#[cfg(unix)]
fn lock_log(path: &Path) -> std::io::Result<std::fs::File> {
    use std::os::fd::AsRawFd;
    use std::os::unix::fs::OpenOptionsExt;
    let file = std::fs::OpenOptions::new()
        .create(true)
        .read(true)
        .write(true)
        .mode(0o600)
        .open(path.with_extension("lock"))?;
    // SAFETY: file owns this descriptor for the duration of the lock. Closing
    // the returned File releases flock, including on an append error.
    if unsafe { libc::flock(file.as_raw_fd(), libc::LOCK_EX) } != 0 {
        return Err(std::io::Error::last_os_error());
    }
    Ok(file)
}

#[cfg(target_os = "linux")]
const COLLECTOR_ARG: &str = "--ormah-internal-log-collector";

/// A pipe collector must outlive the app when a fallback daemon inherits its
/// writers. This internal mode runs before Tauri/GTK initialization.
#[cfg(target_os = "linux")]
pub fn run_collector_if_requested() -> bool {
    let mut args = std::env::args_os().skip(1);
    if args.next().as_deref() != Some(std::ffi::OsStr::new(COLLECTOR_ARG)) {
        return false;
    }
    let Some(path) = args.next().map(PathBuf::from) else {
        return true;
    };
    if let Ok(mut target) = LOG.lock() {
        *target = Some(path);
    }
    let mut reader = std::io::stdin().lock();
    let mut redactor = OutputRedactor::default();
    let mut buffer = [0u8; 4096];
    while let Ok(count) = reader.read(&mut buffer) {
        if count == 0 {
            break;
        }
        let safe = redactor.push(&buffer[..count]);
        if !safe.is_empty() {
            log(&format!("daemon output: {safe}"));
        }
    }
    let safe = redactor.finish();
    if !safe.is_empty() {
        log(&format!("daemon output: {safe}"));
    }
    true
}

/// Capture output without making daemon lifetime depend on the GUI's pipes.
#[cfg(target_os = "linux")]
pub fn daemon_status(
    command: &mut std::process::Command,
) -> std::io::Result<std::process::ExitStatus> {
    use std::os::fd::OwnedFd;
    use std::os::unix::process::CommandExt;
    use std::process::{Command, Stdio};
    let collector = (|| -> std::io::Result<_> {
        let path = path().ok_or_else(|| std::io::Error::other("log directory unavailable"))?;
        let mut collector = Command::new(std::env::current_exe()?)
            .arg(COLLECTOR_ARG)
            .arg(path)
            .process_group(0)
            .stdin(Stdio::piped())
            .stdout(Stdio::null())
            .stderr(Stdio::null())
            .spawn()?;
        let input: OwnedFd = collector.stdin.take().unwrap().into();
        let writer = std::fs::File::from(input);
        let stderr = writer.try_clone()?;
        command
            .stdout(Stdio::from(writer))
            .stderr(Stdio::from(stderr));
        Ok(collector)
    })();
    match collector {
        Ok(mut collector) => {
            // Reap when the writers close; do not wait for EOF to report CLI exit.
            std::thread::spawn(move || {
                let _ = collector.wait();
            });
        }
        Err(error) => log(&format!("daemon output capture unavailable: {error}")),
    }
    command.status()
}

pub fn recent() -> String {
    let Some(path) = path() else {
        return "Setup log is unavailable.".into();
    };
    let read_tail = |path: &Path| -> std::io::Result<String> {
        let mut file = std::fs::File::open(path)?;
        let len = file.metadata()?.len();
        file.seek(SeekFrom::Start(len.saturating_sub(24 * 1024)))?;
        let mut bytes = Vec::new();
        file.read_to_end(&mut bytes)?;
        Ok(String::from_utf8_lossy(&bytes).into_owned())
    };
    // Include the rotation when the current file has just rolled over.
    let previous = read_tail(&path.with_extension("log.1")).unwrap_or_default();
    let current = read_tail(&path).unwrap_or_default();
    if previous.is_empty() && current.is_empty() {
        "Setup log is unavailable.".into()
    } else {
        redact(&format!("{previous}{current}"))
    }
}

pub fn redact(text: &str) -> String {
    static USERINFO: Lazy<Regex> =
        Lazy::new(|| Regex::new(r"(?i)([a-z][a-z0-9+.-]*://)[^\s/]*@").unwrap());
    static AUTH: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r#"(?i)(authorization[\s"'\\]*[:=]\s*)(?:\\?"[^"]*"|\\?'[^']*'|(?:(?:bearer|basic)\s+)?[^\s"'&,;]+)"#)
            .unwrap()
    });
    static NAMED: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r#"(?i)((?:bearer\s+)|(?:[a-z0-9_]*(?:token|secret|password|api[_-]?key)[a-z0-9_]*[\s"'\\]*[:=]\s*))(?:\\?"[^"]*"|\\?'[^']*'|[^\s"'&,;]+)"#).unwrap()
    });
    static TOKEN: Lazy<Regex> = Lazy::new(|| {
        Regex::new(r"\b(?:eyJ[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+\.[A-Za-z0-9_-]+|[A-Za-z0-9_-]{32,}(?:\.[A-Za-z0-9_-]+)*|(?:AKIA|ASIA)[A-Z0-9]{16}|(?:sk|ghp|gho|github_pat|glpat)[-_][A-Za-z0-9_-]+)\b").unwrap()
    });
    let text = USERINFO.replace_all(text, "${1}[REDACTED]@");
    let text = AUTH.replace_all(&text, "${1}[REDACTED]");
    let text = NAMED.replace_all(&text, "${1}[REDACTED]");
    TOKEN.replace_all(&text, "[REDACTED]").into_owned()
}

/// Buffer partial lines so credentials split across events never hit disk.
#[derive(Default)]
pub struct OutputRedactor {
    pending: String,
    discarding: bool,
}
impl OutputRedactor {
    pub fn push(&mut self, bytes: &[u8]) -> String {
        let mut safe = String::new();
        for ch in String::from_utf8_lossy(bytes).chars() {
            if ch == '\n' || ch == '\r' {
                if !self.discarding {
                    safe.push_str(&redact(&self.pending));
                }
                safe.push(ch);
                self.pending.clear();
                self.discarding = false;
            } else if !self.discarding {
                self.pending.push(ch);
                if self.pending.len() > 8192 {
                    safe.push_str("[REDACTED oversized field]");
                    self.pending.clear();
                    self.discarding = true;
                }
            }
        }
        safe
    }
    pub fn finish(&mut self) -> String {
        let safe = redact(&self.pending);
        self.pending.clear();
        safe
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    #[test]
    fn tauri_resolves_platform_log_directory() {
        let mut context = tauri::test::mock_context(tauri::test::noop_assets());
        context.config_mut().identifier = "dev.ormah.desktop".into();
        let app = tauri::test::mock_builder().build(context).unwrap();
        let path = app.path().app_log_dir().unwrap();
        #[cfg(target_os = "macos")]
        assert_eq!(
            path,
            PathBuf::from(std::env::var_os("HOME").unwrap()).join("Library/Logs/dev.ormah.desktop")
        );
        #[cfg(not(target_os = "macos"))]
        assert!(path.ends_with("dev.ormah.desktop/logs"));
        println!("Tauri app_log_dir: {}", path.display());
    }
    #[test]
    fn redacts_credentials_and_token_shapes() {
        let text = "http://user:pass@proxy:9/p?token=abc Bearer secret123 sk-testsecret ghp_secret123 abcdefghijklmnopqrstuvwxyz123456 token=short";
        let safe = redact(text);
        for secret in [
            "user:pass",
            "abc ",
            "secret123",
            "sk-testsecret",
            "ghp_secret123",
            "abcdefghijklmnopqrstuvwxyz123456",
            "=short",
        ] {
            assert!(!safe.contains(secret), "leaked {secret}: {safe}");
        }
        assert!(safe.contains("proxy:9"));
        for text in [
            "Authorization: Bearer secret123",
            "Authorization: Basic secret123",
            "token='secret with spaces'",
            "password=\"secret with spaces\"",
            "AKIA1234567890123456",
            "eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0In0.signature",
        ] {
            let safe = redact(text);
            assert!(!safe.contains("secret"), "{safe}");
            assert!(safe.contains("[REDACTED]"), "{safe}");
        }
        assert_eq!(
            redact("Connection refused (os error 111)"),
            "Connection refused (os error 111)"
        );
    }
    #[test]
    fn handles_credentials_split_between_events() {
        let mut stream = OutputRedactor::default();
        let a = stream.push(b"fetch http://user:pa");
        assert_eq!(a, "");
        let b = stream.push(b"ss@proxy:9/x?token=sec");
        let c = stream.push(b"ret\n");
        assert!(!format!("{a}{b}{c}").contains("secret"));
        assert!(c.contains("[REDACTED]"));
    }
    #[test]
    fn stream_redacts_bearer_and_spaced_secrets() {
        let mut stream = OutputRedactor::default();
        let mut output = stream.push(b"Authorization: Bearer ");
        output.push_str(&stream.push(b"secret123\n"));
        assert!(!output.contains("secret123"));
    }
    #[test]
    fn redacts_quoted_authorization_including_split_output() {
        for text in [
            r#"{"Authorization": "Basic dXNlcjpwYXNz"}"#,
            "Authorization: 'Basic dXNlcjpwYXNz'",
            r#"{"Authorization": "short-token"}"#,
            r#"{\"Authorization\": \"Basic dXNlcjpwYXNz\"}"#,
            r#"token=\"short-token\""#,
        ] {
            assert!(!redact(text).contains("dXNlcjpwYXNz"));
            assert!(!redact(text).contains("short-token"));
            let mut stream = OutputRedactor::default();
            let mut safe = String::new();
            for byte in text.bytes() {
                safe.push_str(&stream.push(&[byte]));
            }
            safe.push_str(&stream.finish());
            assert!(!safe.contains("dXNlcjpwYXNz"));
            assert!(!safe.contains("short-token"));
        }
    }
    #[test]
    fn logging_failure_does_not_panic() {
        let file = std::env::temp_dir().join(format!("ormah-log-failure-{}", uuid::Uuid::new_v4()));
        std::fs::write(&file, "not a directory").unwrap();
        let previous = LOG.lock().unwrap().replace(file.join("setup.log"));
        log("This failure must not break setup");
        *LOG.lock().unwrap() = previous;
        std::fs::remove_file(file).unwrap();
    }
    #[test]
    fn bounds_log_and_keeps_one_rotation() {
        let dir = std::env::temp_dir().join(format!("ormah-log-test-{}", uuid::Uuid::new_v4()));
        let path = dir.join("setup.log");
        append(&path, &"x".repeat(300), 100).unwrap();
        assert_eq!(std::fs::metadata(&path).unwrap().len(), 100);
        assert_eq!(
            std::fs::metadata(path.with_extension("log.1"))
                .unwrap()
                .len(),
            100
        );
        #[cfg(unix)]
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 3); // two logs + stable lock
        #[cfg(not(unix))]
        assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 2);
        std::fs::remove_dir_all(dir).unwrap();
    }
}
