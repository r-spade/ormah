#![cfg(target_os = "linux")]

use std::io::Write;
use std::os::fd::OwnedFd;
use std::os::unix::process::CommandExt;
use std::process::{Command, Stdio};
use std::time::{Duration, Instant};

#[test]
fn detached_collector_preserves_output_after_cli_and_parent_handles_close() {
    let dir = std::env::temp_dir().join(format!("ormah-collector-{}", uuid::Uuid::new_v4()));
    let log = dir.join("setup.log");
    let mut collector = Command::new(env!("CARGO_BIN_EXE_ormah-desktop"))
        .arg("--ormah-internal-log-collector")
        .arg(&log)
        .process_group(0)
        .stdin(Stdio::piped())
        .stdout(Stdio::null())
        .stderr(Stdio::null())
        .spawn()
        .unwrap();
    let input: OwnedFd = collector.stdin.take().unwrap().into();
    let writer = std::fs::File::from(input);
    let stderr = writer.try_clone().unwrap();
    let started = Instant::now();
    let status = Command::new("/bin/sh").args(["-c",
        r#"printf 'cli startup\n'; (sleep 1; printf 'late daemon output\nAuthorization: "Basic dXNlcjpwYXNz"\n' >&2) & exit 0"#])
        .stdout(Stdio::from(writer)).stderr(Stdio::from(stderr)).status().unwrap();
    assert!(status.success());
    assert!(
        started.elapsed() < Duration::from_millis(700),
        "CLI waited for inherited output EOF"
    );
    // No writer remains in this parent. The independent collector must stay
    // alive for the detached child's inherited writer and redact its output.
    assert!(collector.wait().unwrap().success());
    let text = std::fs::read_to_string(&log).unwrap();
    assert!(text.contains("cli startup"));
    assert!(text.contains("late daemon output"));
    assert!(!text.contains("dXNlcjpwYXNz"));
    std::fs::remove_dir_all(dir).unwrap();
}

#[test]
fn concurrent_collectors_share_bounded_rotation() {
    let dir = std::env::temp_dir().join(format!("ormah-collector-{}", uuid::Uuid::new_v4()));
    let log = dir.join("setup.log");
    let workers: Vec<_> = (0..2)
        .map(|_| {
            let log = log.clone();
            std::thread::spawn(move || {
                let mut child = Command::new(env!("CARGO_BIN_EXE_ormah-desktop"))
                    .arg("--ormah-internal-log-collector")
                    .arg(log)
                    .stdin(Stdio::piped())
                    .stdout(Stdio::null())
                    .stderr(Stdio::null())
                    .spawn()
                    .unwrap();
                let mut writer = child.stdin.take().unwrap();
                writer
                    .write_all("download progress line\n".repeat(60000).as_bytes())
                    .unwrap();
                drop(writer);
                assert!(child.wait().unwrap().success());
            })
        })
        .collect();
    for worker in workers {
        worker.join().unwrap();
    }
    for path in [&log, &log.with_extension("log.1")] {
        assert!(std::fs::metadata(path).unwrap().len() <= 1024 * 1024);
    }
    assert_eq!(std::fs::read_dir(&dir).unwrap().count(), 3); // current, rotation, stable lock
    std::fs::remove_dir_all(dir).unwrap();
}
