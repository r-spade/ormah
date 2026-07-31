fn main() {
    // The desktop app pins which ormah Python package it installs. Read the
    // version from the repo's pyproject.toml at compile time so it can never
    // drift from the released package (it was hand-maintained before and went
    // stale more than once). CI builds from the repo, so the file is always
    // present; a plain "version = " line only appears in [project].
    let pyproject = std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("../../pyproject.toml");
    let text = std::fs::read_to_string(&pyproject)
        .unwrap_or_else(|e| panic!("failed to read {}: {e}", pyproject.display()));
    let version = text
        .lines()
        .find_map(|line| line.strip_prefix("version = "))
        .map(|v| v.trim().trim_matches('"'))
        .expect("no `version = \"…\"` line in pyproject.toml");
    assert!(
        version.chars().next().is_some_and(|c| c.is_ascii_digit()),
        "pyproject.toml version doesn't look like a version: {version}"
    );
    println!("cargo:rustc-env=ORMAH_PY_VERSION={version}");
    println!("cargo:rerun-if-changed={}", pyproject.display());

    tauri_build::try_build(
        tauri_build::Attributes::new().app_manifest(tauri_build::AppManifest::new()),
    )
    .expect("failed to build Tauri context")
}
