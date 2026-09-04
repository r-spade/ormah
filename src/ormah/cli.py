"""Unified CLI entry point for ormah.

Usage:
    ormah server start      Start server (foreground)
    ormah server start -d   Start server (daemon via launchd)
    ormah server stop       Stop the server
    ormah server status     Check if running
    ormah setup             One-shot setup (hooks, MCP, server)
    ormah backup create     Create a local memory backup
    ormah backup restore    Restore a local memory backup
    ormah account login     Sign in to Ormah Cloud
    ormah account status    Show cached account entitlement status
    ormah account logout    Revoke this device token and sign out
    ormah claude-md install Install shared Ormah guidance into CLAUDE.md
    ormah pi-md install     Install shared Ormah guidance into Pi AGENTS.md
    ormah uninstall         Remove integrations and local data; preserve cloud recovery files
    ormah mcp               Run MCP stdio server
    ormah recall <query>    Search memories
    ormah remember <text>   Store a memory
    ormah ingest <file>     Ingest a conversation log
    ormah ingest-session <path>  Ingest a Claude Code session
    ormah node <id>         Get a specific memory
    ormah whisper inject    Hook handler: inject context
    ormah whisper store     Hook handler: store memories
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
import sys
from pathlib import Path


def _cmd_eval_whisper_run(args):
    try:
        from eval.whisper.cli import cmd_eval_whisper_run as _cmd
    except ModuleNotFoundError as exc:
        if exc.name not in {"eval", "eval.whisper", "eval.whisper.cli"}:
            raise
        print(
            "The whisper eval harness is not installed in the published Ormah runtime.\n"
            "Use a source checkout or editable dev install to run 'ormah eval whisper run'."
        )
        sys.exit(1)

    _cmd(args)


def _cmd_eval_whisper_mine(args):
    try:
        from eval.whisper.cli import cmd_eval_whisper_mine as _cmd
    except ModuleNotFoundError as exc:
        if exc.name not in {"eval", "eval.whisper", "eval.whisper.cli"}:
            raise
        print(
            "The whisper eval harness is not installed in the published Ormah runtime.\n"
            "Use a source checkout or editable dev install to run 'ormah eval whisper mine'."
        )
        sys.exit(1)

    _cmd(args)


def _cmd_eval_whisper_import_labels(args):
    try:
        from eval.whisper.cli import cmd_eval_whisper_import_labels as _cmd
    except ModuleNotFoundError as exc:
        if exc.name not in {"eval", "eval.whisper", "eval.whisper.cli"}:
            raise
        print(
            "The whisper eval harness is not installed in the published Ormah runtime.\n"
            "Use a source checkout or editable dev install to run 'ormah eval whisper import-labels'."
        )
        sys.exit(1)

    _cmd(args)


def _cmd_eval_recall_run(args):
    try:
        from eval.recall.cli import cmd_eval_recall_run as _cmd
    except ModuleNotFoundError as exc:
        if exc.name not in {"eval", "eval.recall", "eval.recall.cli"}:
            raise
        print(
            "The recall eval harness is not installed in the published Ormah runtime.\n"
            "Use a source checkout or editable dev install to run 'ormah eval recall run'."
        )
        sys.exit(1)

    _cmd(args)


def _cmd_eval_recall_export(args):
    try:
        from eval.recall.cli import cmd_eval_recall_export_for_labeling as _cmd
    except ModuleNotFoundError as exc:
        if exc.name not in {"eval", "eval.recall", "eval.recall.cli"}:
            raise
        sys.exit(1)
    _cmd(args)


def _cmd_eval_recall_import(args):
    try:
        from eval.recall.cli import cmd_eval_recall_import_labels as _cmd
    except ModuleNotFoundError as exc:
        if exc.name not in {"eval", "eval.recall", "eval.recall.cli"}:
            raise
        sys.exit(1)
    _cmd(args)


def _cmd_server_start(args):
    if args.daemon:
        from ormah.console import info, warn
        from ormah.server_manager import get_ormah_bin_path, restart_with_autostart
        from ormah.setup import WRAPPER_PATH, generate_server_wrapper

        ormah_bin = get_ormah_bin_path()
        if not WRAPPER_PATH.exists():
            generate_server_wrapper(ormah_bin)
        if not restart_with_autostart(
            ormah_bin,
            wrapper_path=str(WRAPPER_PATH),
            show_progress=True,
        ):
            warn("Server did not start in time")
            info("Check ~/.local/share/ormah/logs/ormah.log")
            sys.exit(1)
    else:
        import uvicorn
        from ormah.config import settings
        from ormah.console import info, warn
        from ormah.server_manager import is_port_in_use, is_server_running

        # A duplicate Ormah start is a clean no-op. A foreign listener must fail
        # so launchd/systemd keeps retrying instead of treating the job as done.
        if is_port_in_use(settings.host, settings.port):
            if is_server_running():
                info(
                    f"Ormah is already running on {settings.host}:{settings.port}. "
                    "Exiting."
                )
                return
            warn(
                f"Port {settings.port} is in use by another process; "
                "Ormah could not start."
            )
            sys.exit(1)

        uvicorn.run(
            "ormah.main:app",
            host=settings.host,
            port=settings.port,
            reload=args.reload,
        )


def _cmd_server_stop(args):
    import sys
    from ormah.server_manager import _stop_running_server, uninstall_autostart

    result = _stop_running_server()
    uninstall_autostart()
    if result.failed:
        sys.exit(1)


def _cmd_server_status(args):
    from ormah.server_manager import is_server_running
    from ormah.config import settings

    if is_server_running():
        print(f"Server is running on port {settings.port}.")
    else:
        print("Server is not running.")
        sys.exit(1)


def _cmd_setup(args):
    if getattr(args, "json", False):
        from ormah.setup import run_setup_json

        result = run_setup_json()
        print(json.dumps(result, indent=2))
        if result["errors"]:
            sys.exit(1)
        return

    from ormah.setup import run_setup

    run_setup(
        ci=args.ci,
        update=args.update,
        skip_client_setup=args.skip_client_setup,
    )


def _cmd_uninstall(args):
    from ormah.setup import run_uninstall

    run_uninstall(yes=args.yes)


def _cmd_claude_md_install(args):
    from ormah.setup import install_claude_md

    install_claude_md(scope=args.scope, cwd=Path.cwd())


def _cmd_pi_md_install(args):
    from ormah.setup import install_pi_md

    install_pi_md(scope=args.scope, cwd=Path.cwd())


def _format_bytes(size_bytes: int) -> str:
    units = ["B", "KB", "MB", "GB"]
    value = float(size_bytes)
    for unit in units:
        if value < 1024 or unit == units[-1]:
            if unit == "B":
                return f"{int(value)} {unit}"
            return f"{value:.1f} {unit}"
        value /= 1024
    return f"{value:.1f} GB"


def _format_backup_time(value: datetime) -> str:
    if value.tzinfo is None:
        value = value.replace(tzinfo=timezone.utc)
    return value.astimezone().strftime("%Y-%m-%d %H:%M:%S %Z")


def _backup_service():
    from ormah.backup import service_from_settings
    from ormah.config import settings

    return service_from_settings(settings)


def _cloud_client():
    from ormah.cloud.client import client_from_settings
    from ormah.config import settings

    return client_from_settings(settings)


def _print_account_error(message: str) -> None:
    from ormah.console import fail

    fail(message)
    sys.exit(1)


def _format_cache_age(seconds: int | None) -> str:
    if seconds is None:
        return "none"
    if seconds < 60:
        return f"{seconds}s"
    if seconds < 3600:
        return f"{seconds // 60}m"
    if seconds < 86400:
        return f"{seconds // 3600}h"
    return f"{seconds // 86400}d"


def _format_iso_time(value: str | None) -> str:
    if value is None:
        return "never"
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        return value
    return _format_backup_time(parsed)


def _cmd_account_login(args):
    from ormah.cloud.account import (
        AccountError,
        request_login_code,
        verify_login_code,
    )
    from ormah.config import settings
    from ormah.console import info, ok, warn

    email = (args.email or input("Email: ")).strip().lower()
    if not email:
        _print_account_error("Email is required.")

    client = None
    try:
        client = _cloud_client()
        request_login_code(settings, email, client=client)
        info(f"Sign-in code requested for {email}")
        code = (args.code or input("Sign-in code: ")).strip()
        status = verify_login_code(settings, email, code, client=client)
    except (AccountError, OSError) as exc:
        _print_account_error(str(exc))
    finally:
        close = getattr(client, "close", None) if client is not None else None
        if close:
            close()

    ok(f"Signed in as {email}")
    if not status.entitlement_available:
        warn("Entitlement status is unavailable while offline; it will refresh later.")
        return
    print(f"Plan status: {status.plan_status or 'unknown'}")
    print(f"Cloud backup entitlement: {status.entitlement.value}")


def _cmd_account_status(args):
    from ormah.cloud.account import get_account_status
    from ormah.config import settings

    status = get_account_status(settings)
    result = status.to_payload()
    if args.json:
        print(json.dumps(result, indent=2, sort_keys=True))
        return

    print(f"Account: {status.email or 'not signed in'}")
    print(f"Entitlement: {status.entitlement.value}")
    print(f"Plan status: {result['plan_status'] or 'unknown'}")
    print(f"Cache age: {_format_cache_age(status.cache_age_seconds)}")
    print(f"Device: {result['device_name']}")


def _cmd_account_logout(args):
    from ormah.cloud.account import AccountError, logout_account
    from ormah.config import settings
    from ormah.console import info, ok, warn

    if not args.yes:
        if not sys.stdin.isatty():
            _print_account_error("Logout needs confirmation. Re-run with --yes in non-interactive shells.")
        answer = input("Revoke this device token and sign out? (y/N) ").strip().lower()
        if answer not in {"y", "yes"}:
            info("Logout cancelled")
            return

    try:
        result = logout_account(settings, client_factory=_cloud_client)
    except AccountError as exc:
        _print_account_error(str(exc))
    if result.warning:
        warn(result.warning)
    ok("Signed out locally")


def _print_backup_error(message: str) -> None:
    from ormah.console import fail

    fail(message)
    sys.exit(1)


def _backup_to_dict(backup):
    return {
        "name": backup.name,
        "path": str(backup.path),
        "created_at": backup.created_at.isoformat(),
        "node_count": backup.node_count,
        "deleted_count": backup.deleted_count,
        "size_bytes": backup.size_bytes,
    }


def _cmd_backup_create(args):
    from ormah.backup import BackupError
    from ormah.console import ok

    try:
        backup = _backup_service().create(reason="manual")
    except BackupError as exc:
        _print_backup_error(str(exc))

    if args.json:
        print(json.dumps(_backup_to_dict(backup), indent=2, sort_keys=True))
        return

    ok(f"Created backup: {backup.name}")
    print(f"Path: {backup.path}")
    print(f"Nodes: {backup.node_count} active, {backup.deleted_count} deleted")


def _cmd_backup_list(args):
    backups = _backup_service().list()
    if args.json:
        print(json.dumps([_backup_to_dict(backup) for backup in backups], indent=2, sort_keys=True))
        return

    if not backups:
        print("No Ormah memory backups found.")
        return

    for backup in backups:
        print(
            f"{backup.name}  "
            f"{_format_backup_time(backup.created_at)}  "
            f"{backup.node_count} active / {backup.deleted_count} deleted  "
            f"{_format_bytes(backup.size_bytes)}"
        )


def _cmd_backup_status(args):
    from ormah.cloud.state import cloud_status_payload
    from ormah.config import settings

    service = _backup_service()
    latest = service.latest()
    has_memory = service.has_backupable_memory()
    due = has_memory and service.backup_due(interval_hours=settings.backup_interval_hours)
    cloud = cloud_status_payload(settings)
    status = {
        "enabled": settings.backup_enabled,
        "backup_dir": str(settings.backup_dir),
        "interval_hours": settings.backup_interval_hours,
        "retention_count": settings.backup_retention_count,
        "has_backupable_memory": has_memory,
        "due": due,
        "latest": _backup_to_dict(latest) if latest else None,
        "cloud": cloud,
    }

    if args.json:
        print(json.dumps(status, indent=2, sort_keys=True))
        return

    print(f"Automatic backups: {'enabled' if settings.backup_enabled else 'disabled'}")
    print(f"Backup directory: {settings.backup_dir}")
    print(f"Schedule: every {settings.backup_interval_hours}h, keep last {settings.backup_retention_count}")
    if latest is None:
        print("Latest backup: none")
    else:
        print(f"Latest backup: {latest.name} ({_format_backup_time(latest.created_at)})")
    if not has_memory:
        print("Backup due now: no (no memory nodes yet)")
    else:
        print(f"Backup due now: {'yes' if due else 'no'}")

    print("\nCloud backup:")
    print(f"  Automatic uploads: {'enabled' if cloud['enabled'] else 'disabled'}")
    print(f"  Entitlement: {cloud['entitlement']}")
    if cloud["last_upload_snapshot_id"]:
        age = _format_cache_age(cloud["last_upload_age_seconds"])
        print(f"  Last upload: {cloud['last_upload_snapshot_id']} ({age} ago)")
    else:
        print("  Last upload: never")
    if cloud["last_verify_ok"] is True:
        print(f"  Last verified restorable: ✓ {_format_iso_time(cloud['last_verify_at'])}")
    elif cloud["last_verify_ok"] is False:
        reason = f" — {cloud['last_verify_error']}" if cloud["last_verify_error"] else ""
        print(f"  Last verified restorable: ✗ {_format_iso_time(cloud['last_verify_at'])}{reason}")
    else:
        print("  Last verified restorable: not yet verified")
    for warning in cloud["warnings"]:
        print(f"  WARNING: {warning}")


def _cmd_backup_restore(args):
    from ormah.backup import BackupError
    from ormah.cloud.restore import CloudRestoreError, restore_cloud_snapshot
    from ormah.config import settings
    from ormah.console import info, ok, warn
    from ormah.server_manager import is_server_running

    if is_server_running():
        _print_backup_error("Stop the Ormah server before restoring: ormah server stop")

    cloud_requested = args.cloud is not None
    if cloud_requested and args.backup:
        _print_backup_error("Choose either a local backup name or --cloud, not both.")
    if not cloud_requested and not args.backup:
        _print_backup_error("Provide a local backup name or use --cloud [SNAPSHOT_ID].")

    restore_label = (
        f"cloud snapshot {args.cloud}" if args.cloud else "latest committed cloud snapshot"
    ) if cloud_requested else f"backup {args.backup}"

    if not args.yes:
        if not sys.stdin.isatty():
            _print_backup_error("Restore needs confirmation. Re-run with --yes in non-interactive shells.")
        warn("Restore overwrites current memory files.")
        answer = input(f"Restore {restore_label}? (y/N) ").strip().lower()
        if answer not in {"y", "yes"}:
            info("Restore cancelled")
            return

    try:
        if cloud_requested:
            cloud_result = restore_cloud_snapshot(
                settings,
                args.cloud or None,
                rebuild_index=not args.no_rebuild,
            )
            result = cloud_result.restore
        else:
            result = _backup_service().restore(args.backup, rebuild_index=not args.no_rebuild)
    except (BackupError, CloudRestoreError) as exc:
        _print_backup_error(str(exc))

    ok(f"Restored backup: {result.restored.name}")
    if result.safety_backup is not None:
        info(f"Safety backup of previous memory: {result.safety_backup.name}")
    if result.rebuilt_nodes is not None:
        info(f"Rebuilt search index from {result.rebuilt_nodes} nodes")


def _cmd_cloud_init(args):
    from ormah.cloud.crypto import CloudCryptoError
    from ormah.cloud.keys import (
        KEY_PATH,
        MAX_RECOVERY_KIT_BYTES,
        CloudKeyError,
        apply_recovery_import,
        get_or_create_store_id,
        init_key,
        load_identity_strings,
        plan_recovery_import,
        validate_recovery_kit_destination,
        write_recovery_kit,
    )
    from ormah.config import settings
    from ormah.console import info, ok, warn

    try:
        import_result = None
        kit_path = None
        if args.import_key:
            import_source = args.import_key
            if import_source == "-":
                reader = getattr(sys.stdin, "buffer", sys.stdin)
                raw = reader.read(MAX_RECOVERY_KIT_BYTES + 1)
                raw_bytes = raw.encode("utf-8") if isinstance(raw, str) else raw
                if len(raw_bytes) > MAX_RECOVERY_KIT_BYTES:
                    raise CloudKeyError("The recovery kit exceeds the safe import limit.")
                try:
                    import_source = raw_bytes.decode("utf-8")
                except UnicodeDecodeError as exc:
                    raise CloudKeyError("The recovery kit is not valid UTF-8 text.") from exc
            # Preflight both resources before the first write. A preserved key
            # may contain extra pre-rotation identities, but it must contain
            # every identity carried by the recovery kit.
            kit_path = validate_recovery_kit_destination()
            import_result = apply_recovery_import(
                plan_recovery_import(import_source, settings.memory_dir)
            )
        else:
            init_key()
    except (CloudKeyError, CloudCryptoError, OSError) as exc:
        _print_backup_error(str(exc))

    try:
        store_id = get_or_create_store_id(settings.memory_dir)
        # The installed keyring is authoritative. Always regenerate the kit so
        # an unrelated, stale, or damaged file is never affirmed as recovery.
        kit_path = write_recovery_kit(
            store_id,
            kit_path=kit_path,
            account_email=getattr(settings, "account_email", None),
        )
        identity_count = len(load_identity_strings())
    except (CloudKeyError, CloudCryptoError, OSError) as exc:
        if import_result is not None:
            _print_backup_error(
                "Recovery identity was installed or verified, but the canonical recovery kit "
                f"could not be refreshed: {exc}\nComplete it with: ormah cloud kit"
            )
        else:
            _print_backup_error(
                f"Encryption key was written to {KEY_PATH}, but finishing setup failed: {exc}\n"
                "Complete it with: ormah cloud kit"
            )

    if args.json:
        print(json.dumps({
            "key_path": str(KEY_PATH),
            "identity_count": identity_count,
            "imported": bool(args.import_key),
            "store_id": store_id,
            "recovery_kit": str(kit_path),
            "store_id_status": (
                import_result.store_id_status if import_result is not None else "generated"
            ),
            "key_status": (
                import_result.key_status if import_result is not None else "generated"
            ),
        }, indent=2, sort_keys=True))
        return

    if args.import_key:
        if import_result.store_id_status == "installed":
            ok(f"Store ID installed: {store_id}")
        elif import_result.store_id_status == "already_matched":
            ok(f"Store ID already matched: {store_id}")
        elif import_result.store_id_status == "already_present":
            ok(f"Store ID already present: {store_id}")
        else:
            ok(f"Store ID generated: {store_id}")
        if import_result.key_status == "installed":
            ok(
                f"Recovery key installed: {KEY_PATH} "
                f"({identity_count} identit{'y' if identity_count == 1 else 'ies'})"
            )
        else:
            ok(
                f"Recovery key already matched and preserved: {KEY_PATH} "
                f"({identity_count} identit{'y' if identity_count == 1 else 'ies'})"
            )
    else:
        ok(f"Generated encryption key: {KEY_PATH}")
        info(f"Store id: {store_id}")
    ok(f"Recovery kit written: {kit_path}")
    warn(
        "Store the recovery kit somewhere safe and OFFLINE. Anyone with it can "
        "read your backups; without it, nobody can — including us."
    )


def _cmd_cloud_kit(args):
    """Regenerate the recovery kit from the existing key file (idempotent)."""
    from ormah.cloud.crypto import CloudCryptoError
    from ormah.cloud.keys import (
        CloudKeyError,
        get_or_create_store_id,
        load_identity_strings,
        write_recovery_kit,
    )
    from ormah.config import settings
    from ormah.console import ok, warn

    try:
        store_id = get_or_create_store_id(settings.memory_dir)
        kit_path = write_recovery_kit(
            store_id,
            account_email=getattr(settings, "account_email", None),
        )
        identity_count = len(load_identity_strings())
    except (CloudKeyError, CloudCryptoError, OSError) as exc:
        _print_backup_error(str(exc))

    if args.json:
        print(json.dumps({
            "identity_count": identity_count,
            "recovery_kit": str(kit_path),
            "store_id": store_id,
        }, indent=2, sort_keys=True))
        return

    ok(f"Recovery kit regenerated: {kit_path} ({identity_count} identities)")
    warn("Replace any stored copies of the old kit with this one.")


def _cmd_cloud_rotate_key(args):
    from ormah.cloud.crypto import CloudCryptoError
    from ormah.cloud.keys import (
        CloudKeyError,
        load_identity_strings,
    )
    from ormah.cloud.recovery import RecoveryKitError, RecoveryKitService
    from ormah.cloud.state import CloudStateError
    from ormah.cloud.store_lock import StoreLockTimeout
    from ormah.config import settings
    from ormah.console import info, ok, warn

    if not args.yes:
        if not sys.stdin.isatty():
            _print_backup_error("Key rotation needs confirmation. Re-run with --yes in non-interactive shells.")
        warn("Rotation generates a new encryption key. Old keys are retained so existing backups stay readable.")
        answer = input("Rotate the cloud encryption key? (y/N) ").strip().lower()
        if answer not in {"y", "yes"}:
            info("Rotation cancelled")
            return

    try:
        kit_path = RecoveryKitService(settings).rotate_current_key(
            account_email=getattr(settings, "account_email", None),
        )
        identity_count = len(load_identity_strings())
    except (
        RecoveryKitError,
        CloudKeyError,
        CloudCryptoError,
        CloudStateError,
        StoreLockTimeout,
        OSError,
    ) as exc:
        _print_backup_error(
            f"Key rotation could not be completed safely: {exc}\n"
            "Check `ormah cloud kit` before relying on a previously saved recovery kit."
        )

    if args.json:
        print(json.dumps({
            "identity_count": identity_count,
            "recovery_kit": str(kit_path),
            "rotated": True,
        }, indent=2, sort_keys=True))
        return

    ok("Rotated encryption key; new bundles use the new key.")
    info(f"Identities on file: {identity_count} (all retained — older backups still decrypt)")
    ok(f"Recovery kit regenerated: {kit_path}")
    warn("Replace any stored copies of the old recovery kit with the new one.")


def _cmd_mcp(args):
    from ormah.adapters.mcp_adapter import main as mcp_main

    mcp_main()


def main():
    from importlib.metadata import version as pkg_version

    p = argparse.ArgumentParser(
        prog="ormah",
        description="Local-first memory system for AI agents.",
    )
    p.add_argument(
        "--version", action="version",
        version=f"ormah {pkg_version('ormah')}",
    )
    sub = p.add_subparsers(dest="cmd")

    # --- server ---
    server_p = sub.add_parser("server", help="Manage the ormah server")
    server_sub = server_p.add_subparsers(dest="server_cmd", required=True)

    start_p = server_sub.add_parser("start", help="Start the server")
    start_p.add_argument(
        "-d", "--daemon", action="store_true",
        help="Run as daemon (launchd on macOS)",
    )
    start_p.add_argument(
        "--reload", action="store_true",
        help="Enable auto-reload on file changes (dev mode)",
    )
    start_p.set_defaults(func=_cmd_server_start)

    stop_p = server_sub.add_parser("stop", help="Stop the server")
    stop_p.set_defaults(func=_cmd_server_stop)

    status_p = server_sub.add_parser("status", help="Check server status")
    status_p.set_defaults(func=_cmd_server_status)

    # --- backup ---
    backup_p = sub.add_parser("backup", help="Back up and restore Ormah memories")
    backup_sub = backup_p.add_subparsers(dest="backup_cmd", required=True)

    backup_create = backup_sub.add_parser("create", help="Create a timestamped local backup")
    backup_create.add_argument("--json", action="store_true", help="Output backup metadata as JSON")
    backup_create.set_defaults(func=_cmd_backup_create)

    backup_list = backup_sub.add_parser("list", help="List local backups")
    backup_list.add_argument("--json", action="store_true", help="Output backups as JSON")
    backup_list.set_defaults(func=_cmd_backup_list)

    backup_status = backup_sub.add_parser("status", help="Show backup configuration and freshness")
    backup_status.add_argument("--json", action="store_true", help="Output backup status as JSON")
    backup_status.set_defaults(func=_cmd_backup_status)

    backup_restore = backup_sub.add_parser("restore", help="Restore memory files from a backup")
    backup_restore.add_argument("backup", nargs="?", help="Backup name from `ormah backup list`")
    backup_restore.add_argument(
        "--cloud",
        nargs="?",
        const="",
        metavar="SNAPSHOT_ID",
        help="Restore the latest committed cloud snapshot, or the specified snapshot",
    )
    backup_restore.add_argument("--yes", action="store_true", help="Skip interactive restore confirmation")
    backup_restore.add_argument("--no-rebuild", action="store_true", help="Skip search index rebuild")
    backup_restore.set_defaults(func=_cmd_backup_restore)

    # --- account ---
    account_p = sub.add_parser("account", help="Manage the Ormah Cloud account")
    account_sub = account_p.add_subparsers(dest="account_cmd", required=True)

    account_login = account_sub.add_parser("login", help="Sign in with an emailed one-time code")
    account_login.add_argument("--email", help="Account email (otherwise prompted)")
    account_login.add_argument("--code", help="One-time code (otherwise prompted)")
    account_login.set_defaults(func=_cmd_account_login)

    account_status = account_sub.add_parser("status", help="Show account and entitlement status")
    account_status.add_argument("--json", action="store_true", help="Output status as JSON")
    account_status.set_defaults(func=_cmd_account_status)

    account_logout = account_sub.add_parser("logout", help="Revoke this device token and sign out")
    account_logout.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    account_logout.set_defaults(func=_cmd_account_logout)

    # --- cloud ---
    cloud_p = sub.add_parser("cloud", help="Encrypted cloud backup keys and store identity")
    cloud_sub = cloud_p.add_subparsers(dest="cloud_cmd", required=True)

    cloud_init = cloud_sub.add_parser("init", help="Generate encryption key, store id, and recovery kit")
    cloud_init.add_argument("--json", action="store_true", help="Output result as JSON")
    cloud_init.add_argument(
        "--import-key", metavar="PATH",
        help="Install identities from a recovery kit/key file, or '-' for standard input",
    )
    cloud_init.set_defaults(func=_cmd_cloud_init)

    cloud_rotate = cloud_sub.add_parser("rotate-key", help="Rotate the encryption key (old keys are retained)")
    cloud_rotate.add_argument("--yes", action="store_true", help="Skip interactive confirmation")
    cloud_rotate.add_argument("--json", action="store_true", help="Output result as JSON")
    cloud_rotate.set_defaults(func=_cmd_cloud_rotate_key)

    cloud_kit = cloud_sub.add_parser("kit", help="Regenerate the recovery kit from the existing key")
    cloud_kit.add_argument("--json", action="store_true", help="Output result as JSON")
    cloud_kit.set_defaults(func=_cmd_cloud_kit)

    # --- setup ---
    setup_p = sub.add_parser("setup", help="One-shot setup (hooks, MCP, server)")
    setup_p.add_argument("--ci", action="store_true", help="Non-interactive mode for CI/testing")
    setup_p.add_argument("--update", action="store_true", help="Skip interactive questions, just reapply hooks and MCP config")
    setup_p.add_argument(
        "--skip-client-setup",
        action="store_true",
        help="Skip Claude/Codex/Desktop integration wiring; useful when a plugin manages the client side",
    )
    setup_p.add_argument(
        "--json",
        action="store_true",
        help="Non-interactive agent wiring; print a JSON result (used by the Mac app one-click setup)",
    )
    setup_p.set_defaults(func=_cmd_setup)

    # --- uninstall ---
    uninstall_p = sub.add_parser(
        "uninstall",
        help="Remove integrations and local data; preserve cloud recovery files",
    )
    uninstall_p.add_argument("-y", "--yes", action="store_true", help="Skip confirmation prompts")
    uninstall_p.set_defaults(func=_cmd_uninstall)

    # --- claude-md ---
    claude_md_p = sub.add_parser("claude-md", help="Manage Ormah guidance in Claude Code CLAUDE.md")
    claude_md_sub = claude_md_p.add_subparsers(dest="claude_md_cmd", required=True)

    claude_md_install = claude_md_sub.add_parser(
        "install",
        help="Install the Ormah guidance block into a Claude Code CLAUDE.md file",
    )
    claude_md_install.add_argument(
        "--scope",
        choices=["auto", "project", "user", "local"],
        default="auto",
        help="Install into the CLAUDE.md target that matches plugin scope, or override it explicitly",
    )
    claude_md_install.set_defaults(func=_cmd_claude_md_install)

    # --- pi-md ---
    pi_md_p = sub.add_parser("pi-md", help="Manage Ormah guidance in the Pi agent AGENTS.md")
    pi_md_sub = pi_md_p.add_subparsers(dest="pi_md_cmd", required=True)

    pi_md_install = pi_md_sub.add_parser(
        "install",
        help="Install the Ormah guidance block into a Pi AGENTS.md file",
    )
    pi_md_install.add_argument(
        "--scope",
        choices=["user", "project"],
        default="user",
        help="user = ~/.pi/agent/AGENTS.md (default), project = ./AGENTS.md",
    )
    pi_md_install.set_defaults(func=_cmd_pi_md_install)

    # --- mcp ---
    mcp_p = sub.add_parser("mcp", help="Run MCP stdio server")
    mcp_p.set_defaults(func=_cmd_mcp)

    # --- CLI commands (delegated to cli_adapter) ---
    from ormah.adapters.cli_adapter import (
        cmd_ingest,
        cmd_ingest_session,
        cmd_node,
        cmd_outdated,
        cmd_recall,
        cmd_remember,
        cmd_stats,
        cmd_whisper_inject,
        cmd_whisper_store,
        cmd_whisper_setup,
    )

    # recall
    rec = sub.add_parser("recall", help="Search memories")
    rec.add_argument("query", help="Search query")
    rec.add_argument("--limit", type=int, help="Max results")
    rec.add_argument("--types", help="Comma-separated memory types to filter")
    rec.add_argument("--json", action="store_true", help="Output raw JSON")
    rec.add_argument("--space", help="Override space detection")
    rec.set_defaults(func=cmd_recall)

    # remember
    rem = sub.add_parser("remember", help="Store a memory")
    rem.add_argument("content", help="Memory content (use - for stdin)")
    rem.add_argument("--title", help="Short title")
    rem.add_argument("--type", default="fact", help="Memory type (default: fact)")
    rem.add_argument(
        "--tier", default="working",
        help="Tier: core/working/archival (default: working)",
    )
    rem.add_argument("--tags", help="Comma-separated tags")
    rem.add_argument("--about-self", action="store_true", help="Link to user identity")
    rem.add_argument("--space", help="Override space detection")
    rem.set_defaults(func=cmd_remember)

    # ingest
    ing = sub.add_parser("ingest", help="Ingest a conversation log")
    ing.add_argument("source", help="File path or - for stdin")
    ing.add_argument("--space", help="Override space detection")
    ing.set_defaults(func=cmd_ingest)

    # ingest-session
    ings = sub.add_parser(
        "ingest-session", help="Ingest a Claude Code JSONL session transcript",
    )
    ings.add_argument("path", help="Path to Claude Code JSONL transcript")
    ings.add_argument(
        "--dry-run", action="store_true",
        help="Extract but don't store",
    )
    ings.add_argument("--space", help="Override project space")
    ings.add_argument(
        "--min-turns", type=int, default=5,
        help="Minimum user turns (default: 5)",
    )
    ings.set_defaults(func=cmd_ingest_session)

    # node
    nd = sub.add_parser("node", help="Get a specific memory by ID")
    nd.add_argument("id", help="Memory UUID")
    nd.add_argument("--json", action="store_true", help="Output raw JSON")
    nd.set_defaults(func=cmd_node)

    # outdated
    out = sub.add_parser("outdated", help="Mark a memory as outdated")
    out.add_argument("id", help="Memory UUID")
    out.add_argument("--reason", help="Reason for marking outdated")
    out.set_defaults(func=cmd_outdated)

    # stats
    sts = sub.add_parser("stats", help="Show memory store statistics")
    sts.add_argument("--json", action="store_true", help="Output raw JSON")
    sts.set_defaults(func=cmd_stats)

    # whisper
    wh = sub.add_parser("whisper", help="Claude Code hook integration")
    wh_sub = wh.add_subparsers(dest="whisper_cmd", required=True)

    wh_inject = wh_sub.add_parser("inject", help="(internal) UserPromptSubmit hook — called automatically by Claude Code")
    wh_inject.set_defaults(func=cmd_whisper_inject)

    wh_store = wh_sub.add_parser("store", help="(internal) PreCompact/SessionEnd hook — called automatically by Claude Code")
    wh_store.set_defaults(func=cmd_whisper_store)

    wh_setup = wh_sub.add_parser("setup", help="Write Claude Code hook config to settings.json")
    wh_setup.add_argument(
        "--global", dest="glob", action="store_true",
        help="Write to global ~/.claude/settings.json",
    )
    wh_setup.set_defaults(func=cmd_whisper_setup)

    # eval
    ev = sub.add_parser("eval", help="Run development evaluation harnesses")
    ev_sub = ev.add_subparsers(dest="eval_cmd", required=True)

    ev_wh = ev_sub.add_parser("whisper", help="Evaluate whisper pipeline quality")
    ev_wh_sub = ev_wh.add_subparsers(dest="eval_whisper_cmd", required=True)

    ev_wh_run = ev_wh_sub.add_parser("run", help="Run whisper eval against golden corpus")
    ev_wh_run.add_argument("--category", help="Filter by category (e.g. preference, factual)")
    ev_wh_run.add_argument("--show-failures", action="store_true", dest="show_failures",
                           help="Print failure details")
    ev_wh_run.add_argument(
        "--simulate-session",
        action="store_true",
        help="Force session simulation for all prompts in the corpus",
    )
    ev_wh_run.add_argument(
        "--preserve-self",
        action="store_true",
        help="Preserve the self node when reseeding eval cases",
    )
    ev_wh_run.add_argument("--json", action="store_true", help="Output as JSON")
    ev_wh_run.add_argument("--corpus", default=None,
                           help="Corpus file (absolute, or relative to eval/whisper/corpus/)")
    ev_wh_run.add_argument("--fail-below", default=None, dest="fail_below",
                           help="Fail if metrics below threshold, e.g. f1=0.65,suppression=0.90")
    ev_wh_run.add_argument("--include-provisional", action="store_true", dest="include_provisional",
                           help="Include unreviewed mined (provisional) cases in the run (smoke only)")
    ev_wh_run.set_defaults(func=_cmd_eval_whisper_run)

    ev_wh_mine = ev_wh_sub.add_parser("mine", help="Mine provisional eval cases from the live whisper_log (read-only)")
    ev_wh_mine.add_argument("--db", default=None, help="Path to live index.db (default: from Settings)")
    ev_wh_mine.add_argument("--limit", type=int, default=80, help="Max cases to mine (default: 80)")
    ev_wh_mine.add_argument("--out", default=None, help="Output file (default: eval/whisper/corpus/local/mined.jsonl)")
    ev_wh_mine.set_defaults(func=_cmd_eval_whisper_mine)

    ev_wh_import = ev_wh_sub.add_parser("import-labels", help="Confirm reviewed mined labels (clears provisional flags)")
    ev_wh_import.add_argument("--mined", default=None, help="Mined corpus file (default: eval/whisper/corpus/local/mined.jsonl)")
    ev_wh_import.set_defaults(func=_cmd_eval_whisper_import_labels)

    ev_rc = ev_sub.add_parser("recall", help="Evaluate recall/retrieval quality")
    ev_rc_sub = ev_rc.add_subparsers(dest="eval_recall_cmd", required=True)

    ev_rc_run = ev_rc_sub.add_parser("run", help="Run recall eval against golden corpus")
    ev_rc_run.add_argument("--corpus", default="golden", choices=["golden", "synthetic", "all"],
                           help="Corpus to evaluate (default: golden)")
    ev_rc_run.add_argument("--k", type=int, default=8, help="Top-k for metrics (default: 8)")
    ev_rc_run.add_argument("--fail-below", default=None,
                           help="Fail if metrics below threshold, e.g. recall@8=0.70")
    ev_rc_run.add_argument("--fail-on-regression", default=None,
                           help="Fail if metric drops more than delta vs last run, e.g. delta=0.05")
    ev_rc_run.set_defaults(func=_cmd_eval_recall_run)

    ev_rc_export = ev_rc_sub.add_parser("export-for-labeling", help="Export unlabeled pairs to pending_labels.jsonl")
    ev_rc_export.set_defaults(func=_cmd_eval_recall_export)

    ev_rc_import = ev_rc_sub.add_parser("import-labels", help="Merge labels.jsonl into corpus ground truth")
    ev_rc_import.set_defaults(func=_cmd_eval_recall_import)

    args = p.parse_args()

    if not args.cmd:
        p.print_help()
        sys.exit(1)

    args.func(args)


if __name__ == "__main__":
    main()
