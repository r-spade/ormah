import { invoke } from "@tauri-apps/api/core";

export type EntitlementState = "active" | "grace" | "expired" | "none";

export type ProtectionState =
  | "local_only"
  | "sign_in_required"
  | "subscription_required"
  | "initializing"
  | "uploading_first_backup"
  | "verifying_first_backup"
  | "verification_pending"
  | "protected"
  | "changes_pending"
  | "offline"
  | "paused"
  | "stopped"
  | "attention_required";

export type OperationPhase =
  | "pending"
  | "running"
  | "preparing"
  | "discovering"
  | "encrypting"
  | "uploading"
  | "finalizing"
  | "downloading"
  | "decrypting"
  | "checking"
  | "verifying"
  | "rebuilding"
  | "ready"
  | "safety_backup"
  | "restoring"
  | "reloading"
  | "completed"
  | "failed"
  | "canceled";

const ACTIVE_OPERATION_PHASES = new Set<OperationPhase>([
  "pending",
  "running",
  "preparing",
  "discovering",
  "encrypting",
  "uploading",
  "finalizing",
  "downloading",
  "decrypting",
  "checking",
  "verifying",
  "rebuilding",
  "safety_backup",
  "restoring",
  "reloading",
]);

export function operationPhaseIsActive(phase: OperationPhase | null | undefined): boolean {
  return phase !== null && phase !== undefined && ACTIVE_OPERATION_PHASES.has(phase);
}

export interface BridgeInfo {
  version: number;
  platform: string;
}

export interface AccountStatus {
  signed_in: boolean;
  email: string | null;
  device_name: string;
  entitlement: EntitlementState;
  plan_status: string | null;
  cache_age_seconds: number | null;
  warning?: string | null;
}

export interface BillingOffer {
  name: string;
  unit_amount: number;
  currency: string;
  interval: string;
}

export interface ProtectionStatus {
  enabled: boolean;
  store_id: string | null;
  entitlement: EntitlementState;
  protection_state: ProtectionState;
  protection_intent_id: string | null;
  protection_intent_status: string | null;
  protection_intent_expires_at: string | null;
  last_operation_id: string | null;
  last_operation_kind: string | null;
  last_operation_phase: OperationPhase | null;
  last_successful_upload_at: string | null;
  last_successful_backup_snapshot_id: string | null;
  last_successful_verify_at: string | null;
  last_verified_snapshot_id: string | null;
  recovery_kit_verified_at: string | null;
  device_loss_recovery_ready: boolean;
  last_error_code: string | null;
  last_error_message: string | null;
  warnings: string[];
}

export type RecoveryKitActionResult =
  | {
    status: "saved";
    device_loss_recovery_ready: boolean;
    recovery_kit_verified_at: string;
  }
  | {
    status: "canceled";
    device_loss_recovery_ready: null;
    recovery_kit_verified_at: null;
  };

export function recoveryKitSectionVisible(
  status: ProtectionStatus | null | undefined,
): boolean {
  return Boolean(status?.enabled);
}

export interface ProtectionOperation {
  operation_id: string;
  kind: "enable" | "disable" | "backup" | "verify" | "restore";
  status?: "queued" | "running" | "completed" | "failed";
  submitted_at?: string;
  started_at?: string | null;
  finished_at?: string | null;
  phase: OperationPhase | null;
  protection_state: ProtectionState | null;
  reason_code: string | null;
  message: string | null;
  snapshot_id: string | null;
  protection_intent_id: string | null;
  verified_node_count?: number | null;
  snapshot_created_at?: string | null;
  skipped_newer_snapshots?: number;
  safety_backup_name?: string | null;
}

export function protectionCompletionSummary(
  operation: ProtectionOperation | null | undefined,
): string | null {
  if (
    operation?.phase !== "completed"
    || !Number.isInteger(operation.verified_node_count)
    || (operation.verified_node_count ?? 0) < 1
    || !operation.started_at
    || !operation.finished_at
  ) return null;

  const started = new Date(operation.started_at).getTime();
  const finished = new Date(operation.finished_at).getTime();
  if (!Number.isFinite(started) || !Number.isFinite(finished) || finished < started) return null;

  const count = operation.verified_node_count as number;
  const memories = `${new Intl.NumberFormat().format(count)} active ${
    count === 1 ? "memory" : "memories"
  }`;
  const seconds = Math.max(1, Math.round((finished - started) / 1000));
  const elapsed = `${seconds} ${seconds === 1 ? "second" : "seconds"}`;
  if (operation.kind === "enable" || operation.kind === "backup") {
    return `${memories} encrypted, uploaded, and restore-tested in ${elapsed}.`;
  }
  if (operation.kind === "verify") {
    return `${memories} restore-tested in ${elapsed}.`;
  }
  return null;
}

export function effectiveProtectionState(
  operation: ProtectionOperation | null | undefined,
  status: ProtectionStatus | null | undefined,
): ProtectionState {
  const operationActive = operation?.status === "queued" || operation?.status === "running";
  return (operationActive ? operation.protection_state : null)
    || status?.protection_state
    || "local_only";
}

export interface BillingHandoff {
  status: "checkout_required" | "already_subscribed" | "subscription_pending";
  expires_at: number | null;
  opened: boolean;
}

export interface LogoutResult {
  signed_in: false;
  revoked_remotely: boolean;
  warning: string | null;
}

export function isDesktopApp(): boolean {
  if (typeof window === "undefined") return false;
  const marker = window as unknown as {
    __ORMAH_DESKTOP__?: boolean;
    __TAURI_INTERNALS__?: unknown;
  };
  return Boolean(marker.__ORMAH_DESKTOP__ || marker.__TAURI_INTERNALS__);
}

async function native<T>(command: string, args?: Record<string, unknown>): Promise<T> {
  if (!isDesktopApp()) {
    throw new Error("Cloud protection is available in Ormah Desktop.");
  }
  return invoke<T>(command, args);
}

export const productBridge = {
  info: () => native<BridgeInfo>("desktop_bridge_info"),
  accountStatus: () => native<AccountStatus>("account_status"),
  requestCode: (email: string) => native<{ status: string }>("request_account_code", { email }),
  verifyCode: (email: string, code: string) =>
    native<AccountStatus>("verify_account_code", { email, code }),
  logout: () => native<LogoutResult>("logout_account"),
  offer: () => native<BillingOffer>("billing_offer"),
  status: () => native<ProtectionStatus>("protection_status"),
  remoteSnapshot: () => native<RemoteSnapshot>("remote_snapshot"),
  createIntent: () => native<ProtectionOperation>("create_protection_intent"),
  bindIntent: (intentId: string) =>
    native<ProtectionOperation>("bind_protection_intent", { intentId }),
  cancelIntent: (intentId: string) =>
    native<ProtectionOperation>("cancel_protection_intent", { intentId }),
  enable: (intentId: string) =>
    native<ProtectionOperation>("enable_protection", { intentId }),
  disable: () => native<ProtectionOperation>("disable_protection"),
  backupNow: () => native<ProtectionOperation>("backup_now"),
  verifyNow: () => native<ProtectionOperation>("verify_now"),
  operation: (operationId: string) =>
    native<ProtectionOperation>("operation_status", { operationId }),
  prepareRestore: () => native<ProtectionOperation>("prepare_restore"),
  confirmRestore: (preparationOperationId: string) =>
    native<ProtectionOperation>("confirm_restore", { preparationOperationId }),
  cancelRestore: (preparationOperationId: string) =>
    native<{ status: "discarded" }>("cancel_restore", { preparationOperationId }),
  importRecoveryKit: () =>
    native<{ status: "imported" | "canceled" }>("import_recovery_kit"),
  openCheckout: (intentId: string) =>
    native<BillingHandoff>("open_checkout", { intentId }),
  openPortal: () => native<{ opened: boolean }>("open_billing_portal"),
  saveRecoveryKit: () => native<RecoveryKitActionResult>("save_recovery_kit"),
  openPrintableRecoveryKit: () =>
    native<{ opened: boolean }>("open_printable_recovery_kit"),
};

export interface ProtectionPresentation {
  tone: "neutral" | "working" | "success" | "warning" | "danger";
  title: string;
  detail: string;
  action: "protect" | "signin" | "subscribe" | "wait" | "backup" | "retry" | "resume";
}

export type ProtectionRepairAction =
  | "backup"
  | "verify"
  | "resume"
  | "signin"
  | "subscribe"
  | "refresh"
  | "none";

const VERIFY_FAILURES = new Set([
  "verification_failed",
  "download_failed",
  "ciphertext_hash_unavailable",
  "ciphertext_hash_mismatch",
  "manifest_verification_failed",
  "node_parse_failed",
  "bundle_corrupt",
  "index_environment_unavailable",
  "index_rebuild_failed",
  "search_probe_failed",
  "self_pointer_invalid",
]);

const BACKUP_FAILURES = new Set([
  "upload_failed",
  "upload_status_unknown",
  "operation_interrupted",
  "store_busy",
]);

const PROTECTION_RECONNECT_DELAYS_MS = [5_000, 15_000, 30_000, 60_000] as const;

export function protectionReconnectDelay(attempt: number): number {
  const index = Number.isFinite(attempt) ? Math.max(0, Math.floor(attempt)) : 0;
  return PROTECTION_RECONNECT_DELAYS_MS[
    Math.min(index, PROTECTION_RECONNECT_DELAYS_MS.length - 1)
  ];
}

export function protectionRepairAction(status: ProtectionStatus): ProtectionRepairAction {
  if (status.protection_state === "verification_pending") return "verify";
  const reason = status.last_error_code;
  if (reason === "sign_in_required") return "signin";
  if (reason === "subscription_required" || reason === "entitlement_expired") {
    return "subscribe";
  }
  if (VERIFY_FAILURES.has(reason || "")) return "verify";
  if (BACKUP_FAILURES.has(reason || "")) {
    return status.protection_intent_status === "running" ? "resume" : "backup";
  }
  if (status.protection_state === "offline") {
    if (status.protection_intent_status === "running") return "resume";
    return status.enabled ? "backup" : "refresh";
  }
  // Key, quota, disk, and version failures need a dedicated repair path;
  // retrying verification would be misleading and cannot repair them.
  return "none";
}

/** The newest committed cloud backup, which may come from another device. */
export interface RemoteSnapshot {
  snapshot_id: string | null;
  created_at: string | null;
  size_bytes: number | null;
  from_this_device: boolean;
  /** Only this device's own checks count; it cannot vouch for another's. */
  restore_tested_here: boolean;
  error: string | null;
}

// Two intervals of the weekly restore-verification job. Inside that window an
// unchecked upload is the normal resting state, not a problem, so it must not
// paint the panel with a warning.
const VERIFICATION_OVERDUE_AFTER_MS = 14 * 24 * 60 * 60 * 1000;

export function verificationIsOverdue(
  status: ProtectionStatus | null | undefined,
  now: number = Date.now(),
): boolean {
  if (!status?.last_successful_verify_at) return true;
  const verifiedAt = new Date(status.last_successful_verify_at).getTime();
  if (!Number.isFinite(verifiedAt)) return true;
  return now - verifiedAt > VERIFICATION_OVERDUE_AFTER_MS;
}

export type ProtectionActionKind =
  | "signin"
  | "protect"
  | "backup"
  | "restore"
  | "verify"
  | "repair"
  | "subscribe";

export interface ProtectionAction {
  kind: ProtectionActionKind;
  label: string;
  variant: "primary" | "secondary";
  disabled: boolean;
  /** Why the action is demoted or unavailable. Never null when disabled. */
  reason: string | null;
}

const BACKUP_LABEL = "Back up now";
const RESTORE_LABEL = "Restore newest backup";

const REPAIR_LABELS: Record<Exclude<ProtectionRepairAction, "none">, string> = {
  verify: "Retry recovery check",
  signin: "Sign in again",
  subscribe: "Reactivate subscription",
  refresh: "Check connection",
  resume: "Resume protection",
  backup: "Retry encrypted backup",
};

function blockedBackup(reason: string): ProtectionAction {
  return {
    kind: "backup",
    label: BACKUP_LABEL,
    variant: "secondary",
    disabled: true,
    reason,
  };
}

/**
 * Push and pull, always offered together in a fixed order.
 *
 * Ormah deliberately does not guess which side holds newer memory. The only
 * signal the cloud carries is who *uploaded* the newest snapshot, and a restore
 * records nothing, so a device that has just pulled that very snapshot still
 * looks behind. Weighting the buttons on that signal told a machine it was out
 * of date immediately after it had caught up. The panel states each side's facts
 * instead and lets the person decide which way to move.
 *
 * The one direction Ormah does know is local: unbacked-up changes on this
 * device. That comes from local state, not from the cloud listing, so it is
 * safe to lead with backup.
 */
function pushPullActions(
  status: ProtectionStatus | null | undefined,
): ProtectionAction[] {
  const unsavedChanges = status?.protection_state === "changes_pending";
  const backup: ProtectionAction = {
    kind: "backup",
    label: BACKUP_LABEL,
    variant: unsavedChanges ? "primary" : "secondary",
    disabled: false,
    reason: unsavedChanges
      ? "Uploads the changes this device has made since its last backup."
      : "Ormah backs up on a schedule; this captures changes since then straight away.",
  };
  const restore: ProtectionAction = {
    kind: "restore",
    label: RESTORE_LABEL,
    variant: "secondary",
    disabled: false,
    reason: "Replaces this device's memory. Ormah checks the backup and shows"
      + " what is inside before anything changes, and saves a local safety copy first.",
  };
  return [backup, restore];
}

/**
 * The summary-view actions for one protection state.
 *
 * Backup is the product's core capability, so it never vanishes from a signed-in
 * summary without a reason attached. States that cannot accept a new upload
 * return it disabled and explained rather than omitted. Restore appears
 * whenever the cloud actually holds something to restore — including on a
 * machine that has never been protected itself.
 */
export function protectionActions(
  signedIn: boolean,
  state: ProtectionState,
  status: ProtectionStatus | null | undefined,
  remote?: RemoteSnapshot | null,
): ProtectionAction[] {
  if (!signedIn) {
    return [{
      kind: "signin",
      label: "Sign in to Ormah Cloud",
      variant: "primary",
      disabled: false,
      reason: null,
    }];
  }

  const repair = status ? protectionRepairAction(status) : "none";
  const repairButton: ProtectionAction[] = repair === "none" ? [] : [{
    kind: "repair",
    label: REPAIR_LABELS[repair],
    variant: "primary",
    disabled: false,
    reason: null,
  }];
  const restorable = Boolean(remote?.snapshot_id);
  const restoreOnly: ProtectionAction[] = restorable
    ? [{
      kind: "restore",
      label: RESTORE_LABEL,
      variant: "secondary",
      disabled: false,
      reason: "Brings the newest cloud backup onto this device.",
    }]
    : [];

  switch (state) {
    case "local_only":
    case "sign_in_required":
    case "stopped":
      return [
        {
          kind: "protect",
          label: "Protect this memory",
          variant: "primary",
          disabled: false,
          reason: null,
        },
        ...restoreOnly,
      ];

    case "protected":
    case "changes_pending":
      return pushPullActions(status);

    case "verification_pending":
      // An unchecked upload inside the weekly check window is the normal
      // resting state. Only offer the manual check once it is genuinely
      // overdue, so a routine condition stops looking like a task.
      return [
        ...(verificationIsOverdue(status)
          ? [{
            kind: "verify" as const,
            label: "Check this backup restores",
            variant: "primary" as const,
            disabled: false,
            reason: null,
          }]
          : []),
        ...pushPullActions(status),
      ];

    case "offline":
    case "attention_required": {
      // When repair *is* the backup retry, a second backup button would be a
      // duplicate of the primary action.
      if (repair === "backup") return [...repairButton, ...restoreOnly];
      return [
        ...repairButton,
        blockedBackup(
          state === "offline"
            ? "Ormah Cloud is unreachable. Ormah keeps retrying, and your changes stay safe on this device."
            : "Resolve the problem above before backing up again.",
        ),
        ...restoreOnly,
      ];
    }

    case "subscription_required":
    case "paused":
      return [
        {
          kind: "subscribe",
          label: "Reactivate protection",
          variant: "primary",
          disabled: false,
          reason: null,
        },
        blockedBackup("An active subscription is required to upload new backups."),
        ...restoreOnly,
      ];

    default:
      // initializing, uploading_first_backup, verifying_first_backup
      return [blockedBackup("Ormah is already working on this backup.")];
  }
}

export function protectionPresentation(
  state: ProtectionState,
  status?: ProtectionStatus | null,
): ProtectionPresentation {
  switch (state) {
    case "sign_in_required":
      return {
        tone: "neutral",
        title: "Sign in to protect this memory",
        detail: "Your memory stays on this machine until you explicitly continue.",
        action: "signin",
      };
    case "subscription_required":
      return {
        tone: "neutral",
        title: "Subscription required",
        detail: "Subscribe to keep encrypted backups that are proven to restore.",
        action: "subscribe",
      };
    case "initializing":
      return {
        tone: "working",
        title: "Preparing encryption",
        detail: "Setting up this memory's protected recovery path.",
        action: "wait",
      };
    case "uploading_first_backup":
      return {
        tone: "working",
        title: "Uploading encrypted backup",
        detail: "Only ciphertext is leaving this machine.",
        action: "wait",
      };
    case "verifying_first_backup":
      return {
        tone: "working",
        title: "Proving it restores",
        detail: "Downloading, decrypting, rebuilding, and probing a temporary copy.",
        action: "wait",
      };
    case "verification_pending":
      // Scheduled backups upload daily and the check runs weekly, so an
      // unchecked upload is where a healthy store spends most of its life.
      // Warn only once the check is genuinely overdue, or the panel cries
      // wolf six days in seven.
      return verificationIsOverdue(status)
        ? {
          tone: "warning",
          title: "Backup not proven restorable",
          detail: "Your newest backup is uploaded, but Ormah has not been able to"
            + " confirm it restores. Run a check to be sure.",
          action: "retry",
        }
        : {
          tone: "success",
          title: "Protected",
          detail: "Your newest backup is uploaded. Ormah restore-tests it on a"
            + " schedule, and your last proven backup stays available.",
          action: "backup",
        };
    case "protected":
      return {
        tone: "success",
        title: "Protected",
        detail: "Your newest backup was downloaded, decrypted, and rebuilt on this"
          + " device to prove it restores.",
        action: "backup",
      };
    case "changes_pending":
      return {
        tone: "neutral",
        title: "Changes not backed up yet",
        detail: "Your last proven backup stays available until this device uploads again.",
        action: "backup",
      };
    case "offline":
      return {
        tone: "warning",
        title: "Cloud protection is offline",
        detail: "Ormah will retry automatically. Your changes remain safe on this device.",
        action: "retry",
      };
    case "paused":
      return {
        tone: "warning",
        title: "Cloud uploads are paused",
        detail: "Local memory and existing backups remain available.",
        action: "subscribe",
      };
    case "stopped":
      return {
        tone: "neutral",
        title: "Cloud protection stopped",
        detail: "Existing backups remain available. Billing is managed separately.",
        action: "resume",
      };
    case "attention_required":
      return {
        tone: "danger",
        title: "Protection needs attention",
        detail: "Review the problem below, then retry the failed step.",
        action: "retry",
      };
    default:
      return {
        tone: "neutral",
        title: "Protect this memory",
        detail: "Keep an encrypted backup that Ormah Cloud cannot read, proven to restore.",
        action: "protect",
      };
  }
}
