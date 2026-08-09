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
  "decrypt_failed",
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

export function protectionPresentation(state: ProtectionState): ProtectionPresentation {
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
        detail: "Subscribe to create encrypted, verified recovery points.",
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
        title: "Uploading encrypted recovery point",
        detail: "Only ciphertext is leaving this machine.",
        action: "wait",
      };
    case "verifying_first_backup":
      return {
        tone: "working",
        title: "Checking recovery",
        detail: "Downloading, decrypting, rebuilding, and probing a temporary copy.",
        action: "wait",
      };
    case "verification_pending":
      return {
        tone: "warning",
        title: "Recovery check needed",
        detail: "The encrypted backup is uploaded but has not yet passed a restore test.",
        action: "retry",
      };
    case "protected":
      return {
        tone: "success",
        title: "Protected and verified",
        detail: "The latest recovery point was restored and checked on this device.",
        action: "backup",
      };
    case "changes_pending":
      return {
        tone: "warning",
        title: "Changes waiting for protection",
        detail: "Your last verified recovery point remains available.",
        action: "backup",
      };
    case "offline":
      return {
        tone: "warning",
        title: "Cloud protection is offline",
        detail: "Changes are safe here and will resume when the service is reachable.",
        action: "retry",
      };
    case "paused":
      return {
        tone: "warning",
        title: "Cloud uploads are paused",
        detail: "Local memory and retained recovery points remain available.",
        action: "subscribe",
      };
    case "stopped":
      return {
        tone: "neutral",
        title: "Cloud protection stopped",
        detail: "Existing recovery points remain available. Billing is managed separately.",
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
        detail: "Keep an encrypted, verified recovery copy that Ormah Cloud cannot read.",
        action: "protect",
      };
  }
}
