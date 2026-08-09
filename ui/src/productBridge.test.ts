import { describe, expect, it } from "vitest";

import {
  effectiveProtectionState,
  operationPhaseIsActive,
  protectionCompletionSummary,
  protectionActions,
  protectionReconnectDelay,
  protectionRepairAction,
  protectionPresentation,
  recoveryKitSectionVisible,
  type ProtectionAction,
  type ProtectionActionKind,
  type ProtectionStatus,
  type ProtectionState,
} from "./productBridge";

describe("protectionPresentation", () => {
  it.each<ProtectionState>([
    "initializing",
    "uploading_first_backup",
    "verifying_first_backup",
  ])("never calls incomplete state %s protected", (state) => {
    const result = protectionPresentation(state);

    expect(result.title.toLowerCase()).not.toContain("protected");
    expect(result.action).toBe("wait");
  });

  it("makes an uploaded but unchecked backup an explicit recovery action", () => {
    const result = protectionPresentation("verification_pending");

    expect(result.title).toBe("Recovery check needed");
    expect(result.tone).toBe("warning");
    expect(result.action).toBe("retry");
  });

  it("uses the verified claim only for the protected state", () => {
    const result = protectionPresentation("protected");

    expect(result.title).toBe("Protected and verified");
    expect(result.tone).toBe("success");
  });

  it("keeps retained recovery points explicit when uploads stop", () => {
    const result = protectionPresentation("stopped");

    expect(result.detail).toContain("recovery points remain available");
    expect(result.action).toBe("resume");
  });

  it("does not imply local data loss while offline", () => {
    const result = protectionPresentation("offline");

    expect(result.detail).toContain("safe on this device");
    expect(result.detail).toContain("retry automatically");
    expect(result.tone).toBe("warning");
  });
});

describe("protectionReconnectDelay", () => {
  it("backs off to a capped reconnect interval", () => {
    expect([0, 1, 2, 3, 4, 5].map(protectionReconnectDelay)).toEqual([
      5_000,
      15_000,
      30_000,
      60_000,
      60_000,
      60_000,
    ]);
  });

  it("treats an invalid attempt as the first retry", () => {
    expect(protectionReconnectDelay(Number.NaN)).toBe(5_000);
    expect(protectionReconnectDelay(-3)).toBe(5_000);
  });
});

describe("operationPhaseIsActive", () => {
  it("distinguishes durable progress from terminal and absent phases", () => {
    for (const phase of [
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
    ] as const) {
      expect(operationPhaseIsActive(phase)).toBe(true);
    }
    for (const phase of ["ready", "completed", "failed", "canceled", null, undefined] as const) {
      expect(operationPhaseIsActive(phase)).toBe(false);
    }
  });
});

describe("protectionCompletionSummary", () => {
  it("uses backend timing and verified counts for a backup receipt", () => {
    const summary = protectionCompletionSummary({
      operation_id: "operation",
      kind: "backup",
      status: "completed",
      submitted_at: "2026-08-01T19:00:00Z",
      started_at: "2026-08-01T19:00:01Z",
      finished_at: "2026-08-01T19:00:07.4Z",
      phase: "completed",
      protection_state: "protected",
      reason_code: null,
      message: null,
      snapshot_id: "snapshot",
      protection_intent_id: null,
      verified_node_count: 1839,
    });

    expect(summary).toContain("1,839 active memories");
    expect(summary).toContain("encrypted, uploaded, and restore-tested in 6 seconds");
  });

  it("never invents a receipt without a verified count and valid server timing", () => {
    expect(protectionCompletionSummary({
      operation_id: "operation",
      kind: "backup",
      status: "completed",
      started_at: null,
      finished_at: null,
      phase: "completed",
      protection_state: "protected",
      reason_code: null,
      message: null,
      snapshot_id: "snapshot",
      protection_intent_id: null,
      verified_node_count: null,
    })).toBeNull();
  });
});

function status(overrides: Partial<ProtectionStatus>): ProtectionStatus {
  return {
    enabled: true,
    store_id: "store",
    entitlement: "active",
    protection_state: "attention_required",
    protection_intent_id: null,
    protection_intent_status: null,
    protection_intent_expires_at: null,
    last_operation_id: null,
    last_operation_kind: null,
    last_operation_phase: "failed",
    last_successful_upload_at: null,
    last_successful_backup_snapshot_id: null,
    last_successful_verify_at: null,
    last_verified_snapshot_id: null,
    recovery_kit_verified_at: null,
    device_loss_recovery_ready: false,
    last_error_code: null,
    last_error_message: null,
    warnings: [],
    ...overrides,
  };
}

describe("protectionRepairAction", () => {
  it("routes failures only to operations that can repair them", () => {
    expect(protectionRepairAction(status({ last_error_code: "decrypt_failed" }))).toBe("none");
    expect(protectionRepairAction(status({ last_error_code: "upload_failed" }))).toBe("backup");
    expect(protectionRepairAction(status({ last_error_code: "sign_in_required" }))).toBe("signin");
    expect(protectionRepairAction(status({ last_error_code: "entitlement_expired" }))).toBe("subscribe");
    expect(protectionRepairAction(status({ last_error_code: "key_missing" }))).toBe("none");
    expect(protectionRepairAction(status({ last_error_code: "quota_exceeded" }))).toBe("none");
    expect(protectionRepairAction(status({
      protection_state: "verification_pending",
      last_error_code: null,
    }))).toBe("verify");
  });

  it("resumes an interrupted initial protection intent", () => {
    expect(protectionRepairAction(status({
      last_error_code: "operation_interrupted",
      protection_intent_status: "running",
    }))).toBe("resume");
  });
});

describe("effectiveProtectionState", () => {
  it("does not let a completed operation hide newer durable status", () => {
    const durable = status({ protection_state: "protected" });
    const completed = {
      operation_id: "operation",
      kind: "backup" as const,
      status: "completed" as const,
      phase: "completed" as const,
      protection_state: "verification_pending" as const,
      reason_code: null,
      message: null,
      snapshot_id: "snapshot",
      protection_intent_id: null,
    };

    expect(effectiveProtectionState(completed, durable)).toBe("protected");
  });

  it("uses live operation state while work is active", () => {
    const durable = status({ protection_state: "protected" });
    const running = {
      operation_id: "operation",
      kind: "backup" as const,
      status: "running" as const,
      phase: null,
      protection_state: "verifying_first_backup" as const,
      reason_code: null,
      message: null,
      snapshot_id: null,
      protection_intent_id: null,
    };

    expect(effectiveProtectionState(running, durable)).toBe("verifying_first_backup");
  });
});

describe("protectionActions", () => {
  const find = (
    actions: ProtectionAction[],
    kind: ProtectionActionKind,
  ): ProtectionAction | undefined => actions.find((action) => action.kind === kind);

  it("offers only sign-in before an account exists", () => {
    const actions = protectionActions(false, "protected", status({}));

    expect(actions.map((action) => action.kind)).toEqual(["signin"]);
  });

  it.each<ProtectionState>(["protected", "changes_pending"])(
    "makes backup the single primary action in %s",
    (state) => {
      const actions = protectionActions(true, state, status({ protection_state: state }));

      expect(actions).toEqual([{
        kind: "backup",
        label: "Back up now",
        variant: "primary",
        disabled: false,
        reason: null,
      }]);
    },
  );

  it("prefers verification but keeps backup usable and explained when a check is pending", () => {
    const actions = protectionActions(
      true,
      "verification_pending",
      status({ protection_state: "verification_pending", last_error_code: null }),
    );

    expect(actions.map((action) => action.kind)).toEqual(["verify", "backup"]);
    expect(find(actions, "verify")?.variant).toBe("primary");

    const backup = find(actions, "backup");
    expect(backup?.label).toBe("Back up now");
    expect(backup?.variant).toBe("secondary");
    // A manual backup restore-tests its own upload, so it is slower here, not unsafe.
    expect(backup?.disabled).toBe(false);
    expect(backup?.reason).toContain("restore-tests that one instead");
  });

  it("keeps backup visible and explained while cloud protection is offline", () => {
    const actions = protectionActions(
      true,
      "offline",
      status({ protection_state: "offline", enabled: false }),
    );

    expect(find(actions, "repair")?.label).toBe("Check connection");
    const backup = find(actions, "backup");
    expect(backup?.disabled).toBe(true);
    expect(backup?.reason).toContain("Ormah Cloud is unreachable");
  });

  it("does not duplicate backup when repair is itself the backup retry", () => {
    const actions = protectionActions(
      true,
      "offline",
      status({ protection_state: "offline", enabled: true }),
    );

    expect(actions.map((action) => action.kind)).toEqual(["repair"]);
    expect(find(actions, "repair")?.label).toBe("Retry encrypted backup");
  });

  it("blocks backup with a reason while an unrepairable failure stands", () => {
    const actions = protectionActions(
      true,
      "attention_required",
      status({ protection_state: "attention_required", last_error_code: "quota_exceeded" }),
    );

    expect(find(actions, "repair")).toBeUndefined();
    const backup = find(actions, "backup");
    expect(backup?.disabled).toBe(true);
    expect(backup?.reason).toContain("Resolve the problem above");
  });

  it("keeps the repair path primary and backup explained during a verify failure", () => {
    const actions = protectionActions(
      true,
      "attention_required",
      status({ protection_state: "attention_required", last_error_code: "verification_failed" }),
    );

    expect(actions.map((action) => action.kind)).toEqual(["repair", "backup"]);
    expect(find(actions, "repair")?.label).toBe("Retry recovery check");
    expect(find(actions, "backup")?.disabled).toBe(true);
  });

  it("never hides backup from a signed-in, protected store without saying why", () => {
    const onboarding: ProtectionState[] = ["local_only", "sign_in_required", "stopped"];
    const states: ProtectionState[] = [
      "local_only",
      "sign_in_required",
      "subscription_required",
      "initializing",
      "uploading_first_backup",
      "verifying_first_backup",
      "verification_pending",
      "protected",
      "changes_pending",
      "offline",
      "paused",
      "stopped",
      "attention_required",
    ];

    for (const state of states) {
      const actions = protectionActions(true, state, status({ protection_state: state }));

      expect(actions.length).toBeGreaterThan(0);
      for (const action of actions) {
        if (action.disabled) expect(action.reason).toBeTruthy();
      }

      const backup = find(actions, "backup");
      const repair = find(actions, "repair");
      if (onboarding.includes(state)) {
        // Protection does not exist yet; "Protect this memory" is the entry point.
        expect(find(actions, "protect")).toBeDefined();
        continue;
      }
      if (!backup) {
        expect(repair?.label).toBe("Retry encrypted backup");
        continue;
      }
      if (backup.disabled) expect(backup.reason).toBeTruthy();
    }
  });
});

describe("recoveryKitSectionVisible", () => {
  it("keeps recovery export available while protection needs attention", () => {
    expect(recoveryKitSectionVisible(status({
      enabled: true,
      protection_state: "attention_required",
      last_successful_verify_at: null,
    }))).toBe(true);
    expect(recoveryKitSectionVisible(status({ enabled: false }))).toBe(false);
    expect(recoveryKitSectionVisible(null)).toBe(false);
  });
});
