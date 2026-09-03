import { describe, expect, it } from "vitest";

import {
  CHECKOUT_CONFIRMATION_INTERVAL_MS,
  CHECKOUT_CONFIRMATION_MAX_CHECKS,
  checkoutConfirmationAfterCheck,
  checkoutConfirmationIsDelayed,
  completeRecoveryKitImport,
  effectiveProtectionState,
  operationPhaseIsActive,
  protectionCompletionSummary,
  protectionIntentNeedsReplacement,
  protectionActions,
  protectionReconnectDelay,
  protectionRepairAction,
  protectionPresentation,
  recoveryKitSectionVisible,
  resolveCheckoutIntent,
  verificationIsOverdue,
  type ProtectionAction,
  type ProtectionActionKind,
  type ProtectionStatus,
  type ProtectionState,
  type RemoteSnapshot,
} from "./productBridge";

describe("checkout confirmation", () => {
  it("keeps waiting through normal delayed-payment confirmation", () => {
    expect(CHECKOUT_CONFIRMATION_INTERVAL_MS).toBe(3_000);
    expect(CHECKOUT_CONFIRMATION_MAX_CHECKS).toBe(40);
    expect(checkoutConfirmationIsDelayed(20)).toBe(false);
    expect(checkoutConfirmationIsDelayed(39)).toBe(false);
    expect(checkoutConfirmationIsDelayed(40)).toBe(true);
  });

  it("does not stop waiting when the app regains focus before Stripe is ready", () => {
    expect(checkoutConfirmationAfterCheck(false, false)).toBe("waiting");
    expect(checkoutConfirmationAfterCheck(false, true)).toBe("delayed");
    expect(checkoutConfirmationAfterCheck(true, false)).toBe("idle");
  });
});

describe("protection intent recovery", () => {
  const operation = (reasonCode: string | null) => ({
    operation_id: "operation",
    kind: "enable" as const,
    phase: "canceled" as const,
    protection_state: "paused" as const,
    reason_code: reasonCode,
    message: null,
    snapshot_id: null,
    protection_intent_id: "intent",
  });

  it("replaces terminal intents but keeps repairable subscription state", () => {
    expect(protectionIntentNeedsReplacement(operation("intent_expired"))).toBe(true);
    expect(protectionIntentNeedsReplacement(operation("intent_canceled"))).toBe(true);
    expect(protectionIntentNeedsReplacement(operation("subscription_required"))).toBe(false);
    expect(protectionIntentNeedsReplacement(operation(null))).toBe(false);
    expect(protectionIntentNeedsReplacement(null, status({
      protection_intent_status: "expired",
    }))).toBe(true);
  });
});

describe("resolveCheckoutIntent", () => {
  it("reuses a durable intent without creating another", async () => {
    let calls = 0;
    const result = await resolveCheckoutIntent("intent-existing", async () => {
      calls += 1;
      throw new Error("must not run");
    });

    expect(result).toEqual({ intentId: "intent-existing", created: null });
    expect(calls).toBe(0);
  });

  it("recovers a lost transient intent before opening Checkout", async () => {
    const created = {
      operation_id: "operation",
      kind: "enable" as const,
      status: "completed" as const,
      submitted_at: "2026-08-13T22:00:00Z",
      started_at: null,
      finished_at: null,
      phase: "completed" as const,
      protection_state: "subscription_required" as const,
      reason_code: null,
      message: null,
      snapshot_id: null,
      protection_intent_id: "intent-created",
    };

    await expect(resolveCheckoutIntent(null, async () => created)).resolves.toEqual({
      intentId: "intent-created",
      created,
    });
  });

  it("surfaces a precise failure instead of silently doing nothing", async () => {
    const missing = {
      operation_id: "operation",
      kind: "enable" as const,
      status: "failed" as const,
      submitted_at: "2026-08-13T22:00:00Z",
      started_at: null,
      finished_at: null,
      phase: "failed" as const,
      protection_state: "subscription_required" as const,
      reason_code: null,
      message: null,
      snapshot_id: null,
      protection_intent_id: null,
    };

    await expect(resolveCheckoutIntent(undefined, async () => missing)).rejects.toThrow(
      "durable request for Checkout",
    );
  });
});

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

  it("warns about an unchecked upload only when no check has ever passed", () => {
    const result = protectionPresentation("verification_pending");

    expect(result.title).toBe("Backup not proven restorable");
    expect(result.tone).toBe("warning");
    expect(result.action).toBe("retry");
  });

  it("stays calm while a routine weekly check is still pending", () => {
    // Daily uploads and weekly checks make this the normal resting state; an
    // amber panel six days in seven trains the user to ignore it.
    const result = protectionPresentation(
      "verification_pending",
      status({ last_successful_verify_at: new Date(Date.now() - 86_400_000).toISOString() }),
    );

    expect(result.tone).toBe("success");
    expect(result.title).toBe("Protected");
  });

  it("uses the proven claim only for the protected state", () => {
    const result = protectionPresentation("protected");

    expect(result.title).toBe("Protected");
    expect(result.detail).toContain("prove it restores");
    expect(result.tone).toBe("success");
  });

  it("keeps retained backups explicit when uploads stop", () => {
    const result = protectionPresentation("stopped");

    expect(result.detail).toContain("Existing backups remain available");
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

describe("verificationIsOverdue", () => {
  const now = Date.parse("2026-08-09T18:00:00Z");

  it("treats a never-proven backup as overdue", () => {
    expect(verificationIsOverdue(status({ last_successful_verify_at: null }), now)).toBe(true);
    expect(verificationIsOverdue(null, now)).toBe(true);
  });

  it("tolerates one missed weekly check before calling it overdue", () => {
    // Uploads are daily and checks weekly, so an unchecked upload is normal.
    expect(verificationIsOverdue(
      status({ last_successful_verify_at: "2026-08-04T18:00:00Z" }),
      now,
    )).toBe(false);
    expect(verificationIsOverdue(
      status({ last_successful_verify_at: "2026-07-20T18:00:00Z" }),
      now,
    )).toBe(true);
  });
});

function remote(overrides: Partial<RemoteSnapshot> = {}): RemoteSnapshot {
  return {
    snapshot_id: "01KZKQNY89FGP5CTS04GT7Y3JX",
    created_at: "2026-08-09T17:03:44Z",
    size_bytes: 1238414,
    from_this_device: true,
    restore_tested_here: true,
    reason_code: null,
    error: null,
    ...overrides,
  };
}

describe("protectionActions", () => {
  const find = (
    actions: ProtectionAction[],
    kind: ProtectionActionKind,
  ): ProtectionAction | undefined => actions.find((action) => action.kind === kind);

  it("offers only sign-in before an account exists", () => {
    const actions = protectionActions(false, "protected", status({}), remote());

    expect(actions.map((action) => action.kind)).toEqual(["signin"]);
  });

  it("does not read another device's upload as this device being behind", () => {
    // A restore records nothing locally, so a machine that has just pulled this
    // very snapshot still reports `from_this_device: false`. Urging it to
    // restore again told it to redo the work it had only just finished.
    const actions = protectionActions(
      true,
      "protected",
      status({ protection_state: "protected" }),
      remote({ from_this_device: false }),
    );

    expect(find(actions, "restore")?.variant).toBe("secondary");
    expect(find(actions, "backup")?.variant).toBe("secondary");
  });

  it("keeps push and pull in one stable order whatever the cloud holds", () => {
    const ours = protectionActions(true, "protected", status({}), remote());
    const theirs = protectionActions(
      true,
      "protected",
      status({}),
      remote({ from_this_device: false }),
    );

    expect(ours.map((action) => action.kind)).toEqual(["backup", "restore"]);
    expect(theirs.map((action) => action.kind)).toEqual(ours.map((action) => action.kind));
  });

  it("leads with backup when this device holds unprotected changes", () => {
    const actions = protectionActions(
      true,
      "changes_pending",
      status({ protection_state: "changes_pending" }),
      remote(),
    );

    expect(actions.map((action) => action.kind)).toEqual(["backup", "restore"]);
    expect(find(actions, "backup")?.variant).toBe("primary");
  });

  it("always says what restoring costs before it is chosen", () => {
    const actions = protectionActions(
      true,
      "changes_pending",
      status({ protection_state: "changes_pending" }),
      remote({ from_this_device: false }),
    );

    expect(find(actions, "restore")?.reason).toContain("Replaces this device's memory");
    expect(find(actions, "restore")?.reason).toContain("safety copy");
  });

  it("does not let a healthy store present backup as an outstanding task", () => {
    const actions = protectionActions(
      true,
      "protected",
      status({ protection_state: "protected" }),
      remote(),
    );

    expect(find(actions, "backup")?.variant).toBe("secondary");
    expect(find(actions, "backup")?.disabled).toBe(false);
    expect(find(actions, "restore")).toBeDefined();
  });

  it("keeps a routine unchecked upload free of a check prompt", () => {
    // Daily uploads with a weekly check mean this is the normal resting state.
    const actions = protectionActions(
      true,
      "verification_pending",
      status({
        protection_state: "verification_pending",
        last_successful_verify_at: new Date(Date.now() - 86_400_000).toISOString(),
      }),
      remote(),
    );

    expect(find(actions, "verify")).toBeUndefined();
    expect(find(actions, "backup")).toBeDefined();
  });

  it("asks for a check only once one is genuinely overdue", () => {
    const actions = protectionActions(
      true,
      "verification_pending",
      status({
        protection_state: "verification_pending",
        last_successful_verify_at: null,
      }),
      remote(),
    );

    expect(actions[0].kind).toBe("verify");
    expect(actions[0].variant).toBe("primary");
  });

  it("offers restore on a machine that has never been protected itself", () => {
    // How a second device adopts an existing memory.
    const actions = protectionActions(
      true,
      "local_only",
      status({ protection_state: "local_only", enabled: false }),
      remote({ from_this_device: false }),
    );

    expect(actions.map((action) => action.kind)).toEqual(["protect", "restore"]);
  });

  it("offers kit recovery before a fresh device can list an existing store", () => {
    const actions = protectionActions(
      true,
      "local_only",
      status({ protection_state: "local_only", enabled: false, store_id: null }),
      remote({
        snapshot_id: null,
        created_at: null,
        size_bytes: null,
        reason_code: "key_missing",
        error: "Cloud store id is missing; import a recovery kit first.",
      }),
    );

    expect(actions.map((action) => action.kind)).toEqual(["protect", "recover"]);
    expect(find(actions, "recover")?.label).toBe("Recover from a backup");
  });

  it("keeps Protect available for a genuinely new local-only account", () => {
    const actions = protectionActions(
      true,
      "local_only",
      status({ protection_state: "local_only", enabled: false, store_id: null }),
      remote({ snapshot_id: null, created_at: null, size_bytes: null, reason_code: null }),
    );

    expect(actions.map((action) => action.kind)).toEqual(["protect"]);
  });

  it("keeps backup visible and explained while cloud protection is offline", () => {
    const actions = protectionActions(
      true,
      "offline",
      status({ protection_state: "offline", enabled: false }),
      remote(),
    );

    expect(find(actions, "repair")?.label).toBe("Check connection");
    expect(find(actions, "backup")?.disabled).toBe(true);
    expect(find(actions, "backup")?.reason).toContain("Ormah Cloud is unreachable");
  });

  it("does not duplicate backup when repair is itself the backup retry", () => {
    const actions = protectionActions(
      true,
      "offline",
      status({ protection_state: "offline", enabled: true }),
      null,
    );

    expect(actions.map((action) => action.kind)).toEqual(["repair"]);
    expect(find(actions, "repair")?.label).toBe("Retry encrypted backup");
  });

  it("blocks backup with a reason while an unrepairable failure stands", () => {
    const actions = protectionActions(
      true,
      "attention_required",
      status({ protection_state: "attention_required", last_error_code: "quota_exceeded" }),
      null,
    );

    expect(find(actions, "repair")).toBeUndefined();
    expect(find(actions, "backup")?.disabled).toBe(true);
    expect(find(actions, "backup")?.reason).toContain("Resolve the problem above");
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
      const actions = protectionActions(true, state, status({ protection_state: state }), remote());

      expect(actions.length).toBeGreaterThan(0);
      for (const action of actions) {
        if (action.disabled) expect(action.reason).toBeTruthy();
      }

      const backup = find(actions, "backup");
      if (onboarding.includes(state)) {
        expect(find(actions, "protect")).toBeDefined();
        continue;
      }
      if (!backup) {
        expect(find(actions, "repair")?.label).toBe("Retry encrypted backup");
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

describe("completeRecoveryKitImport", () => {
  it("refreshes discovery before preparing restore after kit import", async () => {
    const calls: string[] = [];

    const result = await completeRecoveryKitImport(
      async () => {
        calls.push("import");
        return { status: "imported" };
      },
      async () => { calls.push("refresh"); },
      async () => { calls.push("prepare"); },
    );

    expect(result).toBe("imported");
    expect(calls).toEqual(["import", "refresh", "prepare"]);
  });

  it("returns to the caller cleanly when choosing a kit is canceled", async () => {
    const calls: string[] = [];

    const result = await completeRecoveryKitImport(
      async () => {
        calls.push("import");
        return { status: "canceled" };
      },
      async () => { calls.push("refresh"); },
      async () => { calls.push("prepare"); },
    );

    expect(result).toBe("canceled");
    expect(calls).toEqual(["import"]);
  });
});
