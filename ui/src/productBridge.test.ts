import { describe, expect, it } from "vitest";

import {
  operationPhaseIsActive,
  protectionRepairAction,
  protectionPresentation,
  type ProtectionStatus,
  type ProtectionState,
} from "./productBridge";

describe("protectionPresentation", () => {
  it.each<ProtectionState>([
    "initializing",
    "uploading_first_backup",
    "verifying_first_backup",
    "verification_pending",
  ])("never calls incomplete state %s protected", (state) => {
    const result = protectionPresentation(state);

    expect(result.title.toLowerCase()).not.toContain("protected");
    expect(result.action).toBe("wait");
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

    expect(result.detail).toContain("safe here");
    expect(result.tone).toBe("warning");
  });
});

describe("operationPhaseIsActive", () => {
  it("distinguishes durable progress from terminal and absent phases", () => {
    for (const phase of ["pending", "running", "uploading", "finalizing", "verifying"] as const) {
      expect(operationPhaseIsActive(phase)).toBe(true);
    }
    for (const phase of ["completed", "failed", "canceled", null, undefined] as const) {
      expect(operationPhaseIsActive(phase)).toBe(false);
    }
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
    expect(protectionRepairAction(status({ last_error_code: "decrypt_failed" }))).toBe("verify");
    expect(protectionRepairAction(status({ last_error_code: "upload_failed" }))).toBe("backup");
    expect(protectionRepairAction(status({ last_error_code: "sign_in_required" }))).toBe("signin");
    expect(protectionRepairAction(status({ last_error_code: "entitlement_expired" }))).toBe("subscribe");
    expect(protectionRepairAction(status({ last_error_code: "key_missing" }))).toBe("none");
    expect(protectionRepairAction(status({ last_error_code: "quota_exceeded" }))).toBe("none");
  });

  it("resumes an interrupted initial protection intent", () => {
    expect(protectionRepairAction(status({
      last_error_code: "operation_interrupted",
      protection_intent_status: "running",
    }))).toBe("resume");
  });
});
