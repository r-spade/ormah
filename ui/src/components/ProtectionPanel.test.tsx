import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { protectionActions, type ProtectionStatus } from "../productBridge";
import {
  ProtectionActionList,
  RemoteListingExplanation,
} from "./ProtectionPanel";

describe("RemoteListingExplanation", () => {
  it("renders fixed product copy instead of a backend diagnostic", () => {
    const markup = renderToStaticMarkup(
      <RemoteListingExplanation
        remote={{
          snapshot_id: null,
          created_at: null,
          size_bytes: null,
          from_this_device: false,
          restore_tested_here: false,
          reason_code: "remote_listing_failed",
          error: "ConnectError(private-hostname:9443)",
        }}
      />,
    );

    expect(markup).toContain("Cloud backups could not be checked. Try again.");
    expect(markup).not.toContain("private-hostname");
  });

  it("does not turn an uninitialized store into a warning", () => {
    const markup = renderToStaticMarkup(
      <RemoteListingExplanation
        remote={{
          snapshot_id: null,
          created_at: null,
          size_bytes: null,
          from_this_device: false,
          restore_tested_here: false,
          reason_code: "key_missing",
          error: "This device is not connected to a cloud memory store.",
        }}
      />,
    );

    expect(markup).toBe("");
  });
});

describe("ProtectionActionList", () => {
  it("renders the two honest choices for a device without a store identity", () => {
    const status: ProtectionStatus = {
      enabled: false,
      store_id: null,
      entitlement: "none",
      protection_state: "local_only",
      protection_intent_id: null,
      protection_intent_status: null,
      protection_intent_expires_at: null,
      last_operation_id: null,
      last_operation_kind: null,
      last_operation_phase: null,
      last_successful_upload_at: null,
      last_successful_backup_snapshot_id: null,
      last_successful_verify_at: null,
      last_verified_snapshot_id: null,
      recovery_kit_verified_at: null,
      device_loss_recovery_ready: false,
      last_error_code: null,
      last_error_message: null,
      warnings: [],
    };
    const actions = protectionActions(true, "local_only", status, null);
    const markup = renderToStaticMarkup(
      <ProtectionActionList
        actions={actions}
        busy={false}
        onAction={() => undefined}
      />,
    );

    expect(markup).toContain("Protect this memory");
    expect(markup).toContain("Recover existing memory");
    expect(markup).toContain("Use the recovery kit saved from an existing protected memory.");
  });
});
