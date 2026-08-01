import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import RecoveryKitSection from "./RecoveryKitSection";

function render(
  overrides: Partial<Parameters<typeof RecoveryKitSection>[0]> = {},
): string {
  return renderToStaticMarkup(
    <RecoveryKitSection
      ready={false}
      verifiedAt="not yet"
      busy={null}
      error={null}
      onSave={() => undefined}
      onOpenPrintable={() => undefined}
      {...overrides}
    />,
  );
}

describe("RecoveryKitSection", () => {
  it("renders a distinct truthful action-required state", () => {
    const markup = render();

    expect(markup).toContain("Device-loss recovery");
    expect(markup).toContain("Action required");
    expect(markup).toContain("Save recovery kit");
    expect(markup).toContain("Open printable copy");
    expect(markup).toContain("Anyone with this file can read your backups");
    expect(markup).toContain("Opening it does not verify a saved copy");
  });

  it("shows readiness only with the supplied verification time", () => {
    const markup = render({
      ready: true,
      verifiedAt: "Jul 31, 2026 at 12:00 PM",
    });

    expect(markup).toContain("Ready");
    expect(markup).toContain("reopened and verified Jul 31, 2026 at 12:00 PM");
    expect(markup).toContain("Save another copy");
  });

  it("does not claim readiness from a saved-copy timestamp alone", () => {
    const markup = render({
      ready: false,
      verifiedAt: "Jul 31, 2026 at 12:00 PM",
    });

    expect(markup).toContain("Action required");
    expect(markup).not.toContain("reopened and verified Jul 31, 2026");
  });

  it("exposes deterministic accessible busy and error states", () => {
    const markup = render({ busy: "save", error: "Verification failed." });

    expect(markup).toContain('aria-busy="true"');
    expect(markup).toContain('role="alert"');
    expect(markup).toContain("Verification failed.");
    expect(markup).toContain("Saving and checking");
    expect(markup.match(/disabled=""/g)).toHaveLength(2);
  });
});
