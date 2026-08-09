import { describe, expect, it } from "vitest";

import { desktopUpdateProgress, type DesktopUpdateStatus } from "./desktopUpdate";

function status(overrides: Partial<DesktopUpdateStatus>): DesktopUpdateStatus {
  return {
    phase: "downloading",
    version: "0.4.0",
    notes: "Safer recovery",
    progress_percent: null,
    message: null,
    ...overrides,
  };
}

describe("desktopUpdateProgress", () => {
  it("shows determinate and indeterminate signed-download progress", () => {
    expect(desktopUpdateProgress(status({ progress_percent: 42 }))).toContain("42%");
    expect(desktopUpdateProgress(status({ progress_percent: null }))).toBe(
      "Downloading signed update",
    );
  });

  it("does not imply installation while only downloading", () => {
    expect(desktopUpdateProgress(status({ phase: "installing" }))).toBe(
      "Installing signed update",
    );
    expect(desktopUpdateProgress(status({ phase: "available" }))).toBe("");
  });
});
