import { renderToStaticMarkup } from "react-dom/server";
import { describe, expect, it } from "vitest";

import { RemoteListingExplanation } from "./ProtectionPanel";

describe("RemoteListingExplanation", () => {
  it("shows the redacted missing-store explanation instead of treating cloud as empty", () => {
    const markup = renderToStaticMarkup(
      <RemoteListingExplanation
        remote={{
          snapshot_id: null,
          created_at: null,
          size_bytes: null,
          from_this_device: false,
          restore_tested_here: false,
          reason_code: "key_missing",
          error: "Cloud store id is missing; import a recovery kit first.",
        }}
      />,
    );

    expect(markup).toContain("Cloud store id is missing; import a recovery kit first.");
    expect(markup).not.toContain("no backup yet");
  });
});
