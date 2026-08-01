import { useEffect, useRef, useState } from "react";
import GraphCanvas from "@/components/GraphCanvas";
import Act1Void from "@/components/Act1Void";
import InstallPanel from "./InstallPanel";
import {
  invoke, graphUrl, waitForServer, sleep,
  winMinimize, winClose, winToggleMaximize, onServerStatus,
} from "./lib/bridge";

type Phase = "intro" | "connect";

// Title bar for the intro/connect phase (frameless window). On the graph view
// the graph's own top bar carries the window controls.
function TitleBar() {
  return (
    <div className="titlebar">
      <div className="titlebar-drag" data-tauri-drag-region />
      <div className="titlebar-controls">
        <button className="tbtn" aria-label="Minimize" onClick={() => winMinimize()}>
          <svg width="11" height="11" viewBox="0 0 11 11"><rect x="1" y="5" width="9" height="1" fill="currentColor" /></svg>
        </button>
        <button className="tbtn" aria-label="Maximize" onClick={() => winToggleMaximize()}>
          <svg width="11" height="11" viewBox="0 0 11 11"><rect x="1.5" y="1.5" width="8" height="8" fill="none" stroke="currentColor" strokeWidth="1" /></svg>
        </button>
        <button className="tbtn close" aria-label="Close" onClick={() => winClose()}>
          <svg width="11" height="11" viewBox="0 0 11 11"><path d="M1 1l9 9M10 1l-9 9" stroke="currentColor" strokeWidth="1.1" /></svg>
        </button>
      </div>
    </div>
  );
}

export default function App() {
  const progressRef = useRef(0);
  const selfNodeReadyRef = useRef<{ x: number; y: number } | null>(null);
  const [phase, setPhase] = useState<Phase>("intro");
  const [statusMsg, setStatusMsg] = useState("");
  const [canRetry, setCanRetry] = useState(false);
  const [retryBusy, setRetryBusy] = useState(false);
  const started = useRef(false);
  const navigating = useRef(false);

  useEffect(() => {
    return onServerStatus((s) => {
      if (s.phase === "installing") {
        setCanRetry(false);
        setStatusMsg(`Installing Ormah ${s.version} (attempt ${s.attempt} of ${s.attempts})…`);
      } else if (s.phase === "retrying") {
        setCanRetry(false);
        setStatusMsg(`Connection unavailable. Trying again in ${s.delay_seconds} seconds…`);
      } else if (s.phase === "starting") {
        setCanRetry(false);
        setStatusMsg("Starting Ormah…");
      } else if (s.phase === "failed") {
        setCanRetry(s.can_retry);
        setStatusMsg(s.reason);
      }
    });
  }, []);

  async function goGraph() {
    if (navigating.current) return;
    navigating.current = true;
    const url = await graphUrl();
    document.getElementById("root")?.classList.add("to-graph");
    await sleep(620);
    // Navigate to the graph; its TopBar becomes the window title bar.
    // Cache-bust so a fresh UI build always loads in the webview.
    window.location.replace(url + (url.includes("?") ? "&" : "?") + "_=" + Date.now());
  }

  async function retrySetup() {
    setRetryBusy(true);
    setCanRetry(false);
    try {
      let onboarded = false;
      try { onboarded = await invoke<boolean>("is_onboarded"); } catch { /* ignore */ }
      const accepted = await invoke<boolean>("retry_runtime_setup");
      if (!accepted) {
        setCanRetry(true);
        return;
      }
      const ready = await waitForServer();
      if (!ready) return;
      if (onboarded) await goGraph();
      else setPhase("connect");
    } finally {
      setRetryBusy(false);
    }
  }

  useEffect(() => {
    if (started.current) return;
    started.current = true;

    (async () => {
      const t0 = Date.now();
      let onboarded = false;
      try { onboarded = await invoke<boolean>("is_onboarded"); } catch { /* ignore */ }

      const ready = await waitForServer();
      if (!ready) return;

      if (onboarded) {
        await sleep(Math.max(0, 2200 - (Date.now() - t0)));
        goGraph();
        return;
      }
      await sleep(Math.max(0, 5200 - (Date.now() - t0)));
      setPhase("connect");
    })();
  }, []);

  return (
    <>
      <TitleBar />
      <div className="stage-wrap">
        <GraphCanvas progressRef={progressRef} selfNodeReadyRef={selfNodeReadyRef} />
        <Act1Void progressRef={progressRef} selfNodeReadyRef={selfNodeReadyRef} />
        {phase === "intro" && statusMsg && (
          <div className="boot-status" role={canRetry ? "alert" : "status"}>
            <span>{statusMsg}</span>
            {canRetry && (
              <button disabled={retryBusy} onClick={() => void retrySetup()}>
                {retryBusy ? "Trying again…" : "Try again"}
              </button>
            )}
          </div>
        )}
        {phase === "connect" && <InstallPanel onDone={goGraph} />}
      </div>
    </>
  );
}
