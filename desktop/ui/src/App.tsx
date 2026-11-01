import { useEffect, useRef, useState } from "react";
import GraphCanvas from "@/components/GraphCanvas";
import Act1Void from "@/components/Act1Void";
import {
  invoke, graphUrl, sleep,
  winMinimize, winClose, winToggleMaximize, type SetupStatus,
} from "./lib/bridge";

// Title bar for the startup splash (frameless window). On the graph view
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

function statusMessage(status: SetupStatus): string {
  switch (status.phase) {
    case "installing":
      return `Installing Ormah ${status.version}… This can take a few minutes.`
        + (status.attempt > 1 ? ` Try ${status.attempt} of ${status.attempts}.` : "");
    case "retrying":
      return `Couldn't connect. Trying again shortly (try ${status.attempt} of ${status.attempts})…`;
    case "failed":
      return status.reason;
    default:
      return "Starting Ormah… The first start downloads models and can take a few minutes.";
  }
}

export default function App() {
  const progressRef = useRef(0);
  const selfNodeReadyRef = useRef<{ x: number; y: number } | null>(null);
  const [status, setStatus] = useState<SetupStatus>({ phase: "starting" });
  const [notice, setNotice] = useState("");
  const navigating = useRef(false);

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
    setNotice("");
    try {
      if (await invoke<boolean>("retry_runtime_setup")) setStatus({ phase: "starting" });
    } catch (error) {
      setNotice(`Couldn't retry: ${error}`);
    }
  }

  async function copyDetails() {
    try {
      await invoke("copy_setup_details");
      setNotice("Error log copied.");
    } catch (error) {
      setNotice(`Couldn't copy the error log: ${error}`);
    }
  }

  useEffect(() => {
    let cancelled = false;
    const t0 = Date.now();

    (async () => {
      while (!cancelled) {
        try {
          const next = await invoke<SetupStatus>("runtime_setup_status");
          if (cancelled) return;
          setStatus(next);
          if (next.phase === "ready") {
            await sleep(Math.max(0, 2200 - (Date.now() - t0)));
            if (!cancelled) await goGraph();
            return;
          }
        } catch (error) {
          if (!cancelled) setNotice(`Couldn't read setup progress: ${error}`);
        }
        await sleep(500);
      }
    })();

    return () => { cancelled = true; };
  }, []);

  const failed = status.phase === "failed";

  return (
    <>
      <TitleBar />
      <div className="stage-wrap">
        <GraphCanvas progressRef={progressRef} selfNodeReadyRef={selfNodeReadyRef} />
        <Act1Void progressRef={progressRef} selfNodeReadyRef={selfNodeReadyRef} />
        <div className="boot-status" role={failed ? "alert" : "status"}>
          <span>{statusMessage(status)}</span>
          {failed && (
            <div className="boot-actions">
              <button onClick={() => void retrySetup()}>Try again</button>
              <button onClick={() => void copyDetails()}>Copy error log</button>
              {status.can_open && <button onClick={() => void goGraph()}>Open Ormah</button>}
            </div>
          )}
          {notice && <span>{notice}</span>}
        </div>
      </div>
    </>
  );
}
