import { useEffect, useRef, useState } from "react";
import GraphCanvas from "@/components/GraphCanvas";
import Act1Void from "@/components/Act1Void";
import {
  invoke, graphUrl, waitForServer, waitForGraphTransition, sleep, errorText,
  winMinimize, winClose, winToggleMaximize, onServerStatus, type ServerStatus,
} from "./lib/bridge";

import { initialSetup, receiveStatus, beginRetry, finishRetry, setupView, markHealthy, canOpenGraph, completeStatusInitialization, type SetupState } from "./lib/setup";

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

export default function App() {
  const progressRef = useRef(0);
  const selfNodeReadyRef = useRef<{ x: number; y: number } | null>(null);
  const [setup, setSetup] = useState(() => initialSetup(Date.now()));
  const setupRef = useRef(setup);
  function updateSetup(update: (state: SetupState) => SetupState) {
    // Lifecycle callbacks update the ref synchronously, so a phase arriving
    // during navigation cannot wait for a React render before blocking it.
    const next = update(setupRef.current);
    setupRef.current = next;
    setSetup(next);
  }
  const [now, setNow] = useState(Date.now());
  const [actionNotice, setActionNotice] = useState("");
  const lifetime = useRef<AbortController | null>(null);
  const retryRequest = useRef(0);
  const view = setupView(setup, now);

  async function refreshSetupStatus(controller: AbortController, request = retryRequest.current) {
    try {
      const snapshot = await invoke<ServerStatus | null>("runtime_setup_status");
      if (controller.signal.aborted || request !== retryRequest.current) return;
      updateSetup((state) => completeStatusInitialization(
        snapshot ? receiveStatus(state, snapshot, Date.now()) : state,
      ));
    } catch (error) {
      if (controller.signal.aborted || request !== retryRequest.current) return;
      updateSetup((state) => completeStatusInitialization({
        ...state, error: `Couldn't read setup progress: ${errorText(error)}`,
      }));
    }
  }

  useEffect(() => {
    const controller = new AbortController();
    const { signal } = controller;
    lifetime.current = controller;
    const startedAt = Date.now();
    let bootstrapSettled = false;
    const reportError = (error: string) => {
      if (!signal.aborted) updateSetup((state) => ({ ...state, error }));
    };
    const stop = onServerStatus(
      (status) => updateSetup((state) => receiveStatus(state, status, Date.now())),
      reportError,
      () => {
        bootstrapSettled = true;
        // A pending mount snapshot must not settle a newer retry's barrier.
        if (retryRequest.current === 0) updateSetup(completeStatusInitialization);
      },
    );
    const clock = setInterval(() => setNow(Date.now()), 250);

    // One health loop handles initial setup, retries and late recovery. Retry
    // buttons only ask Rust to start setup; they never create another poller.
    void (async () => {
      while (!signal.aborted) {
        try {
          const canOpen = () => canOpenGraph(setupRef.current);
          await waitForServer(signal, reportError, 2000, canOpen, async () => {
            // Keep the subscribe-before-snapshot startup ordering. Once that
            // attempt settles, this same loop also covers missing events.
            if (bootstrapSettled) await refreshSetupStatus(controller);
          });
          const url = await graphUrl();
          const transitionReady = await waitForGraphTransition(signal, canOpen,
            (active) => document.getElementById("root")?.classList.toggle("to-graph", active),
            Math.max(0, 2200 - (Date.now() - startedAt)));
          if (signal.aborted) return;
          if (!transitionReady || !canOpen()) {
            document.getElementById("root")?.classList.remove("to-graph");
            continue;
          }
          // The graph's own top bar becomes the window title bar.
          window.location.replace(url + (url.includes("?") ? "&" : "?") + "_=" + Date.now());
          updateSetup((state) => markHealthy(state));
          return;
        } catch (error) {
          if (signal.aborted) return;
          document.getElementById("root")?.classList.remove("to-graph");
          reportError(`Couldn't open Ormah: ${errorText(error)}`);
          try { await sleep(2000, signal); } catch { return; }
        }
      }
    })();

    return () => {
      controller.abort();
      stop();
      clearInterval(clock);
      document.getElementById("root")?.classList.remove("to-graph");
    };
  }, []);

  async function retrySetup() {
    const controller = lifetime.current;
    if (!controller || controller.signal.aborted || setupRef.current.retryBusy) return;
    const request = ++retryRequest.current;
    setActionNotice("");
    updateSetup((state) => beginRetry(state));
    try {
      const accepted = await invoke<boolean>("retry_runtime_setup");
      if (controller.signal.aborted || request !== retryRequest.current) return;
      updateSetup((state) => finishRetry(state, accepted, Date.now()));
      if (accepted) {
        // Native event delivery can lag command acceptance. Refresh the snapshot
        // while the existing subscription remains active; sequence numbers keep
        // an intervening newer event from being replaced by an older snapshot.
        await refreshSetupStatus(controller, request);
      }
    } catch (error) {
      if (!controller.signal.aborted && request === retryRequest.current) {
        updateSetup((state) => finishRetry(state, false, Date.now(), errorText(error)));
      }
    }
  }

  async function setupAction(command: "copy_setup_details" | "reveal_setup_log") {
    const controller = lifetime.current;
    setActionNotice("");
    try {
      await invoke(command);
      if (!controller?.signal.aborted) {
        setActionNotice(command === "copy_setup_details" ? "Setup details copied." : "Setup log shown in your file manager.");
      }
    } catch (error) {
      if (!controller?.signal.aborted) {
        setActionNotice(`Couldn't ${command === "copy_setup_details" ? "copy setup details" : "show the setup log"}: ${errorText(error)}`);
      }
    }
  }

  return (
    <>
      <TitleBar />
      <div className="stage-wrap">
        <GraphCanvas progressRef={progressRef} selfNodeReadyRef={selfNodeReadyRef} />
        <Act1Void progressRef={progressRef} selfNodeReadyRef={selfNodeReadyRef} />
        <div className="boot-status" role={view.failed ? "alert" : "status"}>
          <span>{view.message}</span>
          {view.hint && <span className="boot-hint">{view.hint}</span>}
          {view.error && <span className="boot-error">{view.error}</span>}
          {view.failed && (
            <div className="boot-actions">
              {view.canRetry && (
                <button disabled={setup.retryBusy} onClick={() => void retrySetup()}>
                  {setup.retryBusy ? "Trying again…" : "Try again"}
                </button>
              )}
              <button onClick={() => void setupAction("copy_setup_details")}>Copy details</button>
              <button onClick={() => void setupAction("reveal_setup_log")}>Show log</button>
            </div>
          )}
          {actionNotice && <span role="status">{actionNotice}</span>}
        </div>
      </div>
    </>
  );
}
