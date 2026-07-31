import { useCallback, useRef, useState } from "react";
import { ShieldCheck } from "lucide-react";
import { searchNodes } from "../api";
import SearchResults from "./SearchResults";
import type { MemoryNode } from "../types";
import type { ProtectionState } from "../productBridge";

type PanelId = "protection" | "settings" | "insights" | "admin" | "agents";

interface Props {
  nodeCount: number;
  activePanel: PanelId | null;
  onTogglePanel: (id: PanelId) => void;
  onSearchSelect: (nodeId: string) => void;
  onSearchHover: (nodeId: string) => void;
  onSearchHoverEnd: () => void;
  searchInputRef: React.RefObject<HTMLInputElement>;
  protectionState: ProtectionState | null;
}

export default function TopBar({
  nodeCount,
  activePanel,
  onTogglePanel,
  onSearchSelect,
  onSearchHover,
  onSearchHoverEnd,
  searchInputRef,
  protectionState,
}: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<MemoryNode[] | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();
  const blurTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const wrapperRef = useRef<HTMLDivElement>(null);

  // When running inside the Ormah desktop app, this bar doubles as the window
  // title bar: dragging it moves the window and it carries the window controls.
  const tauriWin = () =>
    (window as unknown as { __TAURI__?: { window?: { getCurrentWindow?: () => { minimize?: () => void; toggleMaximize?: () => void; close?: () => void; startDragging?: () => void } } } })
      .__TAURI__?.window?.getCurrentWindow?.();
  const inApp = typeof window !== "undefined" && (
    !!(window as unknown as { __TAURI__?: unknown }).__TAURI__ ||
    !!(window as unknown as { __ORMAH_DESKTOP__?: unknown }).__ORMAH_DESKTOP__
  );
  if (typeof document !== "undefined") {
    if (inApp) document.body.classList.add("in-app");
    document.documentElement.classList.toggle("ormah-desktop", inApp);
  }
  // Manual drag: data-tauri-drag-region isn't honored on the navigated remote
  // graph page, so call startDragging() on a left-button press anywhere on the
  // bar except interactive elements (search, buttons, the controls).
  const startDrag = (e: React.PointerEvent) => {
    if (e.button !== 0) return;
    const el = e.target as HTMLElement;
    if (el.closest("input, button, a, .search-wrapper, .win-controls, .app-actions, .top-bar-actions")) return;
    tauriWin()?.startDragging?.();
  };

  const actionButtons = (
    <>
      <button
        className={`top-bar-btn protection-entry ${activePanel === "protection" ? "active" : ""} ${protectionState === "protected" ? "protected" : ""}`}
        onClick={() => onTogglePanel("protection")}
        title="Cloud protection"
      >
        <ShieldCheck size={14} aria-hidden="true" />
        {protectionState === "protected" ? "protected" : "protect"}
      </button>
      <button
        className={`top-bar-btn ${activePanel === "settings" ? "active" : ""}`}
        onClick={() => onTogglePanel("settings")}
      >
        settings
      </button>
      <button
        className={`top-bar-btn ${activePanel === "insights" ? "active" : ""}`}
        onClick={() => onTogglePanel("insights")}
      >
        insights
      </button>
      <button
        className={`top-bar-btn ${activePanel === "admin" ? "active" : ""}`}
        onClick={() => onTogglePanel("admin")}
      >
        admin
      </button>
      <button
        className={`top-bar-btn ${activePanel === "agents" ? "active" : ""}`}
        onClick={() => onTogglePanel("agents")}
      >
        agents
      </button>
    </>
  );

  const handleChange = useCallback(
    (e: React.ChangeEvent<HTMLInputElement>) => {
      const val = e.target.value;
      setQuery(val);
      setSelectedIndex(-1);

      if (timerRef.current) clearTimeout(timerRef.current);

      if (!val.trim()) {
        setResults(null);
        return;
      }

      timerRef.current = setTimeout(async () => {
        try {
          const nodes = await searchNodes(val);
          setResults(nodes.length > 0 ? nodes : []);
        } catch {
          setResults([]);
        }
      }, 300);
    },
    []
  );

  const handleSelect = useCallback(
    (nodeId: string) => {
      if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
      setQuery("");
      setResults(null);
      setSelectedIndex(-1);
      onSearchHoverEnd();
      onSearchSelect(nodeId);
    },
    [onSearchSelect, onSearchHoverEnd]
  );

  const handleKeyDown = useCallback(
    (e: React.KeyboardEvent<HTMLInputElement>) => {
      if (!results || results.length === 0) {
        if (e.key === "Escape") {
          setQuery("");
          setResults(null);
          searchInputRef.current?.blur();
        }
        return;
      }

      if (e.key === "ArrowDown") {
        e.preventDefault();
        setSelectedIndex((i) => {
          const next = i < results.length - 1 ? i + 1 : 0;
          if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
          hoverTimerRef.current = setTimeout(() => onSearchHover(results[next].id), 150);
          return next;
        });
      } else if (e.key === "ArrowUp") {
        e.preventDefault();
        setSelectedIndex((i) => {
          const next = i > 0 ? i - 1 : results.length - 1;
          if (hoverTimerRef.current) clearTimeout(hoverTimerRef.current);
          hoverTimerRef.current = setTimeout(() => onSearchHover(results[next].id), 150);
          return next;
        });
      } else if (e.key === "Enter") {
        e.preventDefault();
        if (selectedIndex >= 0 && selectedIndex < results.length) {
          handleSelect(results[selectedIndex].id);
        }
      } else if (e.key === "Escape") {
        setQuery("");
        setResults(null);
        setSelectedIndex(-1);
        onSearchHoverEnd();
        searchInputRef.current?.blur();
      }
    },
    [results, selectedIndex, handleSelect, onSearchHover, onSearchHoverEnd, searchInputRef]
  );

  return (
    <div className="top-bar" onPointerDown={startDrag}>
      <div className="top-bar-logo">
        <span>o</span>
        <span className="top-bar-logo-r">
          r
          <span className="top-bar-logo-dot" />
        </span>
        <span>mah</span>
      </div>
      <div
        className="search-wrapper"
        ref={wrapperRef}
        onMouseEnter={() => {
          if (blurTimerRef.current) clearTimeout(blurTimerRef.current);
        }}
        onMouseLeave={() => {
          if (!document.activeElement?.closest(".search-wrapper")) {
            blurTimerRef.current = setTimeout(() => {
              setResults(null);
              onSearchHoverEnd();
            }, 300);
          }
        }}
      >
        <span className="search-icon">&#x2315;</span>
        <input
          ref={searchInputRef}
          className="search-input"
          type="text"
          placeholder="search... (/ or &#8984;K)"
          value={query}
          onChange={handleChange}
          onKeyDown={handleKeyDown}
          onBlur={() => {
            blurTimerRef.current = setTimeout(() => {
              if (!wrapperRef.current?.matches(":hover")) {
                setResults(null);
                setSelectedIndex(-1);
                onSearchHoverEnd();
              }
            }, 300);
          }}
        />
        {results && results.length > 0 && (
          <SearchResults
            results={results}
            selectedIndex={selectedIndex}
            onSelect={handleSelect}
            onHover={onSearchHover}
            onHoverEnd={onSearchHoverEnd}
          />
        )}
      </div>
      <span className="top-bar-stats">{nodeCount} nodes</span>
      <div className="top-bar-spacer" />
      <div className="top-bar-actions">{actionButtons}</div>
      {inApp && (
        <div className="win-controls">
          <button className="win-btn" aria-label="Minimize" onClick={() => tauriWin()?.minimize?.()}>
            <svg width="11" height="11" viewBox="0 0 11 11"><rect x="1" y="5" width="9" height="1" fill="currentColor" /></svg>
          </button>
          <button className="win-btn" aria-label="Maximize" onClick={() => tauriWin()?.toggleMaximize?.()}>
            <svg width="11" height="11" viewBox="0 0 11 11"><rect x="1.5" y="1.5" width="8" height="8" fill="none" stroke="currentColor" strokeWidth="1" /></svg>
          </button>
          <button className="win-btn win-close" aria-label="Close" onClick={() => tauriWin()?.close?.()}>
            <svg width="11" height="11" viewBox="0 0 11 11"><path d="M1 1l9 9M10 1l-9 9" stroke="currentColor" strokeWidth="1.1" /></svg>
          </button>
        </div>
      )}
    </div>
  );
}
