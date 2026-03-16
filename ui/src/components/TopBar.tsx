import { useCallback, useRef, useState } from "react";
import { searchNodes } from "../api";
import SearchResults from "./SearchResults";
import type { MemoryNode } from "../types";

type PanelId = "filter" | "review" | "insights" | "admin";

interface Props {
  nodeCount: number;
  activePanel: PanelId | null;
  onTogglePanel: (id: PanelId) => void;
  onSearchSelect: (nodeId: string) => void;
  onSearchHover: (nodeId: string) => void;
  onSearchHoverEnd: () => void;
  searchInputRef: React.RefObject<HTMLInputElement>;
}

export default function TopBar({
  nodeCount,
  activePanel,
  onTogglePanel,
  onSearchSelect,
  onSearchHover,
  onSearchHoverEnd,
  searchInputRef,
}: Props) {
  const [query, setQuery] = useState("");
  const [results, setResults] = useState<MemoryNode[] | null>(null);
  const [selectedIndex, setSelectedIndex] = useState(-1);
  const timerRef = useRef<ReturnType<typeof setTimeout>>();
  const blurTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const hoverTimerRef = useRef<ReturnType<typeof setTimeout>>();
  const wrapperRef = useRef<HTMLDivElement>(null);

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
    <div className="top-bar">
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
      <div className="top-bar-actions">
        <button
          className={`top-bar-btn ${activePanel === "filter" ? "active" : ""}`}
          onClick={() => onTogglePanel("filter")}
        >
          filter <kbd>1</kbd>
        </button>
        <button
          className={`top-bar-btn ${activePanel === "review" ? "active" : ""}`}
          onClick={() => onTogglePanel("review")}
        >
          review <kbd>2</kbd>
        </button>
        <button
          className={`top-bar-btn ${activePanel === "insights" ? "active" : ""}`}
          onClick={() => onTogglePanel("insights")}
        >
          insights <kbd>3</kbd>
        </button>
        <button
          className={`top-bar-btn ${activePanel === "admin" ? "active" : ""}`}
          onClick={() => onTogglePanel("admin")}
        >
          admin <kbd>4</kbd>
        </button>
      </div>
    </div>
  );
}
