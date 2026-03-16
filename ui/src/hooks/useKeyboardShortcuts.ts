import { useEffect } from "react";

type PanelId = "filter" | "review" | "insights" | "admin";

interface Options {
  onTogglePanel: (id: PanelId) => void;
  onClosePanel: () => void;
  onCloseDetail: () => void;
  onFocusSearch: () => void;
  activePanel: PanelId | null;
  hasDetail: boolean;
}

const PANEL_KEYS: Record<string, PanelId> = {
  "1": "filter",
  "2": "review",
  "3": "insights",
  "4": "admin",
};

export default function useKeyboardShortcuts({
  onTogglePanel,
  onClosePanel,
  onCloseDetail,
  onFocusSearch,
  activePanel,
  hasDetail,
}: Options) {
  useEffect(() => {
    function handler(e: KeyboardEvent) {
      const tag = (e.target as HTMLElement).tagName;
      const inInput = tag === "INPUT" || tag === "TEXTAREA";

      // Escape works globally
      if (e.key === "Escape") {
        if (activePanel) {
          onClosePanel();
        } else if (hasDetail) {
          onCloseDetail();
        }
        return;
      }

      // Cmd+K works globally
      if (e.key === "k" && (e.metaKey || e.ctrlKey)) {
        e.preventDefault();
        onFocusSearch();
        return;
      }

      // Everything below is blocked when typing in an input
      if (inInput) return;

      if (e.key === "/") {
        e.preventDefault();
        onFocusSearch();
        return;
      }

      const panel = PANEL_KEYS[e.key];
      if (panel) {
        onTogglePanel(panel);
      }
    }

    window.addEventListener("keydown", handler);
    return () => window.removeEventListener("keydown", handler);
  }, [onTogglePanel, onClosePanel, onCloseDetail, onFocusSearch, activePanel, hasDetail]);
}
