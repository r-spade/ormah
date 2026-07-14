"""Mine whisper eval cases from the live DB's whisper_log (read-only).

Drafted labels come from live evidence (what whisper actually injected), so
they are inherently circular: they describe current behavior, not ground
truth. Every mined case therefore carries ``"provisional": true`` and does
NOT bind until a human reviews the drafts (edit the review file, then run
``ormah eval whisper import-labels``).

Output lives in ``eval/whisper/corpus/local/`` which is gitignored — mined
content is real user data and never leaves the machine.
"""
from __future__ import annotations

import json
import sqlite3
from collections import defaultdict
from pathlib import Path

_LOCAL_DIR = Path(__file__).parent / "corpus" / "local"


class MinerError(Exception):
    """Raised when the live DB cannot be mined (e.g. missing schema)."""

# Draft-label thresholds (evidence heuristics, not ground truth).
_DRAFT_INJECT_SCORE = 0.55   # injected + strong score → draft should_inject
_DRAFT_EXCLUDE_SCORE = 0.30  # not injected + weak score → draft should_not_inject

# Sampling: oversample the slices where the golden corpus shows weakness,
# plus silent prompts (silence-integrity is under-observed).
_SILENT_SHARE = 0.25
_WEAK_INTENTS = ("preference", "identity")
_WEAK_SHARE = 0.35
_MIN_CANDIDATES = 2
_MAX_MEMORIES_PER_CASE = 15


def _connect_ro(db_path: Path) -> sqlite3.Connection:
    conn = sqlite3.connect(f"file:{db_path}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _default_classify(prompt: str) -> str:
    return "general"


def _load_prompt_groups(conn: sqlite3.Connection) -> list[dict]:
    """Group whisper_log rows by (session_id, prompt_hash) into prompt records.

    Grouping by prompt_hash alone would merge the same prompt issued in
    different sessions into one case with an unrelated union of candidates;
    the session is part of a whisper call's identity.

    Only groups that also appear in ``whisper_decisions`` (matched on
    session_id + prompt_hash) are kept. ``whisper_log`` is ALSO written by
    deliberate-recall exposures (``_log_feedback_candidates``, was_injected=1),
    which are not whisper decisions; ``whisper_decisions`` has exactly one row
    per real whisper call, so the join filters recall traffic out. Rows that
    predate the ``whisper_decisions`` table simply don't match and are excluded.
    """
    has_decisions = conn.execute(
        "SELECT name FROM sqlite_master WHERE type='table' AND name='whisper_decisions'"
    ).fetchone()
    if has_decisions is None:
        raise MinerError(
            "whisper_decisions table not found in the live DB. It records one row per "
            "whisper call and is required to separate real whisper decisions from "
            "deliberate-recall exposures (both land in whisper_log). Upgrade ormah and "
            "start the server once so the schema is created, then re-run mining."
        )
    decision_keys = {
        (r["session_id"], r["prompt_hash"])
        for r in conn.execute(
            "SELECT DISTINCT session_id, prompt_hash FROM whisper_decisions"
        ).fetchall()
    }

    has_events = conn.execute(
        "SELECT 1 FROM sqlite_master WHERE type='table' AND name='retrieval_events'"
    ).fetchone() is not None
    if has_events:
        rows_sql = """
        SELECT
               wl.session_id, wl.prompt_hash,
               COALESCE(re.prompt_text, wl.prompt_text) AS prompt_text,
               COALESCE(re.space, wl.space) AS space,
               wl.node_id,
               MAX(wl.score) AS score, MAX(wl.was_injected) AS was_injected,
               MAX(wl.logged_at) AS logged_at
        FROM whisper_log wl
        LEFT JOIN retrieval_events re ON re.id = wl.retrieval_event_id
        WHERE COALESCE(re.prompt_text, wl.prompt_text) IS NOT NULL
          AND TRIM(COALESCE(re.prompt_text, wl.prompt_text)) != ''
        GROUP BY wl.session_id, wl.prompt_hash, wl.node_id
        """
    else:
        # Read-only mining can target a DB that has not yet been opened by the
        # upgraded server, so retain the legacy query until migration runs.
        rows_sql = """
        SELECT session_id, prompt_hash, prompt_text, space, node_id,
               MAX(score) AS score, MAX(was_injected) AS was_injected,
               MAX(logged_at) AS logged_at
        FROM whisper_log
        WHERE prompt_text IS NOT NULL AND TRIM(prompt_text) != ''
        GROUP BY session_id, prompt_hash, node_id
        """
    rows = conn.execute(rows_sql).fetchall()
    groups: dict[tuple[str, str], dict] = {}
    for r in rows:
        key = (r["session_id"], r["prompt_hash"])
        if key not in decision_keys:
            continue  # recall-exposure row or pre-whisper_decisions history
        g = groups.setdefault(key, {
            "prompt": r["prompt_text"],
            "space": r["space"],
            "nodes": {},
            "logged_at": r["logged_at"],
        })
        g["nodes"][r["node_id"]] = {
            "score": float(r["score"]),
            "was_injected": bool(r["was_injected"]),
        }
        if r["logged_at"] > g["logged_at"]:
            g["logged_at"] = r["logged_at"]
    out = []
    for g in groups.values():
        g["silent"] = not any(n["was_injected"] for n in g["nodes"].values())
        out.append(g)
    return out


def _snapshot_nodes(conn: sqlite3.Connection, node_ids: list[str]) -> dict[str, dict]:
    """Snapshot live nodes (title/content/type/tier/space/tags/confidence)."""
    if not node_ids:
        return {}
    marks = ",".join("?" for _ in node_ids)
    rows = conn.execute(
        f"SELECT id, type, tier, title, content, space, confidence FROM nodes WHERE id IN ({marks})",
        node_ids,
    ).fetchall()
    tags = defaultdict(list)
    for t in conn.execute(
        f"SELECT node_id, tag FROM node_tags WHERE node_id IN ({marks})", node_ids
    ).fetchall():
        tags[t["node_id"]].append(t["tag"])
    return {
        r["id"]: {
            "node_id": r["id"],
            "title": r["title"],
            "content": r["content"] or "",
            "type": r["type"],
            "tier": r["tier"],
            "tags": tags.get(r["id"], []),
            "space": r["space"],
            "confidence": float(r["confidence"] if r["confidence"] is not None else 1.0),
        }
        for r in rows
    }


def _stratified_sample(groups: list[dict], limit: int, classify) -> list[dict]:
    """Oversample silent prompts and weak intents; fill the rest by recency."""
    for g in groups:
        g["intent"] = classify(g["prompt"])

    eligible = [g for g in groups if len(g["nodes"]) >= _MIN_CANDIDATES or g["silent"]]
    eligible.sort(key=lambda g: g["logged_at"], reverse=True)

    silent = [g for g in eligible if g["silent"]]
    weak = [g for g in eligible if not g["silent"] and g["intent"] in _WEAK_INTENTS]
    rest = [g for g in eligible if not g["silent"] and g["intent"] not in _WEAK_INTENTS]

    picked: list[dict] = []
    picked += silent[: max(1, int(limit * _SILENT_SHARE))]
    picked += weak[: max(1, int(limit * _WEAK_SHARE))]
    for g in rest:
        if len(picked) >= limit:
            break
        picked.append(g)
    # Backfill if silent/weak pools were smaller than their shares.
    if len(picked) < limit:
        seen = {id(g) for g in picked}
        for g in eligible:
            if id(g) not in seen:
                picked.append(g)
                if len(picked) >= limit:
                    break
    return picked[:limit]


def _draft_expected(g: dict, snapshots: dict[str, dict]) -> dict:
    should_inject, should_not, may = [], [], []
    for nid, ev in g["nodes"].items():
        if nid not in snapshots:
            continue
        if ev["was_injected"] and ev["score"] >= _DRAFT_INJECT_SCORE:
            should_inject.append(nid)
        elif not ev["was_injected"] and ev["score"] < _DRAFT_EXCLUDE_SCORE:
            should_not.append(nid)
        else:
            may.append(nid)
    exp: dict = {"should_inject": should_inject, "should_suppress": g["silent"] and not should_inject}
    if may:
        exp["may_include"] = may
    if should_not:
        exp["should_not_inject"] = should_not
    return exp


def mine(db_path: Path, limit: int = 80, out_path: Path | None = None,
         classify=None) -> tuple[Path, Path]:
    """Mine up to *limit* cases from *db_path* (opened read-only).

    *classify* is an optional callable prompt→category used for stratified
    sampling (the CLI passes the engine's PromptClassifier).
    Returns (mined_jsonl_path, review_md_path).
    """
    out_path = out_path or (_LOCAL_DIR / "mined.jsonl")
    review_path = out_path.with_name("mined-review.md")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    conn = _connect_ro(db_path)
    try:
        groups = _load_prompt_groups(conn)
        picked = _stratified_sample(groups, limit, classify or _default_classify)

        cases, review_lines = [], []
        review_lines.append("# Mined whisper eval cases — review before labels bind\n")
        review_lines.append(
            "Drafted labels reflect what whisper ACTUALLY did (circular evidence), "
            "not ground truth. For each case: fix wrong drafts by editing "
            "`mined.jsonl` (same line number as the case), then run "
            "`ormah eval whisper import-labels` to clear the provisional flags.\n"
        )
        for i, g in enumerate(picked, start=1):
            # Deterministic truncation: keep injected candidates first, then by
            # score, then node_id for stability — never let dict iteration order
            # drop the actually-injected node from an over-long candidate set.
            ordered = sorted(
                g["nodes"].items(),
                key=lambda kv: (not kv[1]["was_injected"], -kv[1]["score"], kv[0]),
            )
            node_ids = [nid for nid, _ in ordered[:_MAX_MEMORIES_PER_CASE]]
            snapshots = _snapshot_nodes(conn, node_ids)
            if not snapshots:
                continue  # every involved node was deleted since logging
            exp = _draft_expected(g, snapshots)
            case_id = f"mined-{i:03d}"
            cases.append({
                "id": case_id,
                "space": g["space"],
                "provisional": True,
                "memories": list(snapshots.values()),
                "prompts": [{
                    "text": g["prompt"],
                    "category": g["intent"],
                    "expected": exp,
                    "notes": (
                        f"Mined from whisper_log (logged {g['logged_at'][:10]}); "
                        "labels drafted from live evidence — provisional until reviewed."
                    ),
                }],
            })
            review_lines.append(f"## {case_id} [{g['intent']}{', silent' if g['silent'] else ''}]")
            review_lines.append(f"**Prompt:** {g['prompt']}\n")
            for nid, ev in g["nodes"].items():
                snap = snapshots.get(nid)
                if not snap:
                    continue
                if nid in exp.get("should_inject", []):
                    draft = "INJECT"
                elif nid in exp.get("should_not_inject", []):
                    draft = "exclude"
                else:
                    draft = "may"
                review_lines.append(
                    f"- [{draft}] `{nid[:8]}` (score {ev['score']:.2f}"
                    f"{', injected' if ev['was_injected'] else ''}) {snap['title'] or snap['content'][:70]}"
                )
            if exp.get("should_suppress"):
                review_lines.append("- drafted SILENT (nothing was injected)")
            review_lines.append("")

        out_path.write_text(
            "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n" if cases else ""
        )
        review_path.write_text("\n".join(review_lines))
        return out_path, review_path
    finally:
        conn.close()


def import_labels(mined_path: Path | None = None) -> int:
    """Clear provisional flags on mined cases after human review.

    The review workflow edits ``mined.jsonl`` labels directly; running this
    command is the sign-off that makes them bind. Returns cases confirmed.
    """
    mined_path = mined_path or (_LOCAL_DIR / "mined.jsonl")
    if not mined_path.exists():
        raise FileNotFoundError(f"{mined_path} not found — run 'ormah eval whisper mine' first")
    cases = [json.loads(line) for line in mined_path.read_text().splitlines() if line.strip()]
    confirmed = 0
    for c in cases:
        if c.pop("provisional", False):
            confirmed += 1
    mined_path.write_text(
        "\n".join(json.dumps(c, ensure_ascii=False) for c in cases) + "\n" if cases else ""
    )
    return confirmed
