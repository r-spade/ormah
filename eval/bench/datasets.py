"""Public dataset adapters; bounded-memory iteration over JSON arrays."""

from __future__ import annotations

import hashlib
import json
import re
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path

SOURCES = {
    "longmemeval": (
        "longmemeval_s.json",
        "https://huggingface.co/datasets/xiaowu0162/longmemeval/resolve/main/longmemeval_s_cleaned.json",
    ),
    "locomo": (
        "locomo10.json",
        "https://raw.githubusercontent.com/snap-research/locomo/main/data/locomo10.json",
    ),
}


@dataclass
class Turn:
    speaker: str
    text: str
    turn_id: str


@dataclass
class Session:
    session_id: str
    date: str
    turns: list[Turn]


@dataclass
class Question:
    question_id: str
    dataset: str
    question: str
    gold: str
    question_type: str
    question_date: str
    gold_ids: list[str]
    abstention: bool = False
    conversation_id: str | None = None
    sessions: list[Session] = field(default_factory=list)

    def metadata(self) -> dict:
        # Never deep-copy the haystack just to discard it.
        return {k: v for k, v in vars(self).items() if k != "sessions"}


def sha256_file(path: Path) -> str:
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def iter_array(path: Path, chunk_size: int = 1024 * 1024):
    """Decode one top-level item at a time, releasing previous haystacks."""
    decoder = json.JSONDecoder()
    with path.open(encoding="utf-8") as f:
        buffer = ""
        eof = False

        def fill():
            nonlocal buffer, eof
            chunk = f.read(chunk_size)
            eof = not chunk
            buffer += chunk

        fill()
        buffer = buffer.lstrip()
        if not buffer.startswith("["):
            raise ValueError("Dataset must be a JSON array")
        buffer = buffer[1:]
        first = True
        while True:
            buffer = buffer.lstrip()
            while not buffer and not eof:
                fill()
                buffer = buffer.lstrip()
            if buffer.startswith("]"):
                if buffer[1:].strip() or f.read().strip():
                    raise ValueError("Trailing data after dataset")
                return
            if not first:
                if not buffer.startswith(","):
                    raise ValueError("Missing comma or truncated dataset")
                buffer = buffer[1:].lstrip()
            while True:
                try:
                    item, end = decoder.raw_decode(buffer)
                    break
                except json.JSONDecodeError:
                    if eof:
                        raise
                    fill()
                    buffer = buffer.lstrip()
            buffer = buffer[end:]
            yield item
            del item
            first = False


def parse_date(value: str) -> str:
    value = value.strip()
    try:
        dt = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        for fmt in (
            "%Y/%m/%d (%a) %H:%M",
            "%d %B %Y",
            "%I:%M %p on %d %B, %Y",
            "%H:%M on %d %B, %Y",
        ):
            try:
                dt = datetime.strptime(value, fmt)
                break
            except ValueError:
                continue
        else:
            raise ValueError(f"Unsupported dataset date: {value!r}") from None
    return dt.replace(tzinfo=dt.tzinfo or timezone.utc).astimezone(timezone.utc).isoformat()


def load_questions(path: Path, dataset: str):
    for ci, entry in enumerate(iter_array(path)):
        if dataset == "longmemeval":
            sessions = []
            for sid, date, turns in zip(
                entry["haystack_session_ids"],
                entry["haystack_dates"],
                entry["haystack_sessions"],
                strict=True,
            ):
                sessions.append(
                    Session(
                        str(sid),
                        parse_date(date),
                        [Turn(t["role"], t["content"], f"{sid}:{i}") for i, t in enumerate(turns)],
                    )
                )
            yield Question(
                str(entry["question_id"]),
                dataset,
                entry["question"],
                str(entry["answer"]),
                entry["question_type"],
                parse_date(entry["question_date"]),
                list(entry["answer_session_ids"]),
                str(entry["question_id"]).endswith("_abs"),
                sessions=sessions,
            )
        elif dataset == "locomo":
            conv = entry["conversation"]
            sessions = []
            keys = sorted(
                (k for k in conv if re.fullmatch(r"session_\d+", k)),
                key=lambda k: int(k.split("_")[1]),
            )
            for key in keys:
                sessions.append(
                    Session(
                        f"locomo:{ci}:{key}",
                        parse_date(conv[key + "_date_time"]),
                        [Turn(t["speaker"], t["text"], t["dia_id"]) for t in conv[key]],
                    )
                )
            for qi, qa in enumerate(entry["qa"]):
                category = int(qa["category"])
                gold = str(qa.get("answer", ""))
                if category == 3:
                    gold = gold.split(";", 1)[0].strip()
                yield Question(
                    f"locomo:{ci}:{qi}",
                    dataset,
                    qa["question"],
                    gold,
                    str(category),
                    max(s.date for s in sessions),
                    list(qa.get("evidence", [])),
                    category == 5,
                    str(ci),
                    sessions,
                )
        else:
            raise ValueError(f"Unknown dataset {dataset}")
        del entry, sessions


def download(data_dir: Path, dataset: str = "all") -> dict:
    data_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    for name in SOURCES if dataset == "all" else [dataset]:
        filename, url = SOURCES[name]
        path = data_dir / filename
        if not path.exists():
            tmp = path.with_suffix(".part")
            with urllib.request.urlopen(url, timeout=120) as response, tmp.open("wb") as out:
                while chunk := response.read(1024 * 1024):
                    out.write(chunk)
            tmp.replace(path)
        manifest[name] = {
            "file": filename,
            "url": url,
            "sha256": sha256_file(path),
            "bytes": path.stat().st_size,
        }
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def session_dict(session: Session) -> dict:
    return asdict(session)
