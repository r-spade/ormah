"""SQLite database connection management."""

from __future__ import annotations

import logging
import re
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    """Manages per-thread SQLite connections with WAL mode and serialized writes."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._all_conns: list[sqlite3.Connection] = []
        self._conns_lock = threading.Lock()
        self._lock = threading.RLock()  # serializes write transactions across threads

    def _new_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(
            str(self.db_path),
            check_same_thread=False,
            timeout=10,
            isolation_level=None,
        )
        conn.row_factory = sqlite3.Row
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA foreign_keys=ON")
        conn.execute("PRAGMA synchronous=NORMAL")
        conn.execute("PRAGMA busy_timeout=5000")
        # sqlite-vec is loaded per connection: a connection without it cannot
        # query the node_vectors (vec0) virtual table.
        try:
            import sqlite_vec

            conn.enable_load_extension(True)
            sqlite_vec.load(conn)
            conn.enable_load_extension(False)
        except ImportError:
            pass  # sqlite-vec not installed — vector search disabled
        with self._conns_lock:
            self._all_conns.append(conn)
        return conn

    @property
    def conn(self) -> sqlite3.Connection:
        existing = getattr(self._local, "conn", None)
        if existing is None:
            existing = self._new_connection()
            self._local.conn = existing
        return existing

    @contextmanager
    def transaction(self):
        """Serialize write transactions across threads.

        Reentrant per thread: only the outermost call on a given thread issues
        BEGIN/COMMIT/ROLLBACK. Inner (nested) calls are pass-throughs.
        """
        with self._lock:
            depth = getattr(self._local, "tx_depth", 0) + 1
            self._local.tx_depth = depth
            try:
                if depth == 1:
                    self.conn.execute("BEGIN IMMEDIATE")
                yield self.conn
                if depth == 1:
                    self.conn.execute("COMMIT")
            except BaseException:
                if depth == 1:
                    self.conn.execute("ROLLBACK")
                raise
            finally:
                self._local.tx_depth = depth - 1

    def init_schema(self) -> None:
        """Create tables from schema.sql."""
        schema = _SCHEMA_PATH.read_text(encoding="utf-8")
        # executescript issues its own implicit COMMIT, safe outside transaction
        self.conn.executescript(schema)
        self._migrate()

    @staticmethod
    def insert_retrieval_event(
        conn: sqlite3.Connection,
        *,
        surface: str,
        session_id: str,
        space: str | None,
        prompt_hash: str,
        prompt_text: str,
        prompt_vec: bytes,
        logged_at: str,
    ) -> int:
        """Insert one prompt payload shared by its candidate log rows."""
        cursor = conn.execute(
            """
            INSERT INTO retrieval_events
                (surface, session_id, space, prompt_hash, prompt_text, prompt_vec, logged_at)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            (
                surface,
                session_id,
                space,
                prompt_hash,
                prompt_text,
                prompt_vec,
                logged_at,
            ),
        )
        return int(cursor.lastrowid)

    def _migrate(self) -> None:
        """Run migrations for existing databases."""
        with self.transaction() as conn:
            # Add reason column to edges table if missing
            edge_cols = [
                row[1]
                for row in conn.execute("PRAGMA table_info(edges)").fetchall()
            ]
            if "reason" not in edge_cols:
                conn.execute("ALTER TABLE edges ADD COLUMN reason TEXT")

            # Add enrichment columns to nodes table if missing
            node_cols = [
                row[1]
                for row in conn.execute("PRAGMA table_info(nodes)").fetchall()
            ]
            enrichment_migrations = [
                ("confidence", "ALTER TABLE nodes ADD COLUMN confidence REAL DEFAULT 1.0"),
                ("importance", "ALTER TABLE nodes ADD COLUMN importance REAL DEFAULT 0.5"),
                ("valid_until", "ALTER TABLE nodes ADD COLUMN valid_until TEXT"),
                ("stability", "ALTER TABLE nodes ADD COLUMN stability REAL DEFAULT 1.0"),
                ("last_review", "ALTER TABLE nodes ADD COLUMN last_review TEXT"),
            ]
            for col_name, ddl in enrichment_migrations:
                if col_name not in node_cols:
                    conn.execute(ddl)

            if "seq" not in node_cols:
                conn.execute("ALTER TABLE nodes ADD COLUMN seq INTEGER NOT NULL DEFAULT 0")
                # Backfill existing rows: oldest (created ASC) gets the lowest seq,
                # so the historical backlog drains oldest-first.
                rows = conn.execute(
                    "SELECT id FROM nodes ORDER BY created ASC, rowid ASC"
                ).fetchall()
                for i, row in enumerate(rows, start=1):
                    conn.execute("UPDATE nodes SET seq = ? WHERE id = ?", (i, row[0]))
                # Initialize the durable monotonic counter past the backfilled max,
                # so future writes always allocate seq above any current watermark.
                conn.execute(
                    "INSERT OR REPLACE INTO meta (key, value) VALUES ('node_seq_next', ?)",
                    (str(len(rows) + 1),),
                )

            # Index created here (not in schema.sql): on a legacy DB schema.sql's
            # executescript runs before the column exists. Unconditional so a fresh DB
            # (which skips the block above) still gets the index.
            conn.execute("CREATE INDEX IF NOT EXISTS idx_nodes_seq ON nodes(seq)")

            # Create new feedback/logging tables if missing
            existing_tables = {
                row[0]
                for row in conn.execute(
                    "SELECT name FROM sqlite_master WHERE type='table'"
                ).fetchall()
            }

            if "whisper_log" not in existing_tables:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS whisper_log (
                        id                  INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id          TEXT NOT NULL,
                        space               TEXT,
                        prompt_hash         TEXT NOT NULL,
                        prompt_text         TEXT,
                        prompt_vec          BLOB NOT NULL,
                        node_id             TEXT NOT NULL,
                        score               REAL NOT NULL,
                        retrieval_score     REAL,
                        raw_cosine          REAL,
                        cross_encoder_score REAL,
                        ce_absolute         REAL,
                        gate_score          REAL,
                        source              TEXT,
                        retrieval_rank      INTEGER,
                        final_rank          INTEGER,
                        decision_stage      TEXT NOT NULL DEFAULT 'legacy',
                        was_injected        INTEGER NOT NULL,
                        logged_at           TEXT NOT NULL,
                        retrieval_event_id  INTEGER REFERENCES retrieval_events(id)
                            ON DELETE RESTRICT
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_whisper_log_session ON whisper_log(session_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_whisper_log_node ON whisper_log(node_id)"
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_whisper_log_logged ON whisper_log(logged_at)"
                )

            self._migrate_whisper_log_schema(conn)
            self._migrate_retrieval_event_schema(conn)

            self._migrate_affinity_schema(conn, existing_tables)
            self._ensure_signals_schema(conn)

            if "review_log" not in existing_tables:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS review_log (
                        id          INTEGER PRIMARY KEY AUTOINCREMENT,
                        node_id     TEXT NOT NULL,
                        session_id  TEXT NOT NULL,
                        surfaced_at TEXT NOT NULL,
                        answered    INTEGER DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_review_log_node ON review_log(node_id)"
                )

        # Migrate FTS table to porter stemmer if needed
        self._migrate_fts_tokenizer()

    def _migrate_whisper_log_schema(self, conn: sqlite3.Connection) -> None:
        """Add candidate-stage diagnostics without rebuilding feedback history."""
        cols = {row[1] for row in conn.execute("PRAGMA table_info(whisper_log)").fetchall()}
        additions = {
            "retrieval_score": "REAL",
            "raw_cosine": "REAL",
            "cross_encoder_score": "REAL",
            "ce_absolute": "REAL",
            "gate_score": "REAL",
            "source": "TEXT",
            "retrieval_rank": "INTEGER",
            "final_rank": "INTEGER",
            "decision_stage": "TEXT NOT NULL DEFAULT 'legacy'",
        }
        for name, column_type in additions.items():
            if name not in cols:
                conn.execute(f"ALTER TABLE whisper_log ADD COLUMN {name} {column_type}")

    def _migrate_retrieval_event_schema(self, conn: sqlite3.Connection) -> None:
        """Normalize prompt payloads without rebuilding exact feedback rows."""
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS retrieval_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                surface     TEXT NOT NULL,
                session_id  TEXT NOT NULL,
                space       TEXT,
                prompt_hash TEXT NOT NULL,
                prompt_text TEXT,
                prompt_vec  BLOB NOT NULL,
                logged_at   TEXT NOT NULL
            )
            """
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_events_session "
            "ON retrieval_events(session_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_retrieval_events_logged "
            "ON retrieval_events(logged_at)"
        )

        cols = {row[1] for row in conn.execute("PRAGMA table_info(whisper_log)").fetchall()}
        if "retrieval_event_id" not in cols:
            conn.execute(
                "ALTER TABLE whisper_log ADD COLUMN retrieval_event_id INTEGER "
                "REFERENCES retrieval_events(id) ON DELETE RESTRICT"
            )

        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_whisper_log_event "
            "ON whisper_log(retrieval_event_id)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_whisper_log_retention "
            "ON whisper_log(was_injected, logged_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_whisper_log_session_injected "
            "ON whisper_log(session_id, was_injected)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_whisper_log_node_injected_logged "
            "ON whisper_log(node_id, was_injected, logged_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_whisper_log_event_key "
            "ON whisper_log(session_id, prompt_hash, logged_at)"
        )

        # Older schemas duplicated the same prompt payload on every candidate.
        # Backfill one parent per writer event, preserve candidate IDs, then
        # release the duplicate blobs for SQLite to reuse. No VACUUM runs on
        # startup: compacting the file is an explicit operator choice.
        groups = conn.execute(
            """
            SELECT
                session_id, prompt_hash, logged_at,
                MAX(space) AS space,
                MAX(prompt_text) AS prompt_text,
                MIN(id) AS sample_id
            FROM whisper_log
            WHERE retrieval_event_id IS NULL
            GROUP BY session_id, prompt_hash, logged_at
            """
        ).fetchall()
        for group in groups:
            sample = conn.execute(
                "SELECT prompt_vec FROM whisper_log WHERE id = ?",
                (group["sample_id"],),
            ).fetchone()
            cursor = conn.execute(
                """
                INSERT INTO retrieval_events
                    (surface, session_id, space, prompt_hash, prompt_text, prompt_vec, logged_at)
                VALUES ('legacy', ?, ?, ?, ?, ?, ?)
                """,
                (
                    group["session_id"],
                    group["space"],
                    group["prompt_hash"],
                    group["prompt_text"],
                    sample["prompt_vec"] if sample is not None else b"",
                    group["logged_at"],
                ),
            )
            conn.execute(
                """
                UPDATE whisper_log
                SET retrieval_event_id = ?, prompt_text = NULL, prompt_vec = X''
                WHERE retrieval_event_id IS NULL
                  AND session_id = ? AND prompt_hash = ? AND logged_at = ?
                """,
                (
                    int(cursor.lastrowid),
                    group["session_id"],
                    group["prompt_hash"],
                    group["logged_at"],
                ),
            )

    def _create_affinity_table(self, conn: sqlite3.Connection, table: str = "affinity") -> None:
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS {table} (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                prompt_vec     BLOB NOT NULL,
                prompt_text    TEXT,
                node_id        TEXT NOT NULL,
                signal         INTEGER NOT NULL,
                source         TEXT NOT NULL DEFAULT 'explicit',
                confirmed_at   TEXT NOT NULL,
                space          TEXT,
                session_id     TEXT NOT NULL,
                whisper_log_id INTEGER REFERENCES whisper_log(id) ON DELETE SET NULL
            )
            """
        )

    def _ensure_affinity_indexes(self, conn: sqlite3.Connection) -> None:
        conn.execute("CREATE INDEX IF NOT EXISTS idx_affinity_node ON affinity(node_id)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_affinity_whisper_log "
            "ON affinity(whisper_log_id)"
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_affinity_node_whisper_log_unique
            ON affinity(node_id, whisper_log_id)
            WHERE whisper_log_id IS NOT NULL
            """
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_affinity_node_session_legacy_unique
            ON affinity(node_id, session_id)
            WHERE whisper_log_id IS NULL
            """
        )

    def _migrate_affinity_schema(
        self,
        conn: sqlite3.Connection,
        existing_tables: set[str],
    ) -> None:
        if "affinity" not in existing_tables:
            self._create_affinity_table(conn)
            self._ensure_affinity_indexes(conn)
            return

        cols = [row[1] for row in conn.execute("PRAGMA table_info(affinity)").fetchall()]
        row = conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='affinity'"
        ).fetchone()
        create_sql = (row[0] or "").lower() if row else ""
        has_session_unique = "unique (node_id, session_id)" in create_sql

        if "whisper_log_id" not in cols or has_session_unique:
            logger.info("Migrating affinity table to turn-level whisper_log_id keys")
            conn.execute("DROP TABLE IF EXISTS affinity_new")
            self._create_affinity_table(conn, "affinity_new")
            conn.execute(
                """
                INSERT INTO affinity_new
                    (
                        id, prompt_vec, prompt_text, node_id, signal, source,
                        confirmed_at, space, session_id, whisper_log_id
                    )
                SELECT
                    id, prompt_vec, prompt_text, node_id, signal, source,
                    confirmed_at, space, session_id, NULL
                FROM affinity
                """
            )
            conn.execute("DROP TABLE affinity")
            conn.execute("ALTER TABLE affinity_new RENAME TO affinity")

        self._ensure_affinity_indexes(conn)

    def _ensure_signals_schema(self, conn: sqlite3.Connection) -> None:
        conn.execute(
            """
            CREATE TABLE IF NOT EXISTS signals (
                id             INTEGER PRIMARY KEY AUTOINCREMENT,
                whisper_log_id INTEGER REFERENCES whisper_log(id) ON DELETE SET NULL,
                node_id        TEXT NOT NULL,
                signal_type    TEXT NOT NULL,
                polarity       INTEGER NOT NULL,
                strength       REAL NOT NULL DEFAULT 1.0,
                source         TEXT NOT NULL,
                session_id     TEXT,
                agent_id       TEXT,
                surface        TEXT,
                space          TEXT,
                prompt_hash    TEXT,
                evidence       TEXT,
                created        TEXT NOT NULL
            )
            """
        )
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_node ON signals(node_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_session ON signals(session_id)")
        conn.execute("CREATE INDEX IF NOT EXISTS idx_signals_created ON signals(created)")
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_signals_whisper_log ON signals(whisper_log_id)"
        )
        conn.execute(
            """
            CREATE UNIQUE INDEX IF NOT EXISTS idx_signals_whisper_type_source_unique
            ON signals(whisper_log_id, signal_type, source)
            WHERE whisper_log_id IS NOT NULL
            """
        )

    def _migrate_fts_tokenizer(self) -> None:
        """Recreate FTS table with porter stemmer if it uses the old tokenizer."""
        row = self.conn.execute(
            "SELECT sql FROM sqlite_master WHERE type='table' AND name='nodes_fts'"
        ).fetchone()
        if row is None:
            return  # table doesn't exist yet, schema.sql will create it
        create_sql = row[0] or ""
        if "porter" in create_sql.lower():
            return  # already using porter tokenizer
        logger.info("Migrating FTS table to porter stemmer")
        with self.transaction() as conn:
            conn.execute("DROP TABLE IF EXISTS nodes_fts")
            conn.execute(
                "CREATE VIRTUAL TABLE nodes_fts USING fts5("
                "id UNINDEXED, title, content, tags, "
                "tokenize='porter unicode61')"
            )
            # Mark that a full FTS rebuild is needed
            conn.execute(
                "INSERT OR REPLACE INTO meta (key, value) VALUES ('fts_needs_rebuild', '1')"
            )

    def init_vec_table(self, dim: int = 768, *, allow_drop: bool = False) -> None:
        """Create the sqlite-vec virtual table. Requires sqlite-vec extension.

        The stored dimension is read from the table DDL in sqlite_master, so an
        EMPTY mismatched table is detected too (a row probe cannot see it). On a
        mismatch: an empty table is dropped and recreated at *dim*; a POPULATED
        table is protected — RuntimeError — unless ``allow_drop=True`` (a
        deliberate embedding-model migration). The caller (engine startup) is
        responsible for re-embedding after any authorized drop; the
        embedding_backfill job first fires ~10s after boot, so recovery is
        automatic.
        """
        try:
            import sqlite_vec

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)
        except ImportError:
            return  # sqlite-vec not available, vector search disabled

        # The whole read → decide → drop → mark → create sequence runs inside a
        # SINGLE BEGIN IMMEDIATE. transaction() takes the write lock (via the
        # internal RLock across threads, and via SQLite's write lock across
        # processes), so a concurrent initializer can no longer read the marker
        # as absent, THEN drop after another initializer already consumed it —
        # closing the check-then-drop TOCTOU (council: codex high). A concurrent
        # caller blocks here, then re-reads the marker itself once serialized.
        with self.transaction() as conn:
            row = conn.execute(
                "SELECT sql FROM sqlite_master WHERE type='table' AND name='node_vectors'"
            ).fetchone()
            if row is not None:
                match = re.search(r"FLOAT\[(\d+)\]", row[0] or "")
                if match is None:
                    # Corrupt/foreign schema: fail closed — never guess-drop, and never
                    # boot into broken vector search (MATCH would fail at runtime).
                    raise RuntimeError(
                        "node_vectors exists but its DDL has no FLOAT[dim] — "
                        "unsupported or corrupt schema. Rows were preserved; inspect "
                        "the table, then restore a vec0 table (or move this one aside "
                        "and let the embedding backfill re-embed)."
                    )
                existing_dim = int(match.group(1))
                if existing_dim != dim:
                    count = conn.execute(
                        "SELECT count(*) FROM node_vectors"
                    ).fetchone()[0]
                    if count > 0:
                        consumed = conn.execute(
                            "SELECT value FROM meta WHERE key = 'reindex_consumed_dim'"
                        ).fetchone()
                        already_consumed = (
                            consumed is not None and consumed["value"] == str(dim)
                        )
                        if not allow_drop:
                            raise RuntimeError(
                                f"Embedding dimension mismatch: configured "
                                f"ORMAH_EMBEDDING_DIM={dim} but node_vectors holds "
                                f"{count} vectors of dim {existing_dim}. Refusing to "
                                f"DROP them. If this is a deliberate embedding-model "
                                f"change, set ORMAH_REINDEX_ON_DIM_CHANGE={dim} for one "
                                f"boot (remove it afterwards); otherwise fix "
                                f"ORMAH_EMBEDDING_DIM — the code default is 768."
                            )
                        if already_consumed:
                            raise RuntimeError(
                                f"Embedding dimension mismatch: a reindex to dim {dim} "
                                f"was already performed and the ORMAH_REINDEX_ON_DIM_CHANGE "
                                f"authorization is spent. node_vectors now holds {count} "
                                f"vectors of dim {existing_dim} — remove the stale "
                                f"ORMAH_REINDEX_ON_DIM_CHANGE from your config. If a second "
                                f"deliberate reindex to dim {dim} is genuinely needed, clear "
                                f"the marker first: DELETE FROM meta WHERE "
                                f"key='reindex_consumed_dim'."
                            )
                        logger.info(
                            "Recreating vec table (%d → %d, %d vectors dropped)",
                            existing_dim, dim, count,
                        )
                        conn.execute("DROP TABLE node_vectors")
                        conn.execute(
                            "INSERT OR REPLACE INTO meta (key, value) VALUES "
                            "('reindex_consumed_dim', ?)",
                            (str(dim),),
                        )
                    else:
                        conn.execute("DROP TABLE node_vectors")

            conn.execute(
                f"CREATE VIRTUAL TABLE IF NOT EXISTS node_vectors USING vec0("
                f"id TEXT PRIMARY KEY, embedding FLOAT[{dim}])"
            )

    def close(self) -> None:
        with self._conns_lock:
            for conn in self._all_conns:
                try:
                    conn.close()
                except Exception:  # noqa: BLE001
                    pass
            self._all_conns.clear()
        self._local = threading.local()
