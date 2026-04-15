"""SQLite database connection management."""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path

logger = logging.getLogger(__name__)

_SCHEMA_PATH = Path(__file__).parent / "schema.sql"


class Database:
    """Manages a single SQLite connection with WAL mode and serialized writes."""

    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn: sqlite3.Connection | None = None
        self._lock = threading.RLock()
        self._tx_depth = 0

    @property
    def conn(self) -> sqlite3.Connection:
        if self._conn is None:
            self._conn = sqlite3.connect(
                str(self.db_path),
                check_same_thread=False,
                timeout=10,
                isolation_level=None,
            )
            self._conn.row_factory = sqlite3.Row
            self._conn.execute("PRAGMA journal_mode=WAL")
            self._conn.execute("PRAGMA foreign_keys=ON")
            self._conn.execute("PRAGMA synchronous=NORMAL")
            self._conn.execute("PRAGMA busy_timeout=5000")
        return self._conn

    @contextmanager
    def transaction(self):
        """Serialize write transactions across threads.

        Reentrant: only the outermost call issues BEGIN/COMMIT/ROLLBACK.
        Inner (nested) calls are pass-throughs.
        """
        self._lock.acquire()
        self._tx_depth += 1
        try:
            if self._tx_depth == 1:
                self.conn.execute("BEGIN IMMEDIATE")
            yield self.conn
            if self._tx_depth == 1:
                self.conn.execute("COMMIT")
        except BaseException:
            if self._tx_depth == 1:
                self.conn.execute("ROLLBACK")
            raise
        finally:
            self._tx_depth -= 1
            self._lock.release()

    def init_schema(self) -> None:
        """Create tables from schema.sql."""
        schema = _SCHEMA_PATH.read_text(encoding="utf-8")
        # executescript issues its own implicit COMMIT, safe outside transaction
        self.conn.executescript(schema)
        self._migrate()

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
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        session_id   TEXT NOT NULL,
                        space        TEXT,
                        prompt_hash  TEXT NOT NULL,
                        prompt_text  TEXT,
                        prompt_vec   BLOB NOT NULL,
                        node_id      TEXT NOT NULL,
                        score        REAL NOT NULL,
                        was_injected INTEGER NOT NULL,
                        logged_at    TEXT NOT NULL
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

            if "affinity" not in existing_tables:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS affinity (
                        id           INTEGER PRIMARY KEY AUTOINCREMENT,
                        prompt_vec   BLOB NOT NULL,
                        prompt_text  TEXT,
                        node_id      TEXT NOT NULL,
                        signal       INTEGER NOT NULL,
                        source       TEXT NOT NULL DEFAULT 'explicit',
                        confirmed_at TEXT NOT NULL,
                        space        TEXT,
                        session_id   TEXT NOT NULL,
                        UNIQUE (node_id, session_id)
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_affinity_node ON affinity(node_id)"
                )

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

            if "decay_alert_log" not in existing_tables:
                conn.execute(
                    """
                    CREATE TABLE IF NOT EXISTS decay_alert_log (
                        id             INTEGER PRIMARY KEY AUTOINCREMENT,
                        node_id        TEXT NOT NULL,
                        alerted_at     TEXT NOT NULL,
                        retrievability REAL NOT NULL,
                        tier           TEXT NOT NULL,
                        acknowledged   INTEGER DEFAULT 0
                    )
                    """
                )
                conn.execute(
                    "CREATE INDEX IF NOT EXISTS idx_decay_alert_node ON decay_alert_log(node_id)"
                )

        # Migrate FTS table to porter stemmer if needed
        self._migrate_fts_tokenizer()

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

    def init_vec_table(self, dim: int = 768) -> None:
        """Create the sqlite-vec virtual table. Requires sqlite-vec extension.

        If the existing table has a different dimension than *dim*, it is
        dropped and recreated.  The caller (engine startup) is responsible
        for re-embedding all nodes afterwards.
        """
        try:
            import sqlite_vec

            self.conn.enable_load_extension(True)
            sqlite_vec.load(self.conn)
            self.conn.enable_load_extension(False)

            # Check for dimension mismatch on an existing table
            existing = self.conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table' AND name='node_vectors'"
            ).fetchone()
            if existing:
                try:
                    row = self.conn.execute(
                        "SELECT embedding FROM node_vectors LIMIT 1"
                    ).fetchone()
                    if row is not None:
                        import struct

                        blob = row[0]
                        existing_dim = len(blob) // struct.calcsize("f")
                        if existing_dim != dim:
                            logger.info(
                                "Embedding dimension changed (%d → %d), recreating vec table",
                                existing_dim,
                                dim,
                            )
                            with self.transaction() as conn:
                                conn.execute("DROP TABLE node_vectors")
                except Exception:
                    pass  # empty table or parse error — just ensure it exists

            with self.transaction() as conn:
                conn.execute(
                    f"CREATE VIRTUAL TABLE IF NOT EXISTS node_vectors USING vec0("
                    f"id TEXT PRIMARY KEY, embedding FLOAT[{dim}])"
                )
        except ImportError:
            pass  # sqlite-vec not available, vector search disabled

    def close(self) -> None:
        if self._conn is not None:
            self._conn.close()
            self._conn = None
