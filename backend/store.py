"""Corpus store: article metadata in SQLite, embeddings as one contiguous row-major N x 384 float32 matrix.

Row i of the matrix is article id i, so the engine's output indices map straight to articles.
"""
import os
import sqlite3
import threading

import numpy as np

from .config import DATA_DIR, DB_PATH, EMB_PATH, DIM

SCHEMA = """
CREATE TABLE IF NOT EXISTS articles (
    id INTEGER PRIMARY KEY,
    dedup_key TEXT UNIQUE NOT NULL,
    domain TEXT NOT NULL,
    title TEXT NOT NULL,
    summary TEXT NOT NULL,
    url TEXT NOT NULL,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    origin TEXT NOT NULL DEFAULT 'live'
);
CREATE TABLE IF NOT EXISTS ingest_log (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    feeds_ok INTEGER, feeds_failed INTEGER,
    seen INTEGER, exact_dups INTEGER, added INTEGER, seconds REAL
);
"""


class Store:
    def __init__(self):
        os.makedirs(DATA_DIR, exist_ok=True)
        self.lock = threading.RLock()
        self.db = sqlite3.connect(DB_PATH, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)
        self._load_matrix()

    def _load_matrix(self):
        n_rows = self.db.execute("SELECT COUNT(*) FROM articles").fetchone()[0]
        if os.path.exists(EMB_PATH):
            flat = np.fromfile(EMB_PATH, dtype=np.float32)
            emb = flat.reshape(-1, DIM)
        else:
            emb = np.zeros((0, DIM), dtype=np.float32)
        if emb.shape[0] != n_rows:  # crash between DB commit and file append: trust the shorter one
            n = min(n_rows, emb.shape[0])
            self.db.execute("DELETE FROM articles WHERE id >= ?", (n,))
            self.db.commit()
            emb = emb[:n]
            emb.tofile(EMB_PATH)
        self.emb = np.ascontiguousarray(emb)

    @property
    def size(self):
        return self.emb.shape[0]

    def existing_keys(self):
        return {r[0] for r in self.db.execute("SELECT dedup_key FROM articles")}

    def add(self, rows, vectors):
        """rows: list of dicts (no id); vectors: (n, 384) unit-normalised float32."""
        if not rows:
            return 0
        with self.lock:
            start = self.size
            with self.db:
                for j, r in enumerate(rows):
                    self.db.execute(
                        "INSERT INTO articles (id, dedup_key, domain, title, summary, url, published_at, fetched_at, origin)"
                        " VALUES (?,?,?,?,?,?,?,?,?)",
                        (start + j, r["dedup_key"], r["domain"], r["title"], r["summary"], r["url"],
                         r.get("published_at"), r["fetched_at"], r.get("origin", "live")),
                    )
            vectors = np.ascontiguousarray(vectors, dtype=np.float32)
            with open(EMB_PATH, "ab") as f:
                vectors.tofile(f)
            self.emb = np.ascontiguousarray(np.vstack([self.emb, vectors]))
            return len(rows)

    def articles(self, ids):
        ids = [int(i) for i in ids]
        if not ids:
            return {}
        q = "SELECT * FROM articles WHERE id IN (%s)" % ",".join("?" * len(ids))
        return {r["id"]: dict(r) for r in self.db.execute(q, ids)}

    def log_ingest(self, **kw):
        with self.db:
            self.db.execute(
                "INSERT INTO ingest_log (at, feeds_ok, feeds_failed, seen, exact_dups, added, seconds)"
                " VALUES (:at, :feeds_ok, :feeds_failed, :seen, :exact_dups, :added, :seconds)", kw)

    def stats(self):
        by_domain = [dict(r) for r in self.db.execute(
            "SELECT domain, COUNT(*) AS n, MAX(published_at) AS latest FROM articles GROUP BY domain ORDER BY n DESC")]
        log = [dict(r) for r in self.db.execute("SELECT * FROM ingest_log ORDER BY id DESC LIMIT 10")]
        recent = [dict(r) for r in self.db.execute(
            "SELECT id, domain, title, url, published_at FROM articles ORDER BY id DESC LIMIT 25")]
        return {
            "articles": self.size,
            "matrix_bytes": int(self.emb.nbytes),
            "domains": by_domain,
            "ingest_log": log,
            "recent": recent,
        }
