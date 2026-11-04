"""Verification history (ported from the team's alternate prototype): every verdict is saved with its full result,
so past checks can be listed, searched, reopened exactly as they were shown, and re-run against today's corpus."""
import json
import os
import sqlite3
import threading
from datetime import datetime, timezone

from .config import DATA_DIR

HISTORY_PATH = os.path.join(DATA_DIR, "history.sqlite")

SCHEMA = """
CREATE TABLE IF NOT EXISTS verifications (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    at TEXT NOT NULL,
    corpus TEXT NOT NULL,
    claim TEXT NOT NULL,
    verdict TEXT NOT NULL,
    S REAL, D REAL,
    clusters INTEGER,
    corpus_size INTEGER,
    latency_ms REAL,
    web_added INTEGER,
    fever_correct INTEGER,
    result_json TEXT NOT NULL
);
CREATE INDEX IF NOT EXISTS verifications_at ON verifications (at);
"""
SUMMARY_COLS = "id, at, corpus, claim, verdict, S, D, clusters, corpus_size, latency_ms, web_added, fever_correct"


class History:
    def __init__(self, path=HISTORY_PATH):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        self.lock = threading.Lock()
        self.db = sqlite3.connect(path, check_same_thread=False)
        self.db.row_factory = sqlite3.Row
        self.db.executescript(SCHEMA)

    def add(self, res):
        if res.get("syndication_test"):  # stress-test runs are simulations, not checks worth keeping
            return None
        fv, web = res.get("fever"), res.get("web")
        row = (datetime.now(timezone.utc).isoformat(timespec="seconds"), res.get("corpus", "live"), res["claim"],
               res["verdict"], res.get("S"), res.get("D"), len(res.get("evidence", [])),
               res["retrieve_wide"]["corpus_size"], res["timings_ms"].get("total"),
               web["added"] if web else None, (1 if fv["correct"] else 0) if fv else None, json.dumps(res))
        with self.lock, self.db:
            cur = self.db.execute(
                f"INSERT INTO verifications ({SUMMARY_COLS.replace('id, ', '', 1)}, result_json)"
                " VALUES (?,?,?,?,?,?,?,?,?,?,?,?)", row)
            return cur.lastrowid

    def list(self, q="", limit=200):
        sql, args = f"SELECT {SUMMARY_COLS} FROM verifications", []
        if q:
            sql += " WHERE claim LIKE ? OR verdict LIKE ?"
            args += [f"%{q}%", f"%{q}%"]
        sql += " ORDER BY id DESC LIMIT ?"
        with self.lock:
            rows = [dict(r) for r in self.db.execute(sql, args + [int(limit)])]
            stats = {r["verdict"]: r["n"] for r in self.db.execute(
                "SELECT verdict, COUNT(*) AS n FROM verifications GROUP BY verdict")}
        return {"items": rows, "total": sum(stats.values()), "by_verdict": stats}

    def get(self, hid):
        with self.lock:
            r = self.db.execute("SELECT * FROM verifications WHERE id = ?", (int(hid),)).fetchone()
        if not r:
            return None
        out = dict(r)
        out["result"] = json.loads(out.pop("result_json"))
        return out

    def delete(self, hid=None):
        with self.lock, self.db:
            if hid is None:
                return self.db.execute("DELETE FROM verifications").rowcount
            return self.db.execute("DELETE FROM verifications WHERE id = ?", (int(hid),)).rowcount
