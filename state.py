"""SQLite state: dedupe papers, dedupe requirements, track RRDs."""
import json
import re
import sqlite3
import time


def _norm(title):
    return re.sub(r"[^a-z0-9]+", "", (title or "").lower())


def _now():
    return time.strftime("%Y-%m-%dT%H:%M:%S")


class State:
    def __init__(self, path):
        self.db = sqlite3.connect(path)
        self.db.executescript("""
        CREATE TABLE IF NOT EXISTS papers(
            arxiv_id TEXT PRIMARY KEY, title TEXT, norm TEXT,
            abstract TEXT, category TEXT, published TEXT, fetched_at TEXT);
        CREATE TABLE IF NOT EXISTS reqs(
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            paper_url TEXT, title TEXT, norm TEXT,
            payload TEXT, scores TEXT, verdict TEXT,
            related TEXT, grill TEXT, rrd_path TEXT, ts TEXT);
        """)
        self.db.commit()

    # --- paper dedup (arxiv_id is the stable key) ---
    @staticmethod
    def _paper_norm(p):
        return re.sub(r"[^a-z0-9]+", "", (p.get("title", "")).lower())

    def seen(self, arxiv_id, norm=None):
        if self.db.execute("SELECT 1 FROM papers WHERE arxiv_id=?", (arxiv_id,)).fetchone():
            return True
        return norm is not None and self.db.execute(
            "SELECT 1 FROM papers WHERE norm=? AND norm IS NOT NULL", (norm,)).fetchone() is not None

    def unseen(self, items):
        return [p for p in items if not self.seen(p["arxiv_id"], self._paper_norm(p))]

    def mark_seen(self, items):
        for p in items:
            self.db.execute(
                "INSERT OR IGNORE INTO papers(arxiv_id,title,norm,abstract,category,published,fetched_at)"
                " VALUES(?,?,?,?,?,?,?)",
                (p["arxiv_id"], p.get("title", ""), self._paper_norm(p),
                 (p.get("abstract") or "")[:1000], p.get("category", ""),
                 p.get("published", ""), _now()))
        self.db.commit()

    # --- requirements ---
    def has_req(self, title):
        return self.db.execute(
            "SELECT 1 FROM reqs WHERE norm=?", (_norm(title),)).fetchone() is not None

    def add_req(self, paper_url, title, payload, scores, verdict,
                related=None, grill=None):
        cur = self.db.execute(
            "INSERT INTO reqs(paper_url,title,norm,payload,scores,verdict,related,grill,ts)"
            " VALUES(?,?,?,?,?,?,?,?,?)",
            (paper_url, title, _norm(title),
             self._j(payload), self._j(scores), verdict,
             self._j(related), self._j(grill), _now()))
        self.db.commit()
        return cur.lastrowid

    def save_rrd(self, req_id, path):
        self.db.execute("UPDATE reqs SET rrd_path=? WHERE id=?", (path, req_id))
        self.db.commit()

    # --- reporting ---
    def recent_reqs(self, n=30):
        """Most recent requirements as (title, one_liner, gap) for semantic dedup."""
        rows = self.db.execute(
            "SELECT title, payload FROM reqs ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        out = []
        for title, payload in rows:
            try:
                p = json.loads(payload or "{}")
            except Exception:
                p = {}
            out.append({"title": title, "one_liner": p.get("one_liner", ""),
                        "gap": p.get("gap", "")})
        return out

    def counts(self):
        c = self.db.execute("SELECT COUNT(*), SUM(rrd_path IS NOT NULL) FROM reqs").fetchone()
        n = self.db.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
        return {"papers_seen": n, "reqs": c[0], "rrds": c[1] or 0}

    def recent_rrds(self, n=10):
        rows = self.db.execute(
            "SELECT title, rrd_path, verdict, ts FROM reqs WHERE rrd_path IS NOT NULL"
            " ORDER BY id DESC LIMIT ?", (n,)).fetchall()
        return [{"title": t, "path": p, "verdict": v, "ts": ts} for t, p, v, ts in rows]

    @staticmethod
    def _j(x):
        return json.dumps(x, ensure_ascii=False) if x is not None else None
