import sqlite3
import threading
import time
import numpy as np
import faiss


class Gallery:
    def __init__(self, path: str = "v3_faces.db", dim: int = 512):
        self.conn = sqlite3.connect(path, check_same_thread=False)
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                first_seen REAL,
                last_seen REAL,
                n_seen INTEGER DEFAULT 1,
                desc TEXT DEFAULT '',
                image BLOB
            )
        """)
        self.conn.commit()
        self.lock = threading.Lock()
        self.dim = dim
        self._vecs: list[tuple[int, np.ndarray]] = []
        self.index: faiss.Index | None = None
        self._load()

    def _load(self):
        with self.lock:
            cur = self.conn.execute("SELECT id FROM persons")
            pids = [r[0] for r in cur.fetchall()]
        for pid in pids:
            emb = self._get_emb(pid)
            if emb is not None:
                self._vecs.append((pid, emb))
        self._rebuild()
        print(f"[gallery] loaded {len(self._vecs)} identities")

    def _get_emb(self, pid: int) -> np.ndarray | None:
        with self.lock:
            cur = self.conn.execute(
                "SELECT embedding FROM embeddings WHERE person_id=?", (pid,)
            )
            row = cur.fetchone()
            return np.frombuffer(row[0], dtype=np.float32) if row and row[0] else None

    def _ensure_tables(self):
        self.conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                person_id INTEGER PRIMARY KEY,
                embedding BLOB,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        """)
        self.conn.commit()

    def _rebuild(self):
        if not self._vecs:
            self.index = None
            return
        vecs = np.array([v for _, v in self._vecs], dtype=np.float32)
        self.index = faiss.IndexFlatIP(self.dim)
        self.index.add(vecs)

    def add(self, embedding: np.ndarray, image: bytes | None = None,
            desc: str = "") -> int:
        with self.lock:
            self._ensure_tables()
            pid = self._next_id()
            t = time.time()
            self.conn.execute(
                "INSERT INTO persons (id, first_seen, last_seen, n_seen, desc, image) "
                "VALUES (?, ?, ?, 1, ?, ?)", (pid, t, t, desc, image)
            )
            self.conn.execute(
                "INSERT INTO embeddings (person_id, embedding) VALUES (?, ?)",
                (pid, embedding.tobytes()),
            )
            self.conn.commit()
        self._vecs.append((pid, embedding))
        self._rebuild()
        return pid

    def match(self, embedding: np.ndarray,
              threshold: float = 0.35) -> tuple[int | None, float]:
        if self.index is None or self.index.ntotal == 0:
            return None, 0.0
        scores, indices = self.index.search(
            embedding.reshape(1, -1).astype(np.float32), 1
        )
        sim = float(scores[0][0])
        if sim >= threshold:
            return self._vecs[indices[0][0]][0], sim
        return None, sim

    def update(self, pid: int):
        with self.lock:
            cur = self.conn.execute("SELECT n_seen FROM persons WHERE id=?", (pid,))
            row = cur.fetchone()
            if row:
                self.conn.execute(
                    "UPDATE persons SET last_seen=?, n_seen=? WHERE id=?",
                    (time.time(), row[0] + 1, pid),
                )
                self.conn.commit()

    def delete(self, pid: int):
        with self.lock:
            self.conn.execute("DELETE FROM persons WHERE id=?", (pid,))
            self.conn.execute("DELETE FROM embeddings WHERE person_id=?", (pid,))
            self.conn.commit()
        self._vecs = [(p, e) for p, e in self._vecs if p != pid]
        self._rebuild()

    def get_image(self, pid: int) -> bytes | None:
        with self.lock:
            cur = self.conn.execute("SELECT image FROM persons WHERE id=?", (pid,))
            row = cur.fetchone()
            return row[0] if row else None

    def count(self) -> int:
        with self.lock:
            return self.conn.execute("SELECT COUNT(*) FROM persons").fetchone()[0]

    def list_persons(self) -> list[dict]:
        with self.lock:
            cur = self.conn.execute(
                "SELECT id, n_seen, desc, (image IS NOT NULL) FROM persons ORDER BY id"
            )
            return [
                {"id": r[0], "n_seen": r[1], "desc": r[2], "has_image": bool(r[3])}
                for r in cur.fetchall()
            ]

    def _next_id(self) -> int:
        used = set(r[0] for r in self.conn.execute("SELECT id FROM persons").fetchall())
        i = 1
        while i in used:
            i += 1
        return i
