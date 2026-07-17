import json
import logging
import os
import sqlite3
import threading
import time

import faiss
import numpy as np

from gpu_utils import has_cuda

logger = logging.getLogger(__name__)


def _make_index(dim: int) -> faiss.Index:
    index = faiss.IndexFlatIP(dim)
    if has_cuda():
        try:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, 0, index)
            logger.info("FAISS index on GPU")
        except Exception as e:
            logger.warning(f"FAISS GPU failed, falling back to CPU: {e}")
    return index


class Gallery:
    def __init__(self, db_path: str = "gallery", dim: int = 512):
        self.dim = dim
        self.db_path = db_path
        self.index_file = os.path.join(db_path, "faiss_index.bin")
        self.names_file = os.path.join(db_path, "names.json")

        os.makedirs(db_path, exist_ok=True)

        self._conn = sqlite3.connect(os.path.join(db_path, "gallery.db"), check_same_thread=False)
        self._init_db()
        self._lock = threading.Lock()

        self._index = _make_index(dim)
        self._names: dict[int, str] = {}
        self._next_id = 1
        self._load()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS persons (
                id INTEGER PRIMARY KEY,
                name TEXT NOT NULL,
                desc TEXT DEFAULT '',
                image BLOB,
                created REAL,
                updated REAL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS embeddings (
                person_id INTEGER PRIMARY KEY,
                vector BLOB,
                FOREIGN KEY (person_id) REFERENCES persons(id)
            )
        """)
        self._conn.commit()

    def _load(self):
        rows = self._conn.execute(
            "SELECT p.id, p.name, e.vector FROM persons p "
            "JOIN embeddings e ON p.id = e.person_id ORDER BY p.id"
        ).fetchall()
        vecs = []
        for pid, name, vec_blob in rows:
            vec = np.frombuffer(vec_blob, dtype=np.float32)
            vecs.append(vec)
            self._names[pid] = name
            if pid >= self._next_id:
                self._next_id = pid + 1
        if vecs:
            mat = np.stack(vecs)
            self._index.add(mat)
        logger.info(f"Loaded gallery: {len(self._names)} identities")

    def add(self, embedding: np.ndarray, name: str, image: bytes | None = None, desc: str = "") -> int:
        with self._lock:
            pid = self._next_id
            self._next_id += 1
            now = time.time()

            vec = embedding.astype(np.float32)
            norm = np.linalg.norm(vec)
            if norm > 0:
                vec /= norm

            self._conn.execute(
                "INSERT INTO persons (id, name, desc, image, created, updated) VALUES (?, ?, ?, ?, ?, ?)",
                (pid, name, desc, image, now, now),
            )
            self._conn.execute(
                "INSERT INTO embeddings (person_id, vector) VALUES (?, ?)",
                (pid, vec.tobytes()),
            )
            self._conn.commit()
            self._index.add(vec.reshape(1, -1))
            self._names[pid] = name
            self._save_meta()

        logger.info(f"Added {name} (id={pid})")
        return pid

    def delete(self, pid: int) -> bool:
        with self._lock:
            cur = self._conn.execute("SELECT id FROM persons WHERE id=?", (pid,))
            if not cur.fetchone():
                return False
            self._conn.execute("DELETE FROM persons WHERE id=?", (pid,))
            self._conn.execute("DELETE FROM embeddings WHERE person_id=?", (pid,))
            self._conn.commit()
            self._names.pop(pid, None)
            self._rebuild_index()
            self._save_meta()
        logger.info(f"Deleted id={pid}")
        return True

    def _rebuild_index(self):
        self._index = _make_index(self.dim)
        rows = self._conn.execute(
            "SELECT person_id, vector FROM embeddings ORDER BY person_id"
        ).fetchall()
        vecs = []
        self._names.clear()
        for pid, vec_blob in rows:
            vec = np.frombuffer(vec_blob, dtype=np.float32)
            vecs.append(vec)
            row = self._conn.execute("SELECT name FROM persons WHERE id=?", (pid,)).fetchone()
            if row:
                self._names[pid] = row[0]
        if vecs:
            mat = np.stack(vecs)
            self._index.add(mat)

    def _save_meta(self):
        with open(self.names_file, "w") as f:
            json.dump({"names": self._names, "next_id": self._next_id}, f)

    def rename(self, pid: int, new_name: str) -> bool:
        with self._lock:
            cur = self._conn.execute("SELECT id FROM persons WHERE id=?", (pid,))
            if not cur.fetchone():
                return False
            self._names[pid] = new_name
            self._conn.execute("UPDATE persons SET name=?, updated=? WHERE id=?",
                               (new_name, time.time(), pid))
            self._conn.commit()
            self._save_meta()
            return True

    def match(self, embedding: np.ndarray, threshold: float = 0.35) -> tuple[int | None, float, str | None]:
        if self._index.ntotal == 0:
            return None, 0.0, None
        vec = embedding.astype(np.float32).reshape(1, -1)
        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm
        scores, indices = self._index.search(vec, 1)
        sim = float(scores[0][0])
        idx = int(indices[0][0])
        if sim >= threshold and idx < len(self._names):
            pids = list(self._names.keys())
            pid = pids[idx]
            return pid, sim, self._names[pid]
        return None, sim, None

    def list_persons(self) -> list[dict]:
        with self._lock:
            rows = self._conn.execute(
                "SELECT id, name, desc, created, updated, (image IS NOT NULL) FROM persons ORDER BY id"
            ).fetchall()
            return [{"id": r[0], "name": r[1], "desc": r[2],
                     "created": r[3], "updated": r[4], "has_image": bool(r[5])} for r in rows]

    def get_person(self, pid: int) -> dict | None:
        with self._lock:
            row = self._conn.execute(
                "SELECT id, name, desc, image, created, updated FROM persons WHERE id=?", (pid,)
            ).fetchone()
            if not row:
                return None
            return {"id": row[0], "name": row[1], "desc": row[2],
                    "image": row[3], "created": row[4], "updated": row[5]}

    def count(self) -> int:
        with self._lock:
            row = self._conn.execute("SELECT COUNT(*) FROM persons").fetchone()
            return row[0] if row else 0

    def close(self):
        self._conn.close()
