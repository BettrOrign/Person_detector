import logging
import os
import sqlite3
import time

logger = logging.getLogger(__name__)


class PlateGallery:
    def __init__(self, path: str = "gallery"):
        self.db_path = path
        os.makedirs(path, exist_ok=True)
        self._conn = sqlite3.connect(os.path.join(path, "plates.db"), check_same_thread=False)
        self._init_db()

    def _init_db(self):
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS plates (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_text TEXT NOT NULL,
                first_seen REAL,
                last_seen REAL
            )
        """)
        self._conn.execute("""
            CREATE TABLE IF NOT EXISTS sightings (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                plate_id INTEGER,
                image BLOB,
                timestamp REAL,
                FOREIGN KEY (plate_id) REFERENCES plates(id)
            )
        """)
        self._conn.execute("CREATE INDEX IF NOT EXISTS idx_plate_text ON plates(plate_text)")
        self._conn.commit()

    def get_or_create(self, plate_text: str, image: bytes | None = None) -> int:
        now = time.time()
        cur = self._conn.execute("SELECT id FROM plates WHERE plate_text=?", (plate_text,))
        row = cur.fetchone()
        if row:
            pid = row[0]
            self._conn.execute("UPDATE plates SET last_seen=? WHERE id=?", (now, pid))
        else:
            pid = self._conn.execute(
                "INSERT INTO plates (plate_text, first_seen, last_seen) VALUES (?, ?, ?)",
                (plate_text, now, now),
            ).lastrowid
        if image is not None:
            self._conn.execute(
                "INSERT INTO sightings (plate_id, image, timestamp) VALUES (?, ?, ?)",
                (pid, image, now),
            )
        self._conn.commit()
        return pid

    def search(self, plate_text: str) -> int | None:
        cur = self._conn.execute("SELECT id FROM plates WHERE plate_text=?", (plate_text,))
        row = cur.fetchone()
        return row[0] if row else None

    def list_all(self) -> list[dict]:
        rows = self._conn.execute(
            "SELECT id, plate_text, first_seen, last_seen FROM plates ORDER BY last_seen DESC"
        ).fetchall()
        return [
            {"id": r[0], "plate": r[1], "first_seen": r[2], "last_seen": r[3]}
            for r in rows
        ]

    def delete(self, plate_id: int) -> bool:
        cur = self._conn.execute("SELECT id FROM plates WHERE id=?", (plate_id,))
        if not cur.fetchone():
            return False
        self._conn.execute("DELETE FROM sightings WHERE plate_id=?", (plate_id,))
        self._conn.execute("DELETE FROM plates WHERE id=?", (plate_id,))
        self._conn.commit()
        return True

    def count(self) -> int:
        row = self._conn.execute("SELECT COUNT(*) FROM plates").fetchone()
        return row[0] if row else 0

    def close(self):
        self._conn.close()
