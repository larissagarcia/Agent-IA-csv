import sqlite3, json
from datetime import datetime

class Memory:
    def __init__(self, db_path="/content/project/memory.sqlite"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_tables()
    def _create_tables(self):
        c = self.conn.cursor()
        c.execute("""CREATE TABLE IF NOT EXISTS interactions (
                        id INTEGER PRIMARY KEY AUTOINCREMENT,
                        timestamp TEXT,
                        question TEXT,
                        answer TEXT,
                        meta TEXT
                    )""")
        self.conn.commit()
    def add_interaction(self, question, answer, meta=None):
        c = self.conn.cursor()
        c.execute("INSERT INTO interactions (timestamp, question, answer, meta) VALUES (?, ?, ?, ?)",
                  (datetime.utcnow().isoformat(), question, answer, json.dumps(meta or {})))
        self.conn.commit()
    def get_all(self, limit=50):
        c = self.conn.cursor()
        c.execute("SELECT timestamp, question, answer, meta FROM interactions ORDER BY id DESC LIMIT ?", (limit,))
        return c.fetchall()
