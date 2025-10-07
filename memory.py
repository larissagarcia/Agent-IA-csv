import sqlite3, json, time

class Memory:
    def __init__(self, db_path="memory.sqlite"):
        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self._create_table()

    def _create_table(self):
        cur = self.conn.cursor()
        cur.execute("CREATE TABLE IF NOT EXISTS memory (timestamp REAL, question TEXT, answer TEXT, meta TEXT)")
        self.conn.commit()

    def add_interaction(self, question, answer, meta=None):
        cur = self.conn.cursor()
        cur.execute("INSERT INTO memory VALUES (?, ?, ?, ?)", (time.time(), question, answer, json.dumps(meta)))
        self.conn.commit()

    def get_all(self, limit=10):
        cur = self.conn.cursor()
        cur.execute("SELECT * FROM memory ORDER BY timestamp DESC LIMIT ?", (limit,))
        return cur.fetchall()
