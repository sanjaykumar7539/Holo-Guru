"""Append-only SQLite audit trail for security and knowledge operations."""
import json
import sqlite3
from datetime import datetime, timezone


class AuditLog:
    def __init__(self, path):
        self.path = path
        with sqlite3.connect(path) as db:
            db.execute("""CREATE TABLE IF NOT EXISTS audit_events (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TEXT NOT NULL,
                event_type TEXT NOT NULL,
                session_id TEXT,
                client_ip TEXT,
                details_json TEXT NOT NULL
            )""")

    def write(self, event_type, session_id="", client_ip="", details=None):
        payload = json.dumps(details or {}, ensure_ascii=False)
        with sqlite3.connect(self.path) as db:
            db.execute(
                "INSERT INTO audit_events(created_at,event_type,session_id,client_ip,details_json) VALUES(?,?,?,?,?)",
                (datetime.now(timezone.utc).isoformat(), event_type, session_id, client_ip, payload),
            )
