from __future__ import annotations

import json
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from threading import RLock
from typing import Any
from uuid import uuid4


class TeamStore:
    def __init__(self, db_path: str | Path | None = None) -> None:
        self._db_path = Path(db_path) if db_path is not None else None
        database_target = ":memory:" if self._db_path is None else str(self._db_path)
        if self._db_path is not None:
            self._db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = RLock()
        self._conn = sqlite3.connect(database_target, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._initialize_schema()

    def _initialize_schema(self) -> None:
        with self._lock:
            self._conn.execute(
                """
                CREATE TABLE IF NOT EXISTS saved_teams (
                    id TEXT PRIMARY KEY,
                    session_id TEXT NOT NULL,
                    nickname TEXT NOT NULL,
                    rank INTEGER,
                    fitness REAL,
                    config_name TEXT,
                    composition_name TEXT,
                    power_mode TEXT,
                    created_at TEXT NOT NULL,
                    team_json TEXT NOT NULL,
                    metadata_json TEXT NOT NULL
                )
                """
            )
            self._conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_saved_teams_session_created ON saved_teams(session_id, created_at DESC)"
            )
            self._conn.commit()

    def save_team(
        self,
        *,
        session_id: str,
        nickname: str,
        team_payload: dict[str, Any],
        metadata: dict[str, Any] | None = None,
    ) -> str:
        team_id = uuid4().hex
        created_at = datetime.now(timezone.utc).isoformat()
        team_json = json.dumps(team_payload, indent=2, default=str)
        metadata_json = json.dumps(metadata or {}, indent=2, default=str)

        with self._lock:
            self._conn.execute(
                """
                INSERT INTO saved_teams (
                    id, session_id, nickname, rank, fitness,
                    config_name, composition_name, power_mode,
                    created_at, team_json, metadata_json
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    team_id,
                    session_id,
                    nickname.strip(),
                    int(team_payload.get("rank", 0) or 0),
                    float(team_payload.get("fitness", 0.0) or 0.0),
                    (metadata or {}).get("config_name"),
                    (metadata or {}).get("composition_name"),
                    (metadata or {}).get("power_mode"),
                    created_at,
                    team_json,
                    metadata_json,
                ),
            )
            self._conn.commit()
        return team_id

    def list_teams(self, session_id: str | None = None) -> list[dict[str, Any]]:
        query = "SELECT * FROM saved_teams"
        params: tuple[Any, ...] = ()
        if session_id is not None:
            query += " WHERE session_id = ?"
            params = (session_id,)
        query += " ORDER BY created_at DESC"

        with self._lock:
            rows = self._conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def get_team(self, team_id: str) -> dict[str, Any] | None:
        with self._lock:
            row = self._conn.execute("SELECT * FROM saved_teams WHERE id = ?", (team_id,)).fetchone()
        if row is None:
            return None
        return self._row_to_record(row)

    def rename_team(self, team_id: str, nickname: str) -> None:
        with self._lock:
            self._conn.execute("UPDATE saved_teams SET nickname = ? WHERE id = ?", (nickname.strip(), team_id))
            self._conn.commit()

    def delete_team(self, team_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM saved_teams WHERE id = ?", (team_id,))
            self._conn.commit()

    def clear_session(self, session_id: str) -> None:
        with self._lock:
            self._conn.execute("DELETE FROM saved_teams WHERE session_id = ?", (session_id,))
            self._conn.commit()

    def close(self) -> None:
        with self._lock:
            self._conn.close()

    def _row_to_record(self, row: sqlite3.Row) -> dict[str, Any]:
        record = dict(row)
        record["team_payload"] = json.loads(record.pop("team_json"))
        record["metadata"] = json.loads(record.pop("metadata_json"))
        return record