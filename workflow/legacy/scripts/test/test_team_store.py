from __future__ import annotations

import tempfile
import unittest
from pathlib import Path


WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
if str(WORKFLOW_ROOT) not in __import__("sys").path:
    __import__("sys").path.append(str(WORKFLOW_ROOT))

from src.team_store import TeamStore


class TeamStoreTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temp_dir = tempfile.TemporaryDirectory()
        self.db_path = Path(self.temp_dir.name) / "teams.sqlite3"
        self.store = TeamStore(self.db_path)

    def tearDown(self) -> None:
        self.store.close()
        self.temp_dir.cleanup()

    def test_save_list_rename_delete(self) -> None:
        payload = {
            "rank": 1,
            "fitness": 0.7323,
            "pokemon": [
                {"name": "mewtwo", "type1": "psychic", "type2": None, "archetype": "Fast Attacker"},
                {"name": "groudon", "type1": "ground", "type2": None, "archetype": "Generalist"},
            ],
            "breakdown": {"base_strength": 0.71, "type_coverage": 0.94},
        }

        team_id = self.store.save_team(
            session_id="session-1",
            nickname="Main Squad",
            team_payload=payload,
            metadata={"config_name": "C", "composition_name": "balanced", "power_mode": "standard"},
        )

        teams = self.store.list_teams("session-1")
        self.assertEqual(len(teams), 1)
        self.assertEqual(teams[0]["id"], team_id)
        self.assertEqual(teams[0]["nickname"], "Main Squad")
        self.assertEqual(teams[0]["team_payload"]["rank"], 1)
        self.assertEqual(teams[0]["metadata"]["config_name"], "C")

        self.store.rename_team(team_id, "Final Squad")
        renamed = self.store.get_team(team_id)
        self.assertIsNotNone(renamed)
        self.assertEqual(renamed["nickname"], "Final Squad")

        self.store.delete_team(team_id)
        self.assertEqual(self.store.list_teams("session-1"), [])

    def test_session_isolation(self) -> None:
        payload = {"rank": 1, "fitness": 0.5, "pokemon": []}
        self.store.save_team(session_id="session-a", nickname="A", team_payload=payload)
        self.store.save_team(session_id="session-b", nickname="B", team_payload=payload)

        self.assertEqual(len(self.store.list_teams("session-a")), 1)
        self.assertEqual(len(self.store.list_teams("session-b")), 1)
        self.assertEqual(len(self.store.list_teams()), 2)

    def test_clear_session_removes_only_that_session(self) -> None:
        payload = {"rank": 1, "fitness": 0.5, "pokemon": []}
        self.store.save_team(session_id="session-a", nickname="A", team_payload=payload)
        self.store.save_team(session_id="session-b", nickname="B", team_payload=payload)

        self.store.clear_session("session-a")
        self.assertEqual(len(self.store.list_teams("session-a")), 0)
        self.assertEqual(len(self.store.list_teams("session-b")), 1)

    def test_save_trims_nickname_and_defaults_metadata(self) -> None:
        payload = {"rank": 2, "fitness": 0.61, "pokemon": [{"name": "pikachu"}]}
        team_id = self.store.save_team(
            session_id="session-trim",
            nickname="  Trim Me  ",
            team_payload=payload,
        )

        saved = self.store.get_team(team_id)
        self.assertIsNotNone(saved)
        assert saved is not None
        self.assertEqual(saved["nickname"], "Trim Me")
        self.assertEqual(saved["metadata"], {})
        self.assertEqual(len(saved["id"]), 32)

    def test_list_order_is_newest_first(self) -> None:
        payload = {"rank": 1, "fitness": 0.5, "pokemon": []}
        first_id = self.store.save_team(session_id="session-order", nickname="First", team_payload=payload)
        second_id = self.store.save_team(session_id="session-order", nickname="Second", team_payload=payload)

        rows = self.store.list_teams("session-order")
        self.assertEqual(len(rows), 2)
        self.assertEqual(rows[0]["id"], second_id)
        self.assertEqual(rows[1]["id"], first_id)


if __name__ == "__main__":
    unittest.main()