from __future__ import annotations

import unittest
from pathlib import Path

import pandas as pd


WORKFLOW_ROOT = Path(__file__).resolve().parents[3]
if str(WORKFLOW_ROOT) not in __import__("sys").path:
    __import__("sys").path.append(str(WORKFLOW_ROOT))

from legacy.scripts.cli import _build_saved_team_payload, _normalize_composition_counts, _severity_from_issue_score


class CliHelperTests(unittest.TestCase):
    def test_severity_labels(self) -> None:
        self.assertEqual(_severity_from_issue_score(9), "HIGH")
        self.assertEqual(_severity_from_issue_score(6), "MEDIUM")
        self.assertEqual(_severity_from_issue_score(2), "LOW")

    def test_normalize_composition_counts_filters_zero_values(self) -> None:
        archetypes = ["Generalist", "Fast Attacker", "Defensive Tank"]
        raw_counts = {
            "Generalist": 3,
            "Fast Attacker": 2,
            "Defensive Tank": 0,
            "Unknown": 4,
        }

        normalized = _normalize_composition_counts(raw_counts, archetypes)
        self.assertEqual(normalized, {"Generalist": 3, "Fast Attacker": 2})

    def test_build_saved_team_payload_preserves_analysis(self) -> None:
        team_df = pd.DataFrame(
            [
                {
                    "name": "mewtwo",
                    "archetype": "Fast Attacker",
                    "type1": "psychic",
                    "type2": None,
                    "hp": 106,
                    "attack": 110,
                    "defense": 90,
                    "special-attack": 154,
                    "special-defense": 90,
                    "speed": 130,
                }
            ]
        )

        payload = _build_saved_team_payload(
            team_df,
            0.942,
            {"base_strength": 0.88},
            analysis={"overall_score": 92},
        )

        self.assertEqual(payload["rank"], 1)
        self.assertAlmostEqual(payload["fitness"], 0.942)
        self.assertEqual(payload["breakdown"], {"base_strength": 0.88})
        self.assertEqual(payload["pokemon"][0]["name"], "mewtwo")
        self.assertEqual(payload["analysis"]["overall_score"], 92)

    def test_build_saved_team_payload_defaults_rank_and_normalizes_missing_values(self) -> None:
        team_df = pd.DataFrame(
            [
                {
                    "name": "gengar",
                    "type1": "ghost",
                    "type2": float("nan"),
                    "special-attack": 130,
                    "speed": 110,
                }
            ]
        )

        payload = _build_saved_team_payload(team_df, 0.733, {"type_coverage": 0.91})

        self.assertEqual(payload["rank"], 1)
        self.assertNotIn("analysis", payload)
        self.assertTrue(pd.isna(payload["pokemon"][0]["type2"]))


if __name__ == "__main__":
    unittest.main()