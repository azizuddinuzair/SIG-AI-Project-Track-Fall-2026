"""Legacy helper functions preserved from cleanup.

These functions were removed from active paths during dead-code cleanup,
but retained here for historical reference and potential reuse while
re-optimizing GA/GMM workflows.
"""

from typing import List


# From scripts/cli.py
# Original behavior: normalize free-form user input against a constrained option set.
def input_choice_legacy(prompt: str, choices: List[str], default: str) -> str:
    allowed = {c.lower(): c for c in choices}
    while True:
        raw = input(f"{prompt} ({'/'.join(choices)}, default {default}): ").strip().lower()
        if not raw:
            return default
        if raw in allowed:
            return allowed[raw]
        print(f"Invalid choice. Allowed: {', '.join(choices)}")


# From scripts/validation/validate_601_clustering.py
ARCHETYPE_TO_ROLE_LEGACY = {
    "Speed Sweeper": "Sweeper",
    "Fast Attacker": "Sweeper",
    "Physical Attacker": "Sweeper",
    "Defensive Tank": "Wall",
    "Balanced All-Rounder": "Mixed",
    "Generalist": "Mixed",
}


# Original behavior: archetype-only role mapping used prior to enhanced row-based mapping.
def predict_role_legacy(archetype_name: str) -> str:
    return ARCHETYPE_TO_ROLE_LEGACY.get(archetype_name, "Mixed")
