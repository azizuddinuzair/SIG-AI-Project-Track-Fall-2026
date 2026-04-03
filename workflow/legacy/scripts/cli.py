"""
User-facing CLI for clustering and GA workflows.

Commands:
- cluster: Run clustering pipeline and optional deliverable generation/validation
- ga: Run GA optimization with configurable parameters
- pipeline: Run cluster -> ga in one command
"""

from __future__ import annotations

import atexit
import argparse
import json
import subprocess
import sys
import traceback
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List, Tuple
from uuid import uuid4

import pandas as pd

# Ensure imports from Proj1/src work when running from legacy/scripts.
PROJ_ROOT = Path(__file__).resolve().parents[2]
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

from src.ga import PokemonGA, get_config_c
from src.ga.config import get_config_a, get_config_b, get_config_random
from src.ga.optimization import load_pokemon_data
from src.ga.fitness import TYPE_NAMES, get_type_effectiveness
from src.team_store import TeamStore

PIVOT_CANDIDATE_THRESHOLD = 0.62
ABILITY_DATA_PATH = PROJ_ROOT / "data" / "pokemon_abilities.csv"
CLI_SESSION_ID_KEY = "cli-session"
_CLI_SESSION_ID: str | None = None
_CLI_TEAM_STORE: TeamStore | None = None


def _print_header(title: str) -> None:
    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)


def _input_int(prompt: str, min_value: int, max_value: int, default: int) -> int:
    while True:
        raw = input(f"{prompt} [{min_value}-{max_value}] (default {default}): ").strip()
        if not raw:
            return default
        try:
            value = int(raw)
        except ValueError:
            print("Invalid number. Try again.")
            continue
        if value < min_value or value > max_value:
            print(f"Please choose a value between {min_value} and {max_value}.")
            continue
        return value


def _input_yes_no(prompt: str, default_yes: bool = True) -> bool:
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        raw = input(f"{prompt} {suffix}: ").strip().lower()
        if not raw:
            return default_yes
        if raw in {"y", "yes"}:
            return True
        if raw in {"n", "no"}:
            return False
        print("Please enter y or n.")


def _get_cli_session_id() -> str:
    global _CLI_SESSION_ID
    if _CLI_SESSION_ID is None:
        _CLI_SESSION_ID = f"{CLI_SESSION_ID_KEY}-{uuid4().hex}"
    return _CLI_SESSION_ID


def _get_cli_team_store() -> TeamStore:
    global _CLI_TEAM_STORE
    if _CLI_TEAM_STORE is None:
        _CLI_TEAM_STORE = TeamStore()
        atexit.register(_CLI_TEAM_STORE.close)
    return _CLI_TEAM_STORE


def _safe_input(prompt: str) -> str | None:
    try:
        return input(prompt)
    except EOFError:
        return None
    except KeyboardInterrupt:
        return None


def _available_archetypes(pokemon_df: pd.DataFrame) -> List[str]:
    return sorted(pokemon_df["archetype"].dropna().astype(str).unique().tolist())


def _select_anchor_pokemon(pokemon_df: pd.DataFrame) -> List[str]:
    print("Choose 1-5 anchor Pokemon to lock into the generated team.")
    anchor_count = _input_int("How many anchors?", 1, 5, 1)
    anchors: List[str] = []

    for i in range(anchor_count):
        while True:
            raw_input = _safe_input(f"Enter anchor Pokemon #{i + 1}: ")
            if raw_input is None:
                return []
            resolved = _resolve_pokemon_name(raw_input.strip(), pokemon_df)
            if not resolved:
                print("Pokemon not found. Try again.")
                continue
            if resolved in anchors:
                print("Duplicate anchor. Choose a different Pokemon.")
                continue
            anchors.append(resolved)
            break

    return anchors


def _build_composition_presets() -> Dict[str, Dict[str, int]]:
    return {
        "balanced": {
            "Balanced All-Rounder": 2,
            "Generalist": 2,
            "Fast Attacker": 1,
            "Defensive Tank": 1,
        },
        "hyper_offense": {
            "Speed Sweeper": 2,
            "Fast Attacker": 2,
            "Physical Attacker": 1,
            "Generalist": 1,
        },
        "pivot_pressure": {
            "Generalist": 3,
            "Balanced All-Rounder": 1,
            "Fast Attacker": 1,
            "Defensive Tank": 1,
        },
        "bulky_offense": {
            "Defensive Tank": 2,
            "Balanced All-Rounder": 2,
            "Fast Attacker": 1,
            "Physical Attacker": 1,
        },
    }


def _normalize_composition_counts(raw_counts: Dict[str, int], archetypes: List[str]) -> Dict[str, int]:
    return {arch: int(raw_counts.get(arch, 0)) for arch in archetypes if int(raw_counts.get(arch, 0)) > 0}


def _input_custom_composition(archetypes: List[str]) -> Dict[str, int]:
    print("\nCustom composition: choose counts for each archetype. Total must equal 6.")
    while True:
        remaining = 6
        chosen: Dict[str, int] = {}
        for idx, arch in enumerate(archetypes):
            max_allowed = remaining if idx < len(archetypes) - 1 else remaining
            value = _input_int(f"  {arch}", 0, max_allowed, 0 if idx < len(archetypes) - 1 else remaining)
            chosen[arch] = value
            remaining -= value
            if remaining <= 0:
                break

        total = sum(chosen.values())
        if total == 6:
            return _normalize_composition_counts(chosen, archetypes)
        print(f"Total must be 6. You entered {total}. Try again.\n")


def _select_composition_target(pokemon_df: pd.DataFrame) -> Tuple[str, Dict[str, int], float]:
    presets = _build_composition_presets()
    archetypes = _available_archetypes(pokemon_df)

    print("\nChoose a team composition style:")
    print("1. Balanced")
    print("2. Hyper Offense")
    print("3. Pivot Pressure")
    print("4. Bulky Offense")
    print("5. Custom")

    while True:
        raw = _safe_input("Enter choice [1-5] (default 1): ")
        if raw is None:
            return "balanced", presets["balanced"], 0.20
        choice = raw.strip() or "1"
        if choice in {"1", "2", "3", "4", "5"}:
            break
        print("Invalid choice. Please enter 1-5.")

    if choice == "1":
        return "balanced", _normalize_composition_counts(presets["balanced"], archetypes), 0.20
    if choice == "2":
        return "hyper_offense", _normalize_composition_counts(presets["hyper_offense"], archetypes), 0.22
    if choice == "3":
        return "pivot_pressure", _normalize_composition_counts(presets["pivot_pressure"], archetypes), 0.20
    if choice == "4":
        return "bulky_offense", _normalize_composition_counts(presets["bulky_offense"], archetypes), 0.20

    custom_counts = _input_custom_composition(archetypes)
    strictness = _input_int(
        "Custom composition strictness (%)",
        5,
        40,
        20,
    )
    return "custom", custom_counts, float(strictness) / 100.0


def _select_team_power_mode() -> Tuple[str, Dict[str, float]]:
    print("\nChoose a team power mode:")
    print("1. Standard (recommended): soft BST cap, best overall optimization")
    print("2. Competitive Strict: fewer high-BST stacks, may reduce peak fitness")
    print("3. Open: no BST cap, allows full power stacking")

    while True:
        raw = _safe_input("Enter choice [1-3] (default 1): ")
        if raw is None:
            return "standard", {"bst_cap": 3300, "bst_penalty_weight": 2.0}
        choice = raw.strip() or "1"
        if choice in {"1", "2", "3"}:
            break
        print("Invalid choice. Please enter 1-3.")

    if choice == "1":
        return "standard", {"bst_cap": 3300, "bst_penalty_weight": 2.0}
    if choice == "2":
        return "competitive_strict", {"bst_cap": 3200, "bst_penalty_weight": 3.0}
    return "open", {"bst_cap": 0, "bst_penalty_weight": 0.0}


def _severity_from_issue_score(score: int) -> str:
    if score >= 8:
        return "HIGH"
    if score >= 5:
        return "MEDIUM"
    return "LOW"


def _run_python_script(script_path: Path, args: List[str] | None = None) -> int:
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, cwd=str(PROJ_ROOT), check=False)
    return int(result.returncode)


def _write_error_log(exc: Exception) -> Path:
    """Persist full traceback details for post-mortem debugging."""
    logs_dir = PROJ_ROOT / "reports" / "error_logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_path = logs_dir / f"cli_error_{timestamp}.log"
    with open(log_path, "w", encoding="utf-8") as f:
        f.write("Pokemon Team CLI Error Report\n")
        f.write("=" * 80 + "\n")
        f.write(f"Timestamp: {datetime.now().isoformat()}\n")
        f.write(f"Exception: {type(exc).__name__}\n")
        f.write(f"Message: {exc}\n\n")
        f.write("Traceback:\n")
        f.write("".join(traceback.format_exception(type(exc), exc, exc.__traceback__)))
    return log_path


def _run_with_error_guard(label: str, action) -> bool:
    """Run an interactive action and report concise failure with logged details."""
    try:
        action()
        return True
    except Exception as exc:
        log_path = _write_error_log(exc)
        print(f"\n[ERROR] {label} failed: {type(exc).__name__}: {exc}")
        print(f"        Full traceback saved to: {log_path.relative_to(PROJ_ROOT)}")
        print("        Please share this file if you want a targeted fix.")
        return False


def _get_config_by_name(name: str) -> Dict:
    normalized = name.upper()
    if normalized == "A":
        return get_config_a()
    if normalized == "B":
        return get_config_b()
    if normalized == "C":
        return get_config_c()
    raise ValueError(f"Unknown config '{name}'. Use A, B, or C.")


def _team_to_records(team_df: pd.DataFrame) -> List[Dict]:
    preferred_cols = [
        "name",
        "archetype",
        "type1",
        "type2",
        "hp",
        "attack",
        "defense",
        "special-attack",
        "special-defense",
        "speed",
    ]
    cols = [c for c in preferred_cols if c in team_df.columns]
    cleaned = team_df[cols].where(pd.notna(team_df[cols]), None)
    return cleaned.to_dict("records")


def _build_ga_result_payload(
    top_teams: List,
    history_df: pd.DataFrame,
    output_dir: Path | None,
    config: Dict,
    dataset_size: int,
    top_n: int,
    timestamp: str,
) -> Dict:
    top_teams_json = []
    for rank, (team_df, fitness, breakdown) in enumerate(top_teams, 1):
        top_teams_json.append(
            {
                "rank": rank,
                "fitness": float(fitness),
                "breakdown": breakdown,
                "pokemon": _team_to_records(team_df),
            }
        )

    best_fitness = float(top_teams[0][1]) if top_teams else None
    return {
        "timestamp": timestamp,
        "config": config,
        "dataset_size": int(dataset_size),
        "top_n": int(top_n),
        "best_fitness": best_fitness,
        "output_dir": str(output_dir) if output_dir else None,
        "history": {
            "generations": int(len(history_df)),
            "final_max_fitness": float(history_df["max_fitness"].iloc[-1]) if len(history_df) else None,
            "final_mean_fitness": float(history_df["mean_fitness"].iloc[-1]) if len(history_df) else None,
        },
        "top_teams": top_teams_json,
    }


def _build_saved_team_payload(
    team_df: pd.DataFrame,
    fitness: float,
    breakdown: Dict,
    *,
    rank: int = 1,
    analysis: Dict | None = None,
) -> Dict:
    payload = {
        "rank": int(rank),
        "fitness": float(fitness),
        "breakdown": breakdown,
        "pokemon": _team_to_records(team_df),
    }
    if analysis is not None:
        payload["analysis"] = analysis
    return payload


def _prompt_save_generated_team(
    *,
    team_df: pd.DataFrame,
    fitness: float,
    breakdown: Dict,
    config_name: str,
    composition_name: str,
    power_mode: str,
    analysis: Dict | None = None,
) -> None:
    if not _input_yes_no("Save this team to your session collection?", default_yes=True):
        return

    default_nickname = f"{composition_name.replace('_', ' ').title()} Team"
    raw_nickname = _safe_input(f"Nickname [{default_nickname}]: ")
    if raw_nickname is None:
        print("Save cancelled.")
        return

    nickname = raw_nickname.strip() or default_nickname
    team_payload = _build_saved_team_payload(team_df, fitness, breakdown, analysis=analysis)
    metadata = {
        "config_name": config_name,
        "composition_name": composition_name,
        "power_mode": power_mode,
        "session_id": _get_cli_session_id(),
        "saved_via": "legacy-cli",
        "timestamp": datetime.now().isoformat(),
    }

    team_id = _get_cli_team_store().save_team(
        session_id=_get_cli_session_id(),
        nickname=nickname,
        team_payload=team_payload,
        metadata=metadata,
    )
    print(f"Saved as '{nickname}' (team id: {team_id}).")


def _print_saved_teams() -> None:
    session_id = _get_cli_session_id()
    saved_teams = _get_cli_team_store().list_teams(session_id=session_id)

    _print_header("SAVED TEAMS")
    if not saved_teams:
        print("No saved teams in this CLI session yet.")
        return

    print(f"Session ID: {session_id}")
    for index, record in enumerate(saved_teams, 1):
        pokemon_names = ", ".join(pokemon.get("name", "?") for pokemon in record["team_payload"].get("pokemon", []))
        print(
            f"{index}. {record['nickname']} | rank {record.get('rank', 0)} | "
            f"fitness {float(record.get('fitness', 0.0)):.4f} | {pokemon_names}"
        )
        metadata = record.get("metadata", {})
        composition = metadata.get("composition_name")
        power_mode = metadata.get("power_mode")
        if composition or power_mode:
            details = []
            if composition:
                details.append(f"composition={composition}")
            if power_mode:
                details.append(f"power_mode={power_mode}")
            print("   " + ", ".join(details))

    if _input_yes_no("Clear all saved teams for this session?", default_yes=False):
        _get_cli_team_store().clear_session(session_id)
        print("Session saved teams cleared.")


def _team_signature(team_df: pd.DataFrame) -> tuple:
    # Canonical representation to identify duplicate team compositions.
    return tuple(sorted(team_df["name"].tolist()))


def _team_member_set(team_df: pd.DataFrame) -> set:
    return set(team_df["name"].tolist())


def _get_unique_best_teams(
    ga: PokemonGA,
    n: int,
    max_overlap: int = 4,
    allow_backfill: bool = True,
) -> List:
    population_size = int(ga.config["population"]["size"])
    candidates = ga.get_best_teams(population_size)

    # Stage 1: remove exact duplicates
    unique_candidates = []
    seen = set()
    for team_df, fitness, breakdown in candidates:
        sig = _team_signature(team_df)
        if sig in seen:
            continue
        seen.add(sig)
        unique_candidates.append((team_df, fitness, breakdown))

    # Stage 2: enforce diversity across suggestions by overlap cap
    diverse = []
    diverse_sets = []
    for team_df, fitness, breakdown in unique_candidates:
        members = _team_member_set(team_df)
        if all(len(members & chosen) <= max_overlap for chosen in diverse_sets):
            diverse.append((team_df, fitness, breakdown))
            diverse_sets.append(members)
        if len(diverse) >= n:
            break

    # Stage 3: optional backfill with remaining unique candidates.
    if allow_backfill and len(diverse) < n:
        used_signatures = {_team_signature(team_df) for team_df, _, _ in diverse}
        for team_df, fitness, breakdown in unique_candidates:
            sig = _team_signature(team_df)
            if sig in used_signatures:
                continue
            diverse.append((team_df, fitness, breakdown))
            used_signatures.add(sig)
            if len(diverse) >= n:
                break

    return diverse


def _normalize_name(name: str) -> str:
    return name.strip().lower()


def _resolve_pokemon_name(input_name: str, pokemon_df: pd.DataFrame) -> str | None:
    normalized = _normalize_name(input_name)
    if not normalized:
        return None
    names = pokemon_df["name"].tolist()
    name_map = {_normalize_name(n): n for n in names}

    if normalized in name_map:
        return name_map[normalized]

    partial = [n for n in names if normalized in _normalize_name(n)]
    if len(partial) == 1:
        return partial[0]
    if len(partial) > 1:
        print("Multiple matches found:")
        for candidate in partial[:10]:
            print(f"  - {candidate}")
        if len(partial) > 10:
            print(f"  ...and {len(partial) - 10} more")
    return None


def _get_stat_columns(team_df: pd.DataFrame) -> Dict[str, str]:
    mapping = {
        "hp": "hp",
        "attack": "attack",
        "defense": "defense",
        "sp_attack": "sp_attack" if "sp_attack" in team_df.columns else "special-attack",
        "sp_defense": "sp_defense" if "sp_defense" in team_df.columns else "special-defense",
        "speed": "speed",
    }
    return mapping


def _assign_role(row: pd.Series, stat_cols: Dict[str, str]) -> str:
    archetype = str(row.get("archetype", "")).lower()
    speed = row[stat_cols["speed"]]
    attack = row[stat_cols["attack"]]
    sp_attack = row[stat_cols["sp_attack"]]
    defense = row[stat_cols["defense"]]
    sp_defense = row[stat_cols["sp_defense"]]
    hp = row[stat_cols["hp"]]

    # Respect archetype-level defensive labeling to avoid wall false-negatives.
    if archetype == "defensive tank":
        return "wall"

    if speed >= 100 and max(attack, sp_attack) >= 100:
        return "sweeper"
    if max(defense, sp_defense) >= 100 and hp >= 80:
        return "wall"
    if speed >= 90 and max(defense, sp_defense) >= 80:
        return "pivot"
    if attack >= 90 and sp_attack >= 90:
        return "mixed"
    return "balanced"

def _build_pivot_reasons(row: pd.Series) -> List[str]:
    reasons: List[str] = []
    if float(row.get("pivot_bulk_score", 0.0)) >= 0.65:
        reasons.append("strong bulk")
    if float(row.get("pivot_speed_score", 0.0)) >= 0.65:
        reasons.append("usable speed control")
    if float(row.get("pivot_offense_score", 0.0)) >= 0.45:
        reasons.append("threatens progress")
    if float(row.get("pivot_type_utility_score", 0.0)) >= 0.55:
        reasons.append("useful defensive typing")
    if float(row.get("pivot_ability_bonus", 0.0)) > 0:
        reasons.append("pivot-friendly ability")
    return reasons or ["balanced stat profile"]


def _load_ability_lookup() -> Dict[str, List[str]]:
    """Load local ability dataset for CLI Pokemon profile display."""
    if not ABILITY_DATA_PATH.exists():
        return {}

    try:
        df = pd.read_csv(ABILITY_DATA_PATH)
    except Exception:
        return {}

    if "name" not in df.columns:
        return {}

    ability_cols = [col for col in df.columns if "ability" in col.lower()]
    lookup: Dict[str, List[str]] = {}

    for _, row in df.iterrows():
        key = _normalize_name(str(row.get("name", "")))
        if not key:
            continue

        abilities: List[str] = []
        for col in ability_cols:
            raw = str(row.get(col, "")).strip()
            if not raw or raw.lower() == "nan":
                continue
            name = raw.lower().replace("-", " ")
            if name not in abilities:
                abilities.append(name)

        lookup[key] = abilities

    return lookup


def _estimate_ability_bonus_from_names(abilities: List[str] | None) -> float:
    """Mirror GA ability-bonus weights for profile explainability."""
    if not abilities:
        return 0.0

    ability_text = " ".join(abilities).lower()
    bonus = 0.0
    if "regenerator" in ability_text:
        bonus += 0.15
    if "intimidate" in ability_text:
        bonus += 0.12
    if "poison heal" in ability_text:
        bonus += 0.10
    if "natural cure" in ability_text:
        bonus += 0.08
    if "storm drain" in ability_text:
        bonus += 0.05
    if "water absorb" in ability_text:
        bonus += 0.04
    if "volt absorb" in ability_text:
        bonus += 0.04
    if "sap sipper" in ability_text:
        bonus += 0.04
    if "flash fire" in ability_text:
        bonus += 0.04
    if "levitate" in ability_text:
        bonus += 0.05
    if "magic guard" in ability_text:
        bonus += 0.05
    return min(0.18, bonus)


def _collect_pivot_candidates(team_df: pd.DataFrame, threshold: float = PIVOT_CANDIDATE_THRESHOLD) -> List[Dict[str, object]]:
    if "pivot_score" not in team_df.columns:
        return []

    candidates: List[Dict[str, object]] = []
    for _, row in team_df.sort_values("pivot_score", ascending=False).iterrows():
        score = float(row.get("pivot_score", 0.0))
        if score < threshold:
            continue
        candidates.append(
            {
                "pokemon": row["name"],
                "score": score,
                "style": str(row.get("pivot_style_hint", "hybrid")),
                "reasons": _build_pivot_reasons(row),
            }
        )
    return candidates


def analyze_team_by_names(team_names: List[str], pokemon_df: pd.DataFrame) -> Dict:
    team_df = pokemon_df[pokemon_df["name"].isin(team_names)].copy()
    if len(team_df) != 6:
        raise ValueError("Could not resolve all six Pokemon names.")

    stat_cols = _get_stat_columns(team_df)

    weakness_counts = Counter()
    resistance_counts = Counter()
    for _, pokemon in team_df.iterrows():
        defending_types = [pokemon["type1"]]
        if pd.notna(pokemon.get("type2")) and pokemon.get("type2"):
            defending_types.append(pokemon["type2"])

        for attacking_type in TYPE_NAMES:
            effectiveness = get_type_effectiveness(attacking_type, defending_types)
            if effectiveness >= 2.0:
                weakness_counts[attacking_type] += 1
            elif effectiveness <= 0.5:
                resistance_counts[attacking_type] += 1

    weak_to = [t for t, count in weakness_counts.items() if count >= 3]
    max_shared_weakness = max(weakness_counts.values()) if weakness_counts else 0
    type_score = max(0, 100 - (max_shared_weakness * 15))

    stat_view = team_df[
        [
            stat_cols["hp"],
            stat_cols["attack"],
            stat_cols["defense"],
            stat_cols["sp_attack"],
            stat_cols["sp_defense"],
            stat_cols["speed"],
        ]
    ]
    stat_averages = stat_view.mean().to_dict()
    fastest_idx = team_df[stat_cols["speed"]].idxmax()
    slowest_idx = team_df[stat_cols["speed"]].idxmin()
    fastest = team_df.loc[fastest_idx]
    slowest = team_df.loc[slowest_idx]
    speed_gap = int(fastest[stat_cols["speed"]] - slowest[stat_cols["speed"]])
    stat_balance_score = max(0, 100 - stat_view.mean(axis=0).std())

    roles = []
    for _, pokemon in team_df.iterrows():
        roles.append({"pokemon": pokemon["name"], "role": _assign_role(pokemon, stat_cols)})
    role_counts = Counter([r["role"] for r in roles])
    max_role_count = max(role_counts.values()) if role_counts else 0
    role_diversity_score = max(0, 100 - (max_role_count - 2) * 20)

    recommendations = []
    if weak_to:
        recommendations.append(
            "Shared weaknesses detected: " + ", ".join(weak_to) + ". Consider adding resists."
        )
    if speed_gap > 80:
        recommendations.append("Large speed gap detected. Consider speed control options.")
    if stat_averages[stat_cols["speed"]] < 70:
        recommendations.append("Team speed is low on average. Fast matchups may be difficult.")
    if "sweeper" not in role_counts:
        recommendations.append("No dedicated sweeper detected.")
    if "wall" not in role_counts:
        recommendations.append("No dedicated wall detected.")
    if role_diversity_score < 50:
        recommendations.append("Role diversity is low.")
    if not recommendations:
        recommendations.append("Team looks balanced overall.")

    # Severity issues section (kept separate from recommendations)
    team_issues: List[Dict[str, str | int]] = []
    for type_name, count in sorted(weakness_counts.items(), key=lambda item: item[1], reverse=True):
        if count < 3:
            continue
        score = min(10, count * 2)
        team_issues.append(
            {
                "issue": f"{type_name.title()} weakness",
                "severity": _severity_from_issue_score(score),
                "score": score,
                "detail": f"{count}/6 members are weak to {type_name}.",
            }
        )

    speed_issue_score = 0
    if speed_gap >= 100:
        speed_issue_score = 8
    elif speed_gap >= 80:
        speed_issue_score = 6
    elif speed_gap >= 60:
        speed_issue_score = 4
    if stat_averages[stat_cols["speed"]] < 70:
        speed_issue_score = max(speed_issue_score, 5)
    if speed_issue_score > 0:
        team_issues.append(
            {
                "issue": "Speed imbalance",
                "severity": _severity_from_issue_score(speed_issue_score),
                "score": speed_issue_score,
                "detail": f"Speed gap={speed_gap}, avg speed={stat_averages[stat_cols['speed']]:.1f}.",
            }
        )

    if max_role_count >= 4:
        role_issue_score = 8
    elif max_role_count == 3:
        role_issue_score = 5
    else:
        role_issue_score = 0
    if role_issue_score > 0:
        dominant_role = role_counts.most_common(1)[0][0]
        team_issues.append(
            {
                "issue": "Role concentration",
                "severity": _severity_from_issue_score(role_issue_score),
                "score": role_issue_score,
                "detail": f"{max_role_count}/6 members classified as {dominant_role}.",
            }
        )

    # Defensive synergy map: per-Pokemon weakness coverage by teammates
    defensive_synergy = []
    for idx, pokemon in team_df.iterrows():
        pokemon_name = pokemon["name"]
        defending_types = [pokemon["type1"]]
        if pd.notna(pokemon.get("type2")) and pokemon.get("type2"):
            defending_types.append(pokemon["type2"])

        weak_types = []
        for attacking_type in TYPE_NAMES:
            if get_type_effectiveness(attacking_type, defending_types) > 1.0:
                weak_types.append(attacking_type)

        covered = 0
        for weak_type in weak_types:
            covered_by_teammate = False
            for jdx, teammate in team_df.iterrows():
                if jdx == idx:
                    continue
                teammate_types = [teammate["type1"]]
                if pd.notna(teammate.get("type2")) and teammate.get("type2"):
                    teammate_types.append(teammate["type2"])
                if get_type_effectiveness(weak_type, teammate_types) <= 0.5:
                    covered_by_teammate = True
                    break
            if covered_by_teammate:
                covered += 1

        total_weak = len(weak_types)
        coverage_ratio = float(covered / total_weak) if total_weak > 0 else 1.0
        defensive_synergy.append(
            {
                "pokemon": pokemon_name,
                "covered_weaknesses": covered,
                "total_weaknesses": total_weak,
                "coverage_ratio": coverage_ratio,
            }
        )

    # Speed tier audit
    speed_values = team_df[stat_cols["speed"]].astype(float)
    speed_tiers = {
        "120+": int((speed_values >= 120).sum()),
        "100-119": int(((speed_values >= 100) & (speed_values < 120)).sum()),
        "80-99": int(((speed_values >= 80) & (speed_values < 100)).sum()),
        "<80": int((speed_values < 80).sum()),
    }

    # Offensive pressure profile from STAB typing
    offensive_pressure_counts = Counter()
    for _, pokemon in team_df.iterrows():
        attack_types = [pokemon["type1"]]
        if pd.notna(pokemon.get("type2")) and pokemon.get("type2"):
            attack_types.append(pokemon["type2"])
        for target_type in TYPE_NAMES:
            if any(get_type_effectiveness(atk_type, [target_type]) >= 2.0 for atk_type in attack_types):
                offensive_pressure_counts[target_type] += 1

    low_pressure_types = [t for t in TYPE_NAMES if offensive_pressure_counts.get(t, 0) == 0]
    if len(low_pressure_types) >= 4:
        pressure_score = 6
        team_issues.append(
            {
                "issue": "Low offensive pressure",
                "severity": _severity_from_issue_score(pressure_score),
                "score": pressure_score,
                "detail": "No super-effective STAB pressure into: " + ", ".join(low_pressure_types[:6]),
            }
        )

    pivot_candidates = _collect_pivot_candidates(team_df)
    pivot_series = pd.to_numeric(team_df.get("pivot_score", pd.Series(dtype=float)), errors="coerce")
    pivot_summary = {
        "candidate_count": len(pivot_candidates),
        "top_score": float(pivot_series.max()) if len(pivot_series) else 0.0,
        "styles": dict(Counter(entry["style"] for entry in pivot_candidates)) if pivot_candidates else {},
    }
    if len(pivot_candidates) == 0:
        team_issues.append(
            {
                "issue": "Limited pivot presence",
                "severity": "MEDIUM",
                "score": 5,
                "detail": "No team member cleared the pivot candidate threshold.",
            }
        )
        recommendations.append("Consider adding a bulky momentum piece or flexible switch-in to improve pivot flow.")
    elif len(pivot_candidates) >= 3:
        recommendations.append("Pivot depth looks good; the team has multiple workable switch-and-pressure options.")

    # Keep highest-impact items first
    team_issues = sorted(team_issues, key=lambda item: int(item["score"]), reverse=True)

    overall_score = int(type_score * 0.4 + stat_balance_score * 0.3 + role_diversity_score * 0.3)

    return {
        "team": team_df[["name", "type1", "type2"]].to_dict("records"),
        "type": {
            "score": int(type_score),
            "weak_to": weak_to,
            "weakness_counts": dict(weakness_counts),
            "resistances": [t for t, c in resistance_counts.items() if c >= 3],
        },
        "stats": {
            "balance_score": float(stat_balance_score),
            "avg_speed": float(stat_averages[stat_cols["speed"]]),
            "speed_gap": speed_gap,
            "fastest": {"name": fastest["name"], "speed": int(fastest[stat_cols["speed"]])},
            "slowest": {"name": slowest["name"], "speed": int(slowest[stat_cols["speed"]])},
        },
        "roles": {
            "distribution": dict(role_counts),
            "entries": roles,
            "diversity_score": int(role_diversity_score),
        },
        "issues": team_issues,
        "advanced": {
            "defensive_synergy": defensive_synergy,
            "speed_tiers": speed_tiers,
            "pivot_summary": pivot_summary,
            "pivot_candidates": pivot_candidates,
            "offensive_pressure": {
                "counts": dict(offensive_pressure_counts),
                "low_pressure_types": low_pressure_types,
            },
        },
        "recommendations": recommendations,
        "overall_score": overall_score,
    }


def interactive_team_generator() -> None:
    _print_header("MENU 1: TEAM GENERATOR")
    print("Pick one or more anchor Pokemon, then choose a team composition style.")
    print()

    base_pokemon_df = load_pokemon_data()

    anchors = _select_anchor_pokemon(base_pokemon_df)
    if not anchors:
        print("Input interrupted. Returning to menu.")
        return

    composition_name, target_counts, composition_weight = _select_composition_target(base_pokemon_df)
    power_mode_name, power_mode_cfg = _select_team_power_mode()
    pokemon_df = base_pokemon_df

    # Optional: GA tuning parameters
    print()
    generations = _input_int("How many GA generations?", 10, 300, 80)
    population = _input_int("Population size?", 20, 500, 150)
    seed = _input_int("Random seed?", 0, 999999, 42)
    save_outputs = _input_yes_no("Save run artifacts?", default_yes=False)

    print("\nOptimizing team around:", ", ".join(anchors))
    print("Composition style:", composition_name.replace("_", " ").title())
    print("Power mode:", power_mode_name.replace("_", " ").title())
    if power_mode_name == "competitive_strict":
        print("Note: Competitive Strict can trade a little peak fitness for healthier team power balance.")
    
    config = get_config_c()
    config["population"]["size"] = population
    config["population"]["generations"] = generations
    config["random_seed"] = seed
    config["name"] = f"Menu_{composition_name.title()}"
    config["fitness"]["composition_weight"] = composition_weight
    config["fitness"]["target_archetype_counts"] = target_counts
    config["fitness"]["bst_cap"] = int(power_mode_cfg["bst_cap"])
    config["fitness"]["bst_penalty_weight"] = float(power_mode_cfg["bst_penalty_weight"])
    config["fitness"]["pivot_weight"] = 0.0
    config["fitness"]["target_pivot_count"] = 0
    config["fitness"]["pivot_threshold"] = PIVOT_CANDIDATE_THRESHOLD
    if composition_name == "pivot_pressure":
        config["fitness"]["pivot_weight"] = 0.18
        config["fitness"]["target_pivot_count"] = 3

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = None
    if save_outputs:
        anchor_tag = "_".join(a.replace(" ", "") for a in anchors[:3])
        output_dir = PROJ_ROOT / "reports" / "ga_results" / f"run_anchor_{anchor_tag}_{composition_name}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

    ga = PokemonGA(
        pokemon_df=pokemon_df,
        config=config,
        output_dir=output_dir,
        locked_pokemon=anchors,
    )
    history_df = ga.run()
    
    # Get top recommended team (always just 1)
    top_teams = _get_unique_best_teams(ga, 1, max_overlap=6, allow_backfill=True)
    if not top_teams:
        print("ERROR: Could not generate a team. Try again.")
        return
    
    team_df, fitness, breakdown = top_teams[0]

    # Display recommended team with full analysis
    print("\n" + "=" * 80)
    print(f"RECOMMENDED TEAM (fitness={fitness:.4f})")
    print("=" * 80)
    for idx, (_, pokemon) in enumerate(team_df.iterrows(), 1):
        type2 = pokemon.get("type2")
        t2 = f"/{type2}" if pd.notna(type2) and str(type2).strip() else ""
        archetype = pokemon.get("archetype", "Unknown")
        # Calculate BST from individual stat columns
        stat_cols = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']
        bst = sum(pokemon.get(col, 0) for col in stat_cols)
        print(f"{idx}. {pokemon['name']:12s} ({pokemon['type1']}{t2:10s}) [{archetype:20s}] BST={int(bst)}")

    print("\nFitness breakdown:")
    for key, value in breakdown.items():
        if isinstance(value, (float, int)):
            print(f"  {key:25s}: {value:8.4f}")

    # Optionally analyze the team
    analyze = _input_yes_no("\nAnalyze this team in detail?", default_yes=True)
    analysis = None
    if analyze:
        analysis = analyze_team_by_names([p["name"] for p in team_df.to_dict("records")], pokemon_df)
        print("\n" + "-" * 80)
        print("TEAM ANALYSIS")
        print("-" * 80)
        print(f"Overall Score: {analysis['overall_score']}/100")
        print(f"\nType Coverage:")
        print(f"  Weak to: {', '.join(analysis['type']['weak_to']) if analysis['type']['weak_to'] else 'Nothing strong'}")
        print(f"  Resists: {', '.join(analysis['type']['resistances']) if analysis['type']['resistances'] else 'Nothing notable'}")
        print(f"\nStats:")
        print(f"  Avg Speed: {analysis['stats']['avg_speed']:.1f}")
        print(f"  Speed Range: {analysis['stats']['slowest']['name']} ({analysis['stats']['slowest']['speed']}) to {analysis['stats']['fastest']['name']} ({analysis['stats']['fastest']['speed']})")
        print(f"  Balance Score: {analysis['stats']['balance_score']:.1f}/100")
        print(f"\nRoles: {', '.join(f'{role}: {count}' for role, count in analysis['roles']['distribution'].items())}")
        print(f"Role Diversity: {analysis['roles']['diversity_score']}/100")
        print("\nTeam Issues (Severity):")
        if analysis["issues"]:
            for issue in analysis["issues"]:
                print(f"  {issue['issue']:28s} severity: {issue['severity']:6s} ({issue['detail']})")
        else:
            print("  No major issues detected.")

        print("\nAdvanced Snapshot:")
        speed_tiers = analysis["advanced"]["speed_tiers"]
        print(
            "  Speed tiers -> "
            f"120+: {speed_tiers['120+']}, "
            f"100-119: {speed_tiers['100-119']}, "
            f"80-99: {speed_tiers['80-99']}, "
            f"<80: {speed_tiers['<80']}"
        )
        pivot_summary = analysis["advanced"]["pivot_summary"]
        print(
            "  Pivot candidates -> "
            f"count: {pivot_summary['candidate_count']}, "
            f"top score: {pivot_summary['top_score']:.2f}"
        )
        print(
            f"  Pivot score threshold: {PIVOT_CANDIDATE_THRESHOLD:.2f} "
            "(bulk + speed + pressure + defensive utility)"
        )
        if analysis["advanced"]["pivot_candidates"]:
            for entry in analysis["advanced"]["pivot_candidates"][:3]:
                print(
                    f"    - {entry['pokemon']}: {entry['style']} pivot ({entry['score']:.2f}) "
                    f"[{', '.join(entry['reasons'][:3])}]"
                )
        low_pressure = analysis["advanced"]["offensive_pressure"]["low_pressure_types"]
        if low_pressure:
            print("  Low pressure into types:", ", ".join(low_pressure[:8]))
        else:
            print("  Offensive pressure: broad.")
        if analysis['recommendations']:
            print(f"\nRecommendations:")
            for rec in analysis['recommendations']:
                print(f"  * {rec}")

    _prompt_save_generated_team(
        team_df=team_df,
        fitness=fitness,
        breakdown=breakdown,
        config_name=config["name"],
        composition_name=composition_name,
        power_mode=power_mode_name,
        analysis=analysis,
    )

    if output_dir is not None:
        # Print relative path from project root
        rel_path = output_dir.relative_to(PROJ_ROOT)
        print(f"\nResults saved to: {rel_path}")
    
    print("\nRun again with a different Pokémon anchor for a different team!")


def interactive_random_generator() -> None:
    _print_header("MENU 3: RANDOM TEAM GENERATOR")
    print("Generate fun, creative teams with underrepresented Pokemon.")
    print()

    pokemon_df = load_pokemon_data()

    # Optional: Get 1 anchor Pokémon (but not required)
    print("(Optional) Enter a Pokémon name if you want to anchor your team around it.")
    print("Leave blank to generate a fully random team.")
    anchor = None
    while True:
        raw_input = _safe_input("Pokémon anchor (or press Enter to skip): ")
        if raw_input is None:
            print("Input interrupted. Returning to menu.")
            return
        if raw_input.strip() == "":
            break
        anchor = _resolve_pokemon_name(raw_input.strip(), pokemon_df)
        if not anchor:
            print("Pokemon not found. Try again.")
            continue
        break

    # GA tuning parameters
    print()
    generations = _input_int("How many GA generations?", 10, 300, 80)
    population = _input_int("Population size?", 20, 500, 150)
    seed = _input_int("Random seed?", 0, 999999, 42)
    save_outputs = _input_yes_no("Save run artifacts?", default_yes=False)

    locked = [anchor] if anchor else []
    print(f"\nGenerating random team {'around ' + anchor if anchor else ''}...")
    
    config = get_config_random()
    config["population"]["size"] = population
    config["population"]["generations"] = generations
    config["random_seed"] = seed

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = None
    if save_outputs:
        anchor_str = anchor if anchor else "any"
        output_dir = PROJ_ROOT / "reports" / "ga_results" / f"run_random_{anchor_str}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

    ga = PokemonGA(
        pokemon_df=pokemon_df,
        config=config,
        output_dir=output_dir,
        locked_pokemon=locked,
    )
    history_df = ga.run()
    
    # Get top recommended team (always just 1)
    top_teams = _get_unique_best_teams(ga, 1, max_overlap=6, allow_backfill=True)
    if not top_teams:
        print("ERROR: Could not generate a team. Try again.")
        return
    
    team_df, fitness, breakdown = top_teams[0]

    # Display recommended team with full analysis
    print("\n" + "=" * 80)
    print(f"RANDOM TEAM GENERATED (fitness={fitness:.4f})")
    print("=" * 80)
    for idx, (_, pokemon) in enumerate(team_df.iterrows(), 1):
        type2 = pokemon.get("type2")
        t2 = f"/{type2}" if pd.notna(type2) and str(type2).strip() else ""
        archetype = pokemon.get("archetype", "Unknown")
        # Calculate BST from individual stat columns
        stat_cols = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']
        bst = sum(pokemon.get(col, 0) for col in stat_cols)
        print(f"{idx}. {pokemon['name']:12s} ({pokemon['type1']}{t2:10s}) [{archetype:20s}] BST={int(bst)}")

    print("\nFitness breakdown:")
    for key, value in breakdown.items():
        if isinstance(value, (float, int)):
            print(f"  {key:25s}: {value:8.4f}")

    # Optionally analyze the team
    analyze = _input_yes_no("\nAnalyze this team in detail?", default_yes=False)
    analysis = None
    if analyze:
        analysis = analyze_team_by_names([p["name"] for p in team_df.to_dict("records")], pokemon_df)
        print("\n" + "-" * 80)
        print("TEAM ANALYSIS")
        print("-" * 80)
        print(f"Overall Score: {analysis['overall_score']}/100")
        print(f"\nType Coverage:")
        print(f"  Weak to: {', '.join(analysis['type']['weak_to']) if analysis['type']['weak_to'] else 'Nothing strong'}")
        print(f"  Resists: {', '.join(analysis['type']['resistances']) if analysis['type']['resistances'] else 'Nothing notable'}")
        print(f"\nStats:")
        print(f"  Avg Speed: {analysis['stats']['avg_speed']:.1f}")
        print(f"  Speed Range: {analysis['stats']['slowest']['name']} ({analysis['stats']['slowest']['speed']}) to {analysis['stats']['fastest']['name']} ({analysis['stats']['fastest']['speed']})")
        print(f"  Balance Score: {analysis['stats']['balance_score']:.1f}/100")
        print(f"\nRoles: {', '.join(f'{role}: {count}' for role, count in analysis['roles']['distribution'].items())}")
        print(f"Role Diversity: {analysis['roles']['diversity_score']}/100")
        if analysis['recommendations']:
            print(f"\nRecommendations:")
            for rec in analysis['recommendations']:
                print(f"  * {rec}")

    _prompt_save_generated_team(
        team_df=team_df,
        fitness=fitness,
        breakdown=breakdown,
        config_name=config["name"],
        composition_name="random",
        power_mode="open",
        analysis=analysis,
    )

    if output_dir is not None:
        # Print relative path from project root
        rel_path = output_dir.relative_to(PROJ_ROOT)
        print(f"\nResults saved to: {rel_path}")
    
    print("\nRun again to generate a new random team!")


def interactive_team_analyzer() -> None:
    _print_header("MENU 2: TEAM ANALYZER")
    print("Analyze role, stat, and type weaknesses for a team of 6 Pokemon.")

    pokemon_df = load_pokemon_data()
    selected: List[str] = []

    for slot in range(1, 7):
        while True:
            raw_input = _safe_input(f"Enter Pokemon #{slot} name: ")
            if raw_input is None:
                print("Input interrupted. Returning to menu.")
                return
            raw = raw_input.strip()
            resolved = _resolve_pokemon_name(raw, pokemon_df)
            if not resolved:
                print("Pokemon not found. Try again with exact or distinctive partial name.")
                continue
            if resolved in selected:
                print("Duplicate Pokemon not allowed. Choose a different Pokemon.")
                continue
            selected.append(resolved)
            break

    report = analyze_team_by_names(selected, pokemon_df)

    print("\n" + "-" * 80)
    print("TEAM")
    print("-" * 80)
    for idx, p in enumerate(report["team"], 1):
        type2_val = p.get("type2")
        t2 = f"/{type2_val}" if type2_val and str(type2_val).lower() != "nan" else ""
        print(f"{idx}. {p['name']} ({p['type1']}{t2})")

    print("\n" + "-" * 80)
    print("TYPE ANALYSIS")
    print("-" * 80)
    print(f"Type score: {report['type']['score']}/100")
    if report["type"]["weak_to"]:
        print("Shared weaknesses: " + ", ".join(report["type"]["weak_to"]))
    else:
        print("No major shared weaknesses (3+ members weak to same type).")

    print("\n" + "-" * 80)
    print("STAT ANALYSIS")
    print("-" * 80)
    print(f"Stat balance score: {report['stats']['balance_score']:.1f}/100")
    print(f"Average speed: {report['stats']['avg_speed']:.1f}")
    print(f"Speed gap: {report['stats']['speed_gap']}")
    print(
        f"Fastest: {report['stats']['fastest']['name']} ({report['stats']['fastest']['speed']}) | "
        f"Slowest: {report['stats']['slowest']['name']} ({report['stats']['slowest']['speed']})"
    )

    print("\n" + "-" * 80)
    print("ROLE ANALYSIS")
    print("-" * 80)
    print("Distribution:", report["roles"]["distribution"])
    print(f"Role diversity score: {report['roles']['diversity_score']}/100")

    print("\n" + "-" * 80)
    print("TEAM ISSUES (SEVERITY)")
    print("-" * 80)
    if report["issues"]:
        for issue in report["issues"]:
            print(f"{issue['issue']:28s} severity: {issue['severity']:6s} | {issue['detail']}")
    else:
        print("No major issues detected.")

    print("\n" + "-" * 80)
    print("ADVANCED SNAPSHOT")
    print("-" * 80)
    speed_tiers = report["advanced"]["speed_tiers"]
    print(
        f"Speed tiers: 120+={speed_tiers['120+']}, "
        f"100-119={speed_tiers['100-119']}, "
        f"80-99={speed_tiers['80-99']}, <80={speed_tiers['<80']}"
    )
    pivot_summary = report["advanced"]["pivot_summary"]
    print(
        f"Pivot candidates: count={pivot_summary['candidate_count']}, "
        f"top score={pivot_summary['top_score']:.2f}"
    )
    print(
        f"Pivot score threshold: {PIVOT_CANDIDATE_THRESHOLD:.2f} "
        "(bulk + speed + pressure + defensive utility)"
    )
    if report["advanced"]["pivot_candidates"]:
        for entry in report["advanced"]["pivot_candidates"]:
            print(
                f"  {entry['pokemon']:12s} {entry['style']:5s} pivot {entry['score']:.2f} | "
                + ", ".join(entry["reasons"][:3])
            )
    low_pressure = report["advanced"]["offensive_pressure"]["low_pressure_types"]
    if low_pressure:
        print("Low offensive pressure into:", ", ".join(low_pressure[:8]))
    else:
        print("Offensive pressure coverage is broad.")
    print("Defensive synergy by Pokemon:")
    for entry in report["advanced"]["defensive_synergy"]:
        print(
            f"  {entry['pokemon']:12s} "
            f"covered weaknesses {entry['covered_weaknesses']}/{entry['total_weaknesses']} "
            f"({entry['coverage_ratio'] * 100:.0f}%)"
        )

    print("\n" + "-" * 80)
    print("RECOMMENDATIONS")
    print("-" * 80)
    for rec in report["recommendations"]:
        print(f"- {rec}")

    print(f"\nOverall team score: {report['overall_score']}/100")


def interactive_pokemon_info() -> None:
    _print_header("MENU 5: POKEMON INFO")
    print("View typing, stats, archetype, and pivot profile for one Pokemon.")

    pokemon_df = load_pokemon_data()
    stat_cols = _get_stat_columns(pokemon_df)
    ability_lookup = _load_ability_lookup()

    while True:
        raw_input = _safe_input("Enter Pokemon name (or press Enter to return): ")
        if raw_input is None:
            print("Input interrupted. Returning to menu.")
            return

        raw = raw_input.strip()
        if not raw:
            return

        resolved = _resolve_pokemon_name(raw, pokemon_df)
        if not resolved:
            print("Pokemon not found. Try exact name or a distinctive partial name.")
            continue

        row = pokemon_df[pokemon_df["name"] == resolved].iloc[0]

        type2_val = row.get("type2")
        type_suffix = f"/{type2_val}" if pd.notna(type2_val) and str(type2_val).strip() else ""

        hp = int(row.get(stat_cols["hp"], 0))
        attack = int(row.get(stat_cols["attack"], 0))
        defense = int(row.get(stat_cols["defense"], 0))
        sp_attack = int(row.get(stat_cols["sp_attack"], 0))
        sp_defense = int(row.get(stat_cols["sp_defense"], 0))
        speed = int(row.get(stat_cols["speed"], 0))
        bst = hp + attack + defense + sp_attack + sp_defense + speed

        archetype = str(row.get("archetype", "Unknown"))
        cluster = row.get("cluster", "N/A")
        role_hint = _assign_role(row, stat_cols)

        pivot_score = float(pd.to_numeric(pd.Series([row.get("pivot_score", 0.0)]), errors="coerce").fillna(0.0).iloc[0])
        pivot_style = str(row.get("pivot_style_hint", "hybrid"))
        pivot_ability_bonus = float(row.get("pivot_ability_bonus", 0.0))

        abilities = ability_lookup.get(_normalize_name(resolved))
        ability_source = "local dataset"
        ability_bonus_estimate = _estimate_ability_bonus_from_names(abilities)

        print("\n" + "-" * 80)
        print(f"POKEMON PROFILE: {resolved}")
        print("-" * 80)
        print(f"Typing: {row.get('type1', 'Unknown')}{type_suffix}")
        print(f"Archetype: {archetype}")
        print(f"Cluster: {cluster}")
        print(f"Role hint: {role_hint}")
        print(
            "Stats: "
            f"HP={hp}, Atk={attack}, Def={defense}, SpA={sp_attack}, SpD={sp_defense}, Spe={speed}, BST={bst}"
        )
        if abilities:
            print(f"Abilities ({ability_source}):", ", ".join(abilities))
        else:
            print(
                "Abilities: unavailable "
                f"(expected at {ABILITY_DATA_PATH.relative_to(PROJ_ROOT)})"
            )

        if "pivot_score" in pokemon_df.columns:
            candidate_text = "yes" if pivot_score >= PIVOT_CANDIDATE_THRESHOLD else "no"
            print("\nPivot Profile:")
            print(f"  Pivot score: {pivot_score:.2f} (candidate: {candidate_text})")
            print(f"  Style hint: {pivot_style}")
            print(f"  Bulk score: {float(row.get('pivot_bulk_score', 0.0)):.2f}")
            print(f"  Speed score: {float(row.get('pivot_speed_score', 0.0)):.2f}")
            print(f"  Offense score: {float(row.get('pivot_offense_score', 0.0)):.2f}")
            print(f"  Type utility score: {float(row.get('pivot_type_utility_score', 0.0)):.2f}")
            print(f"  Profile score: {float(row.get('pivot_profile_score', 0.0)):.2f}")
            print(f"  Ability bonus (dataset): {pivot_ability_bonus:.2f}")
            if abilities is not None:
                print(f"  Ability bonus (from abilities): {ability_bonus_estimate:.2f}")
            print("  Why:", ", ".join(_build_pivot_reasons(row)))

        again = _input_yes_no("\nLook up another Pokemon?", default_yes=True)
        if not again:
            return


def run_interactive_menu() -> int:
    def _pause_to_menu() -> bool:
        raw = _safe_input("\nPress Enter to return to menu...")
        return raw is not None

    while True:
        _print_header("POKEMON TEAM SYSTEM")
        print("Choose an option:")
        print("1. Team Generator (anchor-based)")
        print("2. Team Analyzer")
        print("3. Random Team Generator")
        print("4. View Saved Teams")
        print("5. Pokemon Info")
        print("6. Exit")

        raw_choice = _safe_input("Enter choice [1-6]: ")
        if raw_choice is None:
            print("\nInput interrupted. Exiting.")
            return 0
        choice = raw_choice.strip()
        if choice == "1":
            _run_with_error_guard("Team Generator", interactive_team_generator)
            if not _pause_to_menu():
                print("\nInput interrupted. Exiting.")
                return 0
        elif choice == "2":
            _run_with_error_guard("Team Analyzer", interactive_team_analyzer)
            if not _pause_to_menu():
                print("\nInput interrupted. Exiting.")
                return 0
        elif choice == "3":
            _run_with_error_guard("Random Team Generator", interactive_random_generator)
            if not _pause_to_menu():
                print("\nInput interrupted. Exiting.")
                return 0
        elif choice == "4":
            _run_with_error_guard("View Saved Teams", _print_saved_teams)
            if not _pause_to_menu():
                print("\nInput interrupted. Exiting.")
                return 0
        elif choice == "5":
            _run_with_error_guard("Pokemon Info", interactive_pokemon_info)
            if not _pause_to_menu():
                print("\nInput interrupted. Exiting.")
                return 0
        elif choice == "6":
            print("Goodbye.")
            return 0
        else:
            print("Invalid choice. Please enter 1, 2, 3, 4, 5, or 6.")


def run_cluster(args: argparse.Namespace) -> int:
    _print_header("CLI: CLUSTER WORKFLOW")

    clustering_script = PROJ_ROOT / "legacy" / "src" / "analysis" / "clustering.py"
    deliverables_script = (
        PROJ_ROOT
        / "reports"
        / "clustering_analysis"
        / "scripts"
        / "01_generate_phase1_deliverables.py"
    )
    validation_script = PROJ_ROOT / "legacy" / "scripts" / "validation" / "validate_601_clustering.py"

    print("[1/3] Running clustering pipeline...")
    rc = _run_python_script(clustering_script)
    if rc != 0:
        print(f"[ERROR] Clustering pipeline failed with code {rc}")
        return rc

    if args.with_deliverables:
        print("[2/3] Generating phase 1 deliverables...")
        rc = _run_python_script(deliverables_script)
        if rc != 0:
            print(f"[ERROR] Deliverable generation failed with code {rc}")
            return rc
    else:
        print("[2/3] Skipped deliverable generation")

    if args.with_validation:
        print("[3/3] Running role validation...")
        rc = _run_python_script(validation_script)
        if rc != 0:
            print(f"[ERROR] Validation failed with code {rc}")
            return rc
    else:
        print("[3/3] Skipped validation")

    print("[OK] Cluster workflow completed")
    return 0


def run_ga(args: argparse.Namespace) -> int:
    _print_header("CLI: GA WORKFLOW")

    print("[1/4] Loading Pokemon dataset...")
    pokemon_df = load_pokemon_data()
    print(f"      Loaded {len(pokemon_df)} Pokemon")

    print("[2/4] Building configuration...")
    config = _get_config_by_name(args.config)
    config["population"]["size"] = args.population
    config["population"]["generations"] = args.generations
    config["random_seed"] = args.seed
    config["name"] = f"CLI_Config{args.config.upper()}"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = None
    if args.output_dir:
        output_dir = Path(args.output_dir)
    elif not args.no_save:
        output_dir = PROJ_ROOT / "reports" / "ga_results" / f"run_cli_{timestamp}"
    if output_dir is not None:
        output_dir.mkdir(parents=True, exist_ok=True)

    print("[3/4] Running GA evolution...")
    ga = PokemonGA(
        pokemon_df=pokemon_df,
        config=config,
        output_dir=output_dir,
        locked_pokemon=getattr(args, "locked_pokemon", None),
    )
    history_df = ga.run()

    print("[4/4] Building results...")
    top_teams = _get_unique_best_teams(ga, args.top_n)
    payload = _build_ga_result_payload(
        top_teams=top_teams,
        history_df=history_df,
        output_dir=output_dir,
        config=config,
        dataset_size=len(pokemon_df),
        top_n=args.top_n,
        timestamp=timestamp,
    )

    if output_dir is not None:
        with open(output_dir / "top_teams.json", "w", encoding="utf-8") as f:
            json.dump(payload["top_teams"], f, indent=2)

        history_df.to_csv(output_dir / "fitness_history.csv", index=False)

        with open(output_dir / "metadata.json", "w", encoding="utf-8") as f:
            json.dump(
                {
                    "timestamp": payload["timestamp"],
                    "config": payload["config"],
                    "dataset_size": payload["dataset_size"],
                    "top_n": payload["top_n"],
                    "best_fitness": payload["best_fitness"],
                    "output_dir": payload["output_dir"],
                    "history": payload["history"],
                },
                f,
                indent=2,
            )

    print("[OK] GA workflow completed")
    if output_dir is not None:
        # Print relative path from project root
        try:
            rel_path = output_dir.relative_to(PROJ_ROOT)
            print(f"     Output directory: {rel_path}")
        except ValueError:
            print(f"     Output directory: {output_dir}")
    else:
        print("     Output directory: (disabled with --no-save)")
    if payload["best_fitness"] is not None:
        print(f"     Best fitness: {payload['best_fitness']:.4f}")

    if args.json_output:
        print(json.dumps(payload, indent=2))

    return 0


def run_pipeline(args: argparse.Namespace) -> int:
    _print_header("CLI: END-TO-END PIPELINE")

    cluster_args = argparse.Namespace(
        with_deliverables=args.with_deliverables,
        with_validation=args.with_validation,
    )
    rc = run_cluster(cluster_args)
    if rc != 0:
        return rc

    ga_args = argparse.Namespace(
        config=args.config,
        population=args.population,
        generations=args.generations,
        seed=args.seed,
        top_n=args.top_n,
        output_dir=args.output_dir,
        no_save=args.no_save,
        json_output=args.json_output,
    )
    return run_ga(ga_args)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Pokemon ML CLI for clustering and GA workflows",
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    cluster_parser = subparsers.add_parser(
        "cluster",
        help="Run clustering workflow",
    )
    cluster_parser.add_argument(
        "--with-deliverables",
        action="store_true",
        help="Also run phase 1 deliverables generation script",
    )
    cluster_parser.add_argument(
        "--with-validation",
        action="store_true",
        help="Also run role validation script",
    )
    cluster_parser.set_defaults(func=run_cluster)

    ga_parser = subparsers.add_parser(
        "ga",
        help="Run GA optimization workflow",
    )
    ga_parser.add_argument("--config", choices=["A", "B", "C", "a", "b", "c"], default="C")
    ga_parser.add_argument("--population", type=int, default=300)
    ga_parser.add_argument("--generations", type=int, default=300)
    ga_parser.add_argument("--seed", type=int, default=42)
    ga_parser.add_argument("--top-n", type=int, default=10)
    ga_parser.add_argument("--output-dir", type=str, default=None)
    ga_parser.add_argument(
        "--no-save",
        action="store_true",
        help="Do not write output files; useful for app-driven runs",
    )
    ga_parser.add_argument(
        "--json-output",
        action="store_true",
        help="Print GA result payload as JSON to stdout",
    )
    ga_parser.set_defaults(func=run_ga)

    pipeline_parser = subparsers.add_parser(
        "pipeline",
        help="Run cluster then GA",
    )
    pipeline_parser.add_argument("--config", choices=["A", "B", "C", "a", "b", "c"], default="C")
    pipeline_parser.add_argument("--population", type=int, default=300)
    pipeline_parser.add_argument("--generations", type=int, default=300)
    pipeline_parser.add_argument("--seed", type=int, default=42)
    pipeline_parser.add_argument("--top-n", type=int, default=10)
    pipeline_parser.add_argument("--output-dir", type=str, default=None)
    pipeline_parser.add_argument("--no-save", action="store_true")
    pipeline_parser.add_argument("--json-output", action="store_true")
    pipeline_parser.add_argument("--with-deliverables", action="store_true")
    pipeline_parser.add_argument("--with-validation", action="store_true")
    pipeline_parser.set_defaults(func=run_pipeline)

    return parser


def main() -> int:
    if len(sys.argv) == 1:
        return run_interactive_menu()

    parser = build_parser()
    args = parser.parse_args()

    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("\n[ABORTED] Interrupted by user")
        return 130
    except Exception as exc:
        print(f"[ERROR] {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
