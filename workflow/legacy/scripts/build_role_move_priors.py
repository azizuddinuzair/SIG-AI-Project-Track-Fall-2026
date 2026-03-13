"""
Phase 3 Bootstrap: Role-Defining Move Priors from GA Teams

Purpose:
1. Build constrained canonical movesets (no "every move" noise)
2. Use only role-defining moves by category
3. Exclude setup moves by default, with small whitelist exceptions
4. Weight move frequencies by GA team fitness
5. Use Layer 1 archetypes as role priors (especially for mixed/defensive handling)

Inputs:
- Top-N GA teams JSON (ranked teams with fitness)
- pokemon_with_clusters.csv (Layer 1 archetypes + engineered features)

Outputs (under reports/ga_results/<run>/phase3_role_bootstrap/):
- canonical_movesets.json
- weighted_move_frequency.csv
- pokemon_role_priors.csv

Notes:
- This script does NOT attempt full competitive moveset optimization.
- It creates a clean, low-noise prior layer for Phase 3.
"""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Dict, List, Set, Tuple

import pandas as pd
import requests


# -----------------------------------------------------------------------------
# Move vocab (role-defining only)
# -----------------------------------------------------------------------------

# Keep this set intentionally small to reduce noise.
DEFINING_MOVES: Set[str] = {
    # Pivot / tempo
    "u-turn", "volt-switch", "flip-turn", "parting-shot", "teleport",

    # Recovery / sustain
    "recover", "roost", "slack-off", "soft-boiled", "wish", "morningsun",
    "synthesis", "moonlight", "drain-punch", "giga-drain", "shore-up",
    "baneful-bunker", "protect",

    # Hazard / utility
    "stealth-rock", "spikes", "toxic-spikes", "rapid-spin", "defog", "taunt",
    "thunder-wave", "will-o-wisp", "toxic", "haze", "encore",

    # High-signal offensive anchors (role identity)
    "wicked-blow", "dark-pulse", "hydro-pump", "surf", "draco-meteor", "shadow-ball",
    "surging-strikes",
    "moonblast", "close-combat", "earthquake", "flare-blitz", "overheat", "ice-beam",
    "thunderbolt", "psychic", "psyshock", "knock-off", "brave-bird", "body-press",
    "seismic-toss", "heat-wave", "dragon-claw", "outrage",
}

MOVE_CLASSES: Dict[str, str] = {
    # Pivot / tempo
    "u-turn": "pivot", "volt-switch": "pivot", "flip-turn": "pivot",
    "parting-shot": "pivot", "teleport": "pivot",

    # Recovery / sustain
    "recover": "recovery", "roost": "recovery", "slack-off": "recovery",
    "soft-boiled": "recovery", "wish": "recovery", "morningsun": "recovery",
    "synthesis": "recovery", "moonlight": "recovery", "shore-up": "recovery",
    "drain-punch": "offense", "giga-drain": "offense",
    "baneful-bunker": "recovery", "protect": "utility",

    # Hazard / utility
    "stealth-rock": "hazard", "spikes": "hazard", "toxic-spikes": "hazard",
    "rapid-spin": "utility", "defog": "utility", "taunt": "utility",
    "thunder-wave": "utility", "will-o-wisp": "utility", "toxic": "utility",
    "haze": "utility", "encore": "utility",

    # Offensive anchors
    "wicked-blow": "offense", "dark-pulse": "offense", "hydro-pump": "offense",
    "surging-strikes": "offense",
    "surf": "offense", "draco-meteor": "offense", "shadow-ball": "offense",
    "moonblast": "offense", "close-combat": "offense", "earthquake": "offense",
    "flare-blitz": "offense", "overheat": "offense", "ice-beam": "offense",
    "thunderbolt": "offense", "psychic": "offense", "psyshock": "offense",
    "knock-off": "offense", "brave-bird": "offense", "body-press": "offense",
    "seismic-toss": "offense", "heat-wave": "offense", "dragon-claw": "offense",
    "outrage": "offense",

    # Setup (only via whitelist)
    "swords-dance": "setup", "nasty-plot": "setup", "calm-mind": "setup",
    "dragon-dance": "setup", "shell-smash": "setup", "bulk-up": "setup",
    "quiver-dance": "setup", "agility": "setup", "curse": "setup",
    "growth": "setup", "iron-defense": "setup",
}

# Setup moves are excluded by default to avoid noisy "everything is a setup sweeper" behavior.
SETUP_MOVES: Set[str] = {
    "swords-dance", "nasty-plot", "calm-mind", "dragon-dance", "shell-smash",
    "bulk-up", "quiver-dance", "agility", "curse", "growth", "iron-defense",
}

# Whitelist setup exceptions: allow only for specific role priors.
# Extend over time with Pokemon-specific entries if needed.
SETUP_WHITELIST_BY_ROLE: Dict[str, Set[str]] = {
    "setup_sweeper": {"shell-smash", "dragon-dance", "swords-dance", "nasty-plot", "quiver-dance"},
    "special_sweeper": {"calm-mind", "nasty-plot"},
    "physical_sweeper": {"swords-dance", "dragon-dance", "bulk-up"},
    "bulky_wincon": {"calm-mind", "curse", "iron-defense"},
}

# Pokemon-specific exceptions for high-precision role signatures.
# Includes setup and non-setup signature moves.
POKEMON_MOVE_EXCEPTIONS: Dict[str, Set[str]] = {
    "cloyster": {"shell-smash"},
    "torterra": {"shell-smash"},
    "blissey": {"soft-boiled"},
    "urshifu-single-strike": {"wicked-blow"},
    "urshifu-rapid-strike": {"surging-strikes"},
}

ROLE_ALLOWED_CLASSES: Dict[str, Set[str]] = {
    "special_sweeper": {"offense", "pivot", "utility", "setup"},
    "physical_sweeper": {"offense", "pivot", "utility", "setup"},
    "defensive_wall": {"recovery", "hazard", "utility", "offense", "setup"},
    "fast_pivot": {"pivot", "offense", "utility"},
    "bulky_pivot": {"pivot", "recovery", "hazard", "utility"},
    "breaker": {"offense", "utility"},
    "bulky_wincon": {"recovery", "utility", "offense", "setup"},
    "mixed_balanced": {"offense", "pivot", "utility", "recovery"},
    "setup_sweeper": {"offense", "pivot", "utility", "setup"},
}

ROLE_MIN_MOVES: Dict[str, int] = {
    "special_sweeper": 3,
    "physical_sweeper": 3,
    "defensive_wall": 3,
    "fast_pivot": 3,
    "bulky_pivot": 3,
    "breaker": 3,
    "bulky_wincon": 3,
    "mixed_balanced": 2,
    "setup_sweeper": 3,
}


# -----------------------------------------------------------------------------
# Data loading
# -----------------------------------------------------------------------------

def load_ga_teams(
    teams_path: Path,
    top_n: int,
    teams_glob: str = "",
    generation_teams_glob: str = "",
) -> List[dict]:
    """
    Load teams from one JSON file or many JSON files matched by glob.

    - teams_glob: run-level team artifacts (e.g., top_10_teams.json across runs)
    - generation_teams_glob: optional per-generation elite snapshots (same schema expected)

    If any glob is provided, all matches are concatenated and rank-sorted by fitness.
    """
    all_teams: List[dict] = []

    def _read_team_file(path_obj: Path, source_tag: str) -> None:
        nonlocal all_teams
        try:
            with path_obj.open("r", encoding="utf-8") as f:
                data = json.load(f)
            if isinstance(data, list):
                for team in data:
                    if isinstance(team, dict):
                        tagged = dict(team)
                        tagged["_source"] = source_tag
                        all_teams.append(tagged)
        except Exception:
            return

    if teams_glob or generation_teams_glob:
        if teams_glob:
            for p in sorted(Path().glob(teams_glob)):
                _read_team_file(p, "run")
        if generation_teams_glob:
            for p in sorted(Path().glob(generation_teams_glob)):
                _read_team_file(p, "generation")
    else:
        _read_team_file(teams_path, "run")

    all_teams.sort(key=lambda t: safe_float(t.get("fitness", 0.0)), reverse=True)

    if top_n <= 0:
        return all_teams
    return all_teams[: min(top_n, len(all_teams))]


def team_signature(team: dict) -> str:
    """Canonical signature for deduplication by team composition."""
    names = sorted(str(p.get("name", "")).strip().lower() for p in team.get("pokemon", []))
    return "|".join(names)


def deduplicate_teams(teams: List[dict]) -> Tuple[List[dict], int]:
    """Keep the highest-fitness team for each identical composition signature."""
    by_sig: Dict[str, dict] = {}
    duplicates = 0
    for t in teams:
        sig = team_signature(t)
        if sig in by_sig:
            duplicates += 1
            if safe_float(t.get("fitness", 0.0)) > safe_float(by_sig[sig].get("fitness", 0.0)):
                by_sig[sig] = t
        else:
            by_sig[sig] = t
    deduped = list(by_sig.values())
    deduped.sort(key=lambda x: safe_float(x.get("fitness", 0.0)), reverse=True)
    return deduped, duplicates


def load_cluster_data(cluster_csv: Path) -> pd.DataFrame:
    df = pd.read_csv(cluster_csv)
    # Fast lookups by name
    return df


# -----------------------------------------------------------------------------
# Layer 1 role priors
# -----------------------------------------------------------------------------

def role_prior_from_row(row: pd.Series) -> str:
    """
    Convert Layer 1 archetype + engineered features into a role prior.

    This is intentionally heuristic for Phase 3 bootstrap.
    """
    archetype = str(row.get("archetype", "")).strip()
    speed_pct = float(row.get("speed_percentile", 0.5))
    off_idx = float(row.get("offensive_index", 0.0))
    def_idx = float(row.get("defensive_index", 0.0))
    bias = float(row.get("physical_special_bias", 0.0))

    if archetype == "Speed Sweeper":
        return "special_sweeper" if bias < 0 else "physical_sweeper"

    if archetype == "Fast Attacker":
        if speed_pct >= 0.85 and off_idx > def_idx:
            return "special_sweeper" if bias < 0 else "physical_sweeper"
        return "fast_pivot"

    if archetype == "Defensive Tank":
        if def_idx >= off_idx + 40:
            return "defensive_wall"
        return "bulky_wincon"

    if archetype == "Physical Attacker":
        if speed_pct >= 0.70:
            return "physical_sweeper"
        return "breaker"

    if archetype == "Balanced All-Rounder":
        # Keep mixed roles with Layer 1 to reduce compute + overfitting.
        diff = off_idx - def_idx
        if diff > 20:
            return "physical_sweeper" if bias > 0 else "special_sweeper"
        if diff < -20:
            return "bulky_pivot"
        return "mixed_balanced"

    # Generalist and fallback
    if speed_pct >= 0.65 and abs(off_idx - def_idx) < 40:
        return "fast_pivot"
    return "mixed_balanced"


# -----------------------------------------------------------------------------
# PokeAPI move retrieval
# -----------------------------------------------------------------------------

def fetch_learnable_moves(pokemon_name: str, timeout_sec: int = 15) -> Set[str]:
    """Fetch all learnable moves for a pokemon from PokeAPI."""
    url = f"https://pokeapi.co/api/v2/pokemon/{pokemon_name}"
    resp = requests.get(url, timeout=timeout_sec)
    resp.raise_for_status()
    data = resp.json()
    return {m["move"]["name"] for m in data.get("moves", [])}


def allowed_moves_for_role(
    pokemon_name: str,
    learnable_moves: Set[str],
    role_prior: str,
    include_setup: bool,
) -> Set[str]:
    """
    Keep only role-defining moves and optional setup whitelist for the assigned role.
    """
    candidates = learnable_moves.intersection(DEFINING_MOVES)

    allowed_classes = ROLE_ALLOWED_CLASSES.get(role_prior, ROLE_ALLOWED_CLASSES["mixed_balanced"])
    candidates = {m for m in candidates if MOVE_CLASSES.get(m, "utility") in allowed_classes}

    if include_setup:
        role_whitelist = SETUP_WHITELIST_BY_ROLE.get(role_prior, set())
        candidates = candidates.union(learnable_moves.intersection(role_whitelist))

    # Pokemon-level signature exceptions (high precision overrides).
    exceptions = POKEMON_MOVE_EXCEPTIONS.get(pokemon_name, set())
    candidates = candidates.union(learnable_moves.intersection(exceptions))

    # If setup not included, explicitly ensure setup moves are removed.
    candidates = candidates.difference(SETUP_MOVES)
    return candidates


# -----------------------------------------------------------------------------
# Canonical moveset construction
# -----------------------------------------------------------------------------

ROLE_MOVE_PREFERENCE: Dict[str, List[str]] = {
    "special_sweeper": [
        "draco-meteor", "shadow-ball", "moonblast", "dark-pulse", "hydro-pump", "overheat",
        "ice-beam", "thunderbolt", "psychic", "psyshock",
    ],
    "physical_sweeper": [
        "wicked-blow", "close-combat", "earthquake", "flare-blitz", "brave-bird", "knock-off",
        "dragon-claw", "outrage", "body-press",
    ],
    "defensive_wall": [
        "recover", "roost", "soft-boiled", "wish", "baneful-bunker", "haze", "toxic",
        "protect", "seismic-toss", "body-press",
    ],
    "fast_pivot": [
        "u-turn", "volt-switch", "flip-turn", "parting-shot", "taunt", "knock-off",
        "dark-pulse", "moonblast", "thunder-wave",
    ],
    "bulky_pivot": [
        "u-turn", "volt-switch", "defog", "roost", "recover", "wish", "stealth-rock", "toxic",
    ],
    "breaker": [
        "close-combat", "earthquake", "knock-off", "flare-blitz", "draco-meteor", "body-press",
    ],
    "bulky_wincon": [
        "recover", "roost", "wish", "calm-mind", "curse", "body-press", "seismic-toss", "toxic",
    ],
    "mixed_balanced": [
        "u-turn", "volt-switch", "knock-off", "earthquake", "ice-beam", "thunderbolt", "roost", "recover",
    ],
    "setup_sweeper": [
        "shell-smash", "dragon-dance", "swords-dance", "nasty-plot", "hydro-pump", "earthquake", "ice-beam",
    ],
}


def build_canonical_moveset(role_prior: str, allowed_moves: Set[str], k: int = 4) -> List[str]:
    """Pick top-k canonical moves by role preference order."""
    prefs = ROLE_MOVE_PREFERENCE.get(role_prior, ROLE_MOVE_PREFERENCE["mixed_balanced"])
    chosen: List[str] = [m for m in prefs if m in allowed_moves]

    # Restrictive fallback: do not fill arbitrarily from any move.
    # Keep only role-compatible classes and return partial/unknown if needed.
    if len(chosen) < k:
        allowed_classes = ROLE_ALLOWED_CLASSES.get(role_prior, ROLE_ALLOWED_CLASSES["mixed_balanced"])
        leftovers = [
            m for m in sorted(list(allowed_moves.difference(set(chosen))))
            if MOVE_CLASSES.get(m, "utility") in allowed_classes
        ]
        chosen.extend(leftovers)

    chosen = chosen[:k]

    # If too sparse after strict filtering, prefer explicit unknown over noisy fill.
    min_moves = ROLE_MIN_MOVES.get(role_prior, 2)
    if len(chosen) < min_moves:
        return chosen + ["unknown"] * (k - len(chosen))

    if len(chosen) < k:
        return chosen + ["unknown"] * (k - len(chosen))

    return chosen


# -----------------------------------------------------------------------------
# Weighted aggregation over GA teams
# -----------------------------------------------------------------------------

def safe_float(x: object) -> float:
    try:
        v = float(x)
        if math.isnan(v):
            return 0.0
        return v
    except Exception:
        return 0.0


def aggregate_weighted_frequency(
    teams: List[dict],
    canonical_movesets: Dict[str, List[str]],
    pokemon_archetype: Dict[str, str],
) -> pd.DataFrame:
    """
    Weighted frequency: sum team fitness for each (pokemon, move) appearance.

    Normalization uses UNIQUE TEAM OCCURRENCES per pokemon (not row count),
    so repeated move rows do not inflate denominator.
    """
    rows: List[dict] = []
    presence_rows: List[dict] = []

    for idx, team in enumerate(teams):
        fitness = safe_float(team.get("fitness", 0.0))
        signature = team_signature(team)
        team_id = f"{idx}:{signature}"

        for p in team.get("pokemon", []):
            name = str(p.get("name", "")).strip().lower()
            if not name:
                continue

            presence_rows.append({
                "team_id": team_id,
                "pokemon": name,
            })

            for mv in canonical_movesets.get(name, []):
                if mv == "unknown":
                    continue
                rows.append({
                    "team_id": team_id,
                    "pokemon": name,
                    "move": mv,
                    "move_class": MOVE_CLASSES.get(mv, "unknown"),
                    "archetype": pokemon_archetype.get(name, "unknown"),
                    "fitness_weight": fitness,
                })

    if not rows:
        return pd.DataFrame(
            columns=[
                "pokemon", "archetype", "move", "move_class",
                "weighted_frequency", "raw_count", "unique_team_occurrences",
                "normalized_weighted_frequency",
            ]
        )

    df = pd.DataFrame(rows)
    presence_df = pd.DataFrame(presence_rows).drop_duplicates()
    pokemon_team_count = (
        presence_df.groupby("pokemon", as_index=False)
        .agg(unique_team_occurrences=("team_id", "nunique"))
    )

    out = (
        df.groupby(["pokemon", "archetype", "move", "move_class"], as_index=False)
        .agg(
            weighted_frequency=("fitness_weight", "sum"),
            raw_count=("move", "count"),
        )
        .merge(pokemon_team_count, on="pokemon", how="left")
    )

    out["normalized_weighted_frequency"] = (
        out["weighted_frequency"] / out["unique_team_occurrences"].clip(lower=1)
    )

    out = out.sort_values(["pokemon", "normalized_weighted_frequency"], ascending=[True, False])
    return out


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build role-defining move priors from GA teams")
    parser.add_argument(
        "--teams-json",
        type=Path,
        default=Path("reports/ga_results/run_601pokemon_20260305_171543/top_10_teams.json"),
        help="Path to fallback GA top teams JSON (used if no globs are provided)",
    )
    parser.add_argument(
        "--teams-glob",
        type=str,
        default="reports/ga_results/run_601pokemon_*/top_10_teams.json",
        help="Glob for loading run-level team JSON files",
    )
    parser.add_argument(
        "--generation-teams-glob",
        type=str,
        default="",
        help="Optional glob for per-generation elite team snapshots (same team JSON schema)",
    )
    parser.add_argument(
        "--cluster-csv",
        type=Path,
        default=Path("reports/clustering_analysis/data/pokemon_with_clusters.csv"),
        help="Path to pokemon_with_clusters.csv",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=500,
        help="How many ranked teams to analyze (use <=0 for all loaded teams)",
    )
    parser.add_argument(
        "--dedupe-mode",
        type=str,
        choices=["composition", "none"],
        default="composition",
        help="Team dedupe strategy before weighting",
    )
    parser.add_argument(
        "--include-setup-whitelist",
        action="store_true",
        help="Allow setup moves only via role whitelist",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("reports/ga_results/run_601pokemon_20260305_171543/phase3_role_bootstrap"),
        help="Output directory for role priors artifacts",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    args.output_dir.mkdir(parents=True, exist_ok=True)

    teams = load_ga_teams(
        args.teams_json,
        args.top_n,
        teams_glob=args.teams_glob,
        generation_teams_glob=args.generation_teams_glob,
    )
    if args.dedupe_mode == "composition":
        deduped_teams, duplicate_count = deduplicate_teams(teams)
    else:
        deduped_teams, duplicate_count = teams, 0
    cluster_df = load_cluster_data(args.cluster_csv)

    # Build lookup by pokemon name
    cluster_lookup = {str(r["name"]).lower(): r for _, r in cluster_df.iterrows()}

    # Collect unique pokemon in analyzed teams
    seen_pokemon: Set[str] = set()
    for team in teams:
        for p in team.get("pokemon", []):
            name = str(p.get("name", "")).strip().lower()
            if name:
                seen_pokemon.add(name)

    role_priors_rows: List[dict] = []
    canonical_movesets: Dict[str, List[str]] = {}

    pokemon_archetype: Dict[str, str] = {}

    for name in sorted(seen_pokemon):
        row = cluster_lookup.get(name)
        if row is None:
            role_prior = "mixed_balanced"
            allowed_moves: Set[str] = set()
            canonical = []
            fetch_status = "missing_in_cluster_csv"
        else:
            role_prior = role_prior_from_row(row)
            pokemon_archetype[name] = str(row.get("archetype", "unknown"))
            try:
                learnable = fetch_learnable_moves(name)
                allowed_moves = allowed_moves_for_role(
                    pokemon_name=name,
                    learnable_moves=learnable,
                    role_prior=role_prior,
                    include_setup=args.include_setup_whitelist,
                )
                canonical = build_canonical_moveset(role_prior, allowed_moves, k=4)
                fetch_status = "ok"
            except Exception as ex:
                allowed_moves = set()
                canonical = []
                fetch_status = f"fetch_error: {type(ex).__name__}"

        canonical_movesets[name] = canonical
        role_priors_rows.append(
            {
                "name": name,
                "role_prior": role_prior,
                "archetype": None if row is None else row.get("archetype"),
                "allowed_move_count": len(allowed_moves),
                "canonical_move_count": len(canonical),
                "canonical_moves": ";".join(canonical),
                "fetch_status": fetch_status,
            }
        )

    # Weighted move frequency over teams
    weighted_df = aggregate_weighted_frequency(deduped_teams, canonical_movesets, pokemon_archetype)

    # Persist artifacts
    priors_df = pd.DataFrame(role_priors_rows).sort_values(["role_prior", "name"])
    priors_df.to_csv(args.output_dir / "pokemon_role_priors.csv", index=False)
    weighted_df.to_csv(args.output_dir / "weighted_move_frequency.csv", index=False)

    # Role distribution sanity output
    role_stats = (
        priors_df.groupby(["role_prior", "archetype"], dropna=False)
        .size()
        .rename("pokemon_count")
        .reset_index()
        .sort_values(["pokemon_count", "role_prior"], ascending=[False, True])
    )
    role_stats.to_csv(args.output_dir / "role_prior_stats.csv", index=False)

    # Quick sanity: top-3 moves per pokemon by normalized weighted frequency
    if len(weighted_df):
        top3 = (
            weighted_df.sort_values(["pokemon", "normalized_weighted_frequency"], ascending=[True, False])
            .groupby("pokemon", as_index=False)
            .head(3)
            .loc[:, [
                "pokemon", "move", "move_class", "normalized_weighted_frequency",
                "weighted_frequency", "raw_count", "unique_team_occurrences",
            ]]
        )
    else:
        top3 = pd.DataFrame(columns=[
            "pokemon", "move", "move_class", "normalized_weighted_frequency",
            "weighted_frequency", "raw_count", "unique_team_occurrences",
        ])
    top3.to_csv(args.output_dir / "top3_moves_per_pokemon.csv", index=False)

    with (args.output_dir / "canonical_movesets.json").open("w", encoding="utf-8") as f:
        json.dump(canonical_movesets, f, indent=2)

    # Summary
    ok_count = int((priors_df["fetch_status"] == "ok").sum()) if len(priors_df) else 0
    print("=" * 80)
    print("PHASE 3 BOOTSTRAP: ROLE-DEFINING MOVE PRIORS")
    print("=" * 80)
    print(f"Teams analyzed: {len(teams)}")
    print(f"Deduplicated teams: {len(deduped_teams)} (removed duplicates: {duplicate_count})")
    print(f"Dedupe mode: {args.dedupe_mode}")
    if args.teams_glob:
        print(f"Teams glob: {args.teams_glob}")
    if args.generation_teams_glob:
        print(f"Generation teams glob: {args.generation_teams_glob}")
    print(f"Unique Pokemon analyzed: {len(seen_pokemon)}")
    print(f"Move fetch success: {ok_count}/{len(priors_df)}")
    print(f"Setup whitelist enabled: {args.include_setup_whitelist}")
    
    # Print relative path from project root
    proj_root = Path(__file__).resolve().parents[1]
    try:
        rel_path = args.output_dir.resolve().relative_to(proj_root)
        print(f"Output directory: {rel_path}")
    except ValueError:
        print(f"Output directory: {args.output_dir}")

    if args.top_n > 0 and len(teams) < args.top_n:
        print(f"Note: Requested top_n={args.top_n}, but only found {len(teams)} teams in JSON.")

    print("Top role priors:")
    if len(priors_df):
        print(priors_df["role_prior"].value_counts().head(8).to_string())

    print("=" * 80)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
