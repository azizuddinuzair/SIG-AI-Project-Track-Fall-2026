"""
User-facing CLI for clustering and GA workflows.

Commands:
- cluster: Run clustering pipeline and optional deliverable generation/validation
- ga: Run GA optimization with configurable parameters
- pipeline: Run cluster -> ga in one command
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from collections import Counter
from typing import Dict, List

import pandas as pd

# Ensure imports from Proj1/src work when running as `py scripts/cli.py ...`
PROJ_ROOT = Path(__file__).resolve().parents[1]
if str(PROJ_ROOT) not in sys.path:
    sys.path.append(str(PROJ_ROOT))

from src.ga import PokemonGA, get_config_c
from src.ga.config import get_config_a, get_config_b, get_config_random
from src.ga.optimization import load_pokemon_data
from src.ga.fitness import TYPE_NAMES, get_type_effectiveness


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


def _safe_input(prompt: str) -> str | None:
    try:
        return input(prompt)
    except EOFError:
        return None
    except KeyboardInterrupt:
        return None


def _run_python_script(script_path: Path, args: List[str] | None = None) -> int:
    cmd = [sys.executable, str(script_path)]
    if args:
        cmd.extend(args)

    result = subprocess.run(cmd, cwd=str(PROJ_ROOT), check=False)
    return int(result.returncode)


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
    speed = row[stat_cols["speed"]]
    attack = row[stat_cols["attack"]]
    sp_attack = row[stat_cols["sp_attack"]]
    defense = row[stat_cols["defense"]]
    sp_defense = row[stat_cols["sp_defense"]]
    hp = row[stat_cols["hp"]]

    if speed >= 100 and max(attack, sp_attack) >= 100:
        return "sweeper"
    if max(defense, sp_defense) >= 100 and hp >= 80:
        return "wall"
    if speed >= 90 and max(defense, sp_defense) >= 80:
        return "pivot"
    if attack >= 90 and sp_attack >= 90:
        return "mixed"
    return "balanced"


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
        "recommendations": recommendations,
        "overall_score": overall_score,
    }


def interactive_team_generator() -> None:
    _print_header("MENU 1: TEAM GENERATOR")
    print("Pick a Pokémon, we'll build a team around it.")
    print()

    pokemon_df = load_pokemon_data()

    # Mandatory: Get 1 anchor Pokémon
    while True:
        raw_input = _safe_input("Enter a Pokémon name (your team anchor): ")
        if raw_input is None:
            print("Input interrupted. Returning to menu.")
            return
        anchor = _resolve_pokemon_name(raw_input.strip(), pokemon_df)
        if not anchor:
            print("Pokémon not found. Try again.")
            continue
        break

    # Optional: GA tuning parameters
    print()
    generations = _input_int("How many GA generations?", 10, 300, 80)
    population = _input_int("Population size?", 20, 500, 150)
    seed = _input_int("Random seed?", 0, 999999, 42)
    save_outputs = _input_yes_no("Save run artifacts?", default_yes=False)

    print("\nOptimizing team around", anchor, "...")
    
    config = get_config_c()
    config["population"]["size"] = population
    config["population"]["generations"] = generations
    config["random_seed"] = seed
    config["name"] = "Menu_SingleAnchor"

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = None
    if save_outputs:
        output_dir = PROJ_ROOT / "reports" / "ga_results" / f"run_anchor_{anchor}_{timestamp}"
        output_dir.mkdir(parents=True, exist_ok=True)

    ga = PokemonGA(
        pokemon_df=pokemon_df,
        config=config,
        output_dir=output_dir,
        locked_pokemon=[anchor],
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
        t2 = f"/{pokemon['type2']}" if pokemon.get("type2") else ""
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
        t2 = f"/{pokemon['type2']}" if pokemon.get("type2") else ""
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
    print("RECOMMENDATIONS")
    print("-" * 80)
    for rec in report["recommendations"]:
        print(f"- {rec}")

    print(f"\nOverall team score: {report['overall_score']}/100")


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
        print("4. Exit")

        raw_choice = _safe_input("Enter choice [1-4]: ")
        if raw_choice is None:
            print("\nInput interrupted. Exiting.")
            return 0
        choice = raw_choice.strip()
        if choice == "1":
            interactive_team_generator()
            if not _pause_to_menu():
                print("\nInput interrupted. Exiting.")
                return 0
        elif choice == "2":
            interactive_team_analyzer()
            if not _pause_to_menu():
                print("\nInput interrupted. Exiting.")
                return 0
        elif choice == "3":
            interactive_random_generator()
            if not _pause_to_menu():
                print("\nInput interrupted. Exiting.")
                return 0
        elif choice == "4":
            print("Goodbye.")
            return 0
        else:
            print("Invalid choice. Please enter 1, 2, 3, or 4.")


def run_cluster(args: argparse.Namespace) -> int:
    _print_header("CLI: CLUSTER WORKFLOW")

    clustering_script = PROJ_ROOT / "src" / "analysis" / "clustering.py"
    deliverables_script = (
        PROJ_ROOT
        / "reports"
        / "clustering_analysis"
        / "scripts"
        / "01_generate_phase1_deliverables.py"
    )
    validation_script = PROJ_ROOT / "scripts" / "validation" / "validate_601_clustering.py"

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
