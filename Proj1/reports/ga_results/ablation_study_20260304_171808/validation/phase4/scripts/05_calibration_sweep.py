# -*- coding: utf-8 -*-
"""
Phase 4: Targeted Calibration Sweep

Goal:
- Tune diversity_weight and weakness_lambda with seed-averaged metrics.
- Use single-source fitness path + consistency assertions.
- Report mean best fitness, base strength, diversity, and stability (CV).
"""

import sys
import io
import json
import time
import copy
import argparse
from pathlib import Path
from datetime import datetime
from typing import Dict, List

import numpy as np
import pandas as pd

# UTF-8 safe output on Windows
if sys.stdout.encoding != "utf-8":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
if sys.stderr.encoding != "utf-8":
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Import project modules
def _find_proj_root(start: Path) -> Path:
    """Walk upward until we find the Proj1 root (contains src/ga)."""
    for candidate in [start] + list(start.parents):
        if (candidate / "src" / "ga").exists():
            return candidate
    raise RuntimeError("Could not locate project root containing src/ga")


PROJ_ROOT = _find_proj_root(Path(__file__).resolve().parent)
sys.path.append(str(PROJ_ROOT))

try:
    # Current module layout
    from src.ga.optimization import PokemonGA, load_pokemon_data
    from src.ga.config import get_config_c
    from src.ga.fitness import evaluate_fitness
except ModuleNotFoundError:
    # Backward compatibility with older ablation package layout
    from src.models.ga_optimization import PokemonGA, load_pokemon_data
    from src.models.ga_config import get_config_c
    from src.models.ga_fitness import evaluate_fitness


FITNESS_CONSISTENCY_TOL = 1e-6


def build_config(base_config: Dict, diversity_weight: float, weakness_lambda: float, population_size: int, generations: int) -> Dict:
    config = copy.deepcopy(base_config)
    config["name"] = f"d{diversity_weight:.2f}_w{weakness_lambda:.2f}"
    config["population"]["size"] = population_size
    config["population"]["generations"] = generations

    # Candidate defaults from Phase 2/3 findings
    config["initialization"]["method"] = "inverse"
    config["mutation"]["weighted"] = True

    # Calibration knobs
    config["fitness"]["diversity_weight"] = float(diversity_weight)
    config["fitness"]["weakness_lambda"] = float(weakness_lambda)
    return config


def run_config_across_seeds(pokemon_df: pd.DataFrame, config_template: Dict, seeds: List[int]) -> Dict:
    seed_runs = []

    for seed in seeds:
        config = copy.deepcopy(config_template)
        config["random_seed"] = seed
        config["name"] = f"{config_template['name']}_seed{seed}"

        ga = PokemonGA(pokemon_df=pokemon_df, config=config)
        ga.run()

        best_team_df, best_fitness, breakdown = ga.get_best_teams(n=1)[0]

        # Integrity check: stored fitness must match recomputed fitness exactly
        recomputed_fitness, _ = evaluate_fitness(best_team_df, config)
        assert abs(float(best_fitness) - float(recomputed_fitness)) < FITNESS_CONSISTENCY_TOL, (
            f"Fitness consistency failed for {config['name']}: "
            f"stored={float(best_fitness):.12f}, recomputed={float(recomputed_fitness):.12f}, "
            f"delta={abs(float(best_fitness) - float(recomputed_fitness)):.12f}"
        )

        seed_runs.append(
            {
                "seed": seed,
                "best_fitness": float(best_fitness),
                "base_strength": float(breakdown["base_strength"]),
                "base_fitness": float(breakdown["base_fitness"]),
                "entropy_bonus": float(breakdown["entropy_bonus"]),
                "weakness_penalty": float(breakdown["weakness_penalty"]),
                "imbalance_penalty": float(breakdown["imbalance_penalty"]),
                "unique_archetypes": int(best_team_df["archetype"].nunique()),
                "best_team": best_team_df["name"].tolist(),
            }
        )

    best_scores = np.array([r["best_fitness"] for r in seed_runs], dtype=float)
    base_strengths = np.array([r["base_strength"] for r in seed_runs], dtype=float)
    unique_archs = np.array([r["unique_archetypes"] for r in seed_runs], dtype=float)

    mean_fitness = float(np.mean(best_scores))
    std_fitness = float(np.std(best_scores))
    cv_fitness = float(std_fitness / mean_fitness) if mean_fitness > 0 else float("inf")

    return {
        "config_name": config_template["name"],
        "diversity_weight": float(config_template["fitness"]["diversity_weight"]),
        "weakness_lambda": float(config_template["fitness"]["weakness_lambda"]),
        "population_size": int(config_template["population"]["size"]),
        "generations": int(config_template["population"]["generations"]),
        "n_seeds": len(seeds),
        "mean_best_fitness": mean_fitness,
        "std_best_fitness": std_fitness,
        "cv_best_fitness": cv_fitness,
        "mean_base_strength": float(np.mean(base_strengths)),
        "mean_unique_archetypes": float(np.mean(unique_archs)),
        "max_best_fitness": float(np.max(best_scores)),
        "min_best_fitness": float(np.min(best_scores)),
        "pass_target": bool(
            mean_fitness >= 0.74 and
            float(np.mean(base_strengths)) >= 0.70 and
            5.0 <= float(np.mean(unique_archs)) <= 6.0 and
            cv_fitness < 0.05
        ),
        "seed_runs": seed_runs,
    }


def main():
    parser = argparse.ArgumentParser(description="Phase 4 calibration sweep")
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=5, help="Number of seeds starting at 42")
    parser.add_argument("--output-dir", type=str, default=str(Path(__file__).parent.parent / "results"))
    args = parser.parse_args()

    diversity_grid = [0.15, 0.20, 0.25, 0.30]
    weakness_grid = [0.00, 0.01, 0.02, 0.03, 0.05]
    seeds = list(range(42, 42 + args.seeds))

    print("\n" + "=" * 72)
    print("PHASE 4: CALIBRATION SWEEP")
    print("=" * 72)
    print(f"Population={args.population}, Generations={args.generations}, Seeds={seeds}")
    print(f"Grid size={len(diversity_grid)} x {len(weakness_grid)} = {len(diversity_grid) * len(weakness_grid)} configs")

    start = time.time()

    print("\nLoading Pokémon data...", end=" ", flush=True)
    pokemon_df = load_pokemon_data()
    print(f"Loaded {len(pokemon_df)}")

    base_config = get_config_c()
    all_results = []

    total_configs = len(diversity_grid) * len(weakness_grid)
    idx = 0

    for d in diversity_grid:
        for w in weakness_grid:
            idx += 1
            config = build_config(
                base_config=base_config,
                diversity_weight=d,
                weakness_lambda=w,
                population_size=args.population,
                generations=args.generations,
            )

            print(f"\n[{idx}/{total_configs}] Testing diversity_weight={d:.2f}, weakness_lambda={w:.2f}...")
            cfg_start = time.time()

            result = run_config_across_seeds(pokemon_df, config, seeds)
            elapsed = time.time() - cfg_start
            result["elapsed_seconds"] = elapsed
            all_results.append(result)

            print(
                f"  mean_best={result['mean_best_fitness']:.4f}, "
                f"base_strength={result['mean_base_strength']:.4f}, "
                f"arch={result['mean_unique_archetypes']:.2f}, "
                f"cv={result['cv_best_fitness']:.4f}, "
                f"pass={result['pass_target']}"
            )

    summary_df = pd.DataFrame(
        [
            {
                "diversity_weight": r["diversity_weight"],
                "weakness_lambda": r["weakness_lambda"],
                "mean_best_fitness": r["mean_best_fitness"],
                "mean_base_strength": r["mean_base_strength"],
                "mean_unique_archetypes": r["mean_unique_archetypes"],
                "cv_best_fitness": r["cv_best_fitness"],
                "pass_target": r["pass_target"],
                "elapsed_seconds": r["elapsed_seconds"],
            }
            for r in all_results
        ]
    )

    summary_df = summary_df.sort_values(["mean_best_fitness", "cv_best_fitness"], ascending=[False, True]).reset_index(drop=True)
    summary_df.insert(0, "rank", np.arange(1, len(summary_df) + 1))

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    results_path = output_dir / "05_calibration_sweep_results.json"
    summary_path = output_dir / "05_calibration_sweep_summary.csv"

    payload = {
        "phase": "Phase 4 - Calibration Sweep",
        "date": datetime.now().isoformat(),
        "population": args.population,
        "generations": args.generations,
        "seeds": seeds,
        "diversity_grid": diversity_grid,
        "weakness_grid": weakness_grid,
        "results": all_results,
        "total_elapsed_seconds": time.time() - start,
    }

    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    summary_df.to_csv(summary_path, index=False)

    print("\n" + "=" * 72)
    print("PHASE 4 COMPLETE")
    print("=" * 72)
    print(f"Total elapsed: {payload['total_elapsed_seconds']:.1f}s ({payload['total_elapsed_seconds']/60:.1f} min)")
    print(f"Results JSON: {results_path}")
    print(f"Summary CSV:  {summary_path}")

    print("\nTop 5 configs:")
    print(summary_df.head(5).to_string(index=False))


if __name__ == "__main__":
    main()
