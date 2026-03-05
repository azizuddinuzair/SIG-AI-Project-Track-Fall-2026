"""
PHASE 1.2: Multi-Seed Validation Test

Tests whether ConfigC's configuration produces consistent results across multiple runs.

Runs GA 5 times with different random seeds (42, 123, 456, 789, 999)
Expected results:
- If all runs converge to similar fitness (std < 2%): Solution is robust
- If results vary widely (std > 5%): Solution is luck-dependent
- If one seed is lucky: Need more seeds to validate

This test determines whether ConfigC is genuinely superior or if we got lucky.
"""

import sys
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

import argparse
import os
from pathlib import Path
import numpy as np
import time
import json
import random
from concurrent.futures import ProcessPoolExecutor, as_completed

# Add project paths
current_dir = Path(__file__).resolve().parent
proj_dir = current_dir.parents[4]  # Navigate to Proj1/
sys.path.insert(0, str(proj_dir / "src" / "models"))

from ga_optimization import PokemonGA, load_pokemon_data
from ga_config import get_config_c

def _run_single_seed(seed, generations, fitness_threshold=0.70):
    np.random.seed(seed)
    random.seed(seed)

    pokemon_df = load_pokemon_data()
    config = get_config_c()
    config["population"]["generations"] = generations

    start_time = time.time()

    ga = PokemonGA(pokemon_df, config)
    ga.random_state = seed
    history = ga.run()

    elapsed = time.time() - start_time

    best_team_df, best_fitness, _ = ga.get_best_teams(1)[0]
    best_team = ",".join(best_team_df['name'].tolist())
    best_series = history['max_fitness'].to_numpy()
    crossed = np.where(best_series > fitness_threshold)[0]
    convergence_gen = int(crossed[0]) if len(crossed) > 0 else len(best_series)

    return {
        "seed": seed,
        "best_fitness": float(best_fitness),
        "best_team": best_team,
        "convergence_generation": convergence_gen,
        "elapsed_seconds": elapsed,
        "final_population_std": float(np.std(best_series[-10:])) if len(best_series) >= 10 else float(np.std(best_series))
    }


def multi_seed_test(
    seeds=None,
    generations=250,
    output_file=None,
    parallel=True,
    max_workers=None
):
    """
    Run GA multiple times with different seeds.
    
    Args:
        seeds: List of random seeds to test
        generations: Number of generations per run
        output_file: Optional file to save results
        
    Returns:
        Dictionary with multi-seed results
    """
    if seeds is None:
        seeds = [42, 123, 456, 789, 999]

    print("\n" + "="*80)
    print("PHASE 1.2: Multi-Seed Validation Test")
    print("="*80)
    print(f"Running ConfigC with {len(seeds)} different seeds...")
    print(f"Seeds: {seeds}\n")

    if max_workers is None:
        cpu_count = os.cpu_count() or 1
        max_workers = min(len(seeds), max(1, cpu_count - 1))
    max_workers = max(1, min(max_workers, len(seeds)))

    print(f"Generations per seed: {generations}")
    print(f"Execution mode: {'parallel' if parallel else 'sequential'}")
    if parallel:
        print(f"Workers: {max_workers}")
    print()
    
    run_results = []
    all_best_fitnesses = []
    if parallel and len(seeds) > 1:
        with ProcessPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(_run_single_seed, seed, generations): seed
                for seed in seeds
            }

            for future in as_completed(futures):
                seed = futures[future]
                print(f"\n[SEED {seed}]")
                print("-" * 40)
                try:
                    run_data = future.result()
                    run_results.append(run_data)
                    all_best_fitnesses.append(run_data["best_fitness"])
                    print(f"  Best fitness:     {run_data['best_fitness']:.6f}")
                    print(f"  Best team:        {run_data['best_team']}")
                    print(f"  Convergence gen:  {run_data['convergence_generation']}")
                    print(f"  Time:             {run_data['elapsed_seconds']:.1f}s")
                except Exception as e:
                    print(f"  [ERROR] {e}")
                    run_results.append({
                        "seed": seed,
                        "best_fitness": None,
                        "error": str(e)
                    })
    else:
        for seed in seeds:
            print(f"\n[SEED {seed}]")
            print("-" * 40)
            try:
                run_data = _run_single_seed(seed, generations)
                run_results.append(run_data)
                all_best_fitnesses.append(run_data["best_fitness"])
                print(f"  Best fitness:     {run_data['best_fitness']:.6f}")
                print(f"  Best team:        {run_data['best_team']}")
                print(f"  Convergence gen:  {run_data['convergence_generation']}")
                print(f"  Time:             {run_data['elapsed_seconds']:.1f}s")
            except Exception as e:
                print(f"  [ERROR] {e}")
                run_results.append({
                    "seed": seed,
                    "best_fitness": None,
                    "error": str(e)
                })

    run_results.sort(key=lambda item: item.get("seed", 0))
    
    # Analysis
    valid_fitnesses = [f for f in all_best_fitnesses if f is not None]
    
    if len(valid_fitnesses) > 0:
        mean_fitness = np.mean(valid_fitnesses)
        std_fitness = np.std(valid_fitnesses)
        std_percent = (std_fitness / mean_fitness * 100) if mean_fitness > 0 else 0
        max_fitness = np.max(valid_fitnesses)
        min_fitness = np.min(valid_fitnesses)
    else:
        mean_fitness = std_fitness = std_percent = max_fitness = min_fitness = None
    
    results = {
        "test": "Multi-Seed Validation",
        "num_seeds": len(seeds),
        "seeds": seeds,
        "successful_runs": len(valid_fitnesses),
        "failed_runs": len(run_results) - len(valid_fitnesses),
        "run_details": run_results,
        "mean_fitness": float(mean_fitness) if mean_fitness else None,
        "max_fitness": float(max_fitness) if max_fitness else None,
        "min_fitness": float(min_fitness) if min_fitness else None,
        "std_fitness": float(std_fitness) if std_fitness else None,
        "std_percent": float(std_percent) if std_percent else None,
        "verdict": "",
        "interpretation": ""
    }
    
    # Verdict
    if std_percent is None or len(valid_fitnesses) < 3:
        results["verdict"] = "[INCONCLUSIVE] Too many failures or seeds"
        results["interpretation"] = f"Only {len(valid_fitnesses)} successful runs. Need more data."
    elif std_percent < 1:
        results["verdict"] = "[PASS] Highly reproducible (std < 1%)"
        results["interpretation"] = "Solution is robust and consistent across seeds."
    elif std_percent < 2:
        results["verdict"] = "[PASS] Very reproducible (std < 2%)"
        results["interpretation"] = "Solution is stable. GA is working as designed."
    elif std_percent < 5:
        results["verdict"] = "[PASS] Reproducible (std < 5%)"
        results["interpretation"] = "Some variance but generally consistent."
    else:
        results["verdict"] = "[WARNING] High variance (std > 5%)"
        results["interpretation"] = "Results vary significantly. Consider luck-dependent or settings-sensitive."
    
    # Print summary
    print("\n" + "-" * 80)
    print(f"Summary across {len(valid_fitnesses)} successful runs:")
    if len(valid_fitnesses) > 0:
        print(f"  Mean fitness:    {mean_fitness:.6f}")
        print(f"  Max fitness:     {max_fitness:.6f}")
        print(f"  Min fitness:     {min_fitness:.6f}")
        print(f"  Std deviation:   {std_fitness:.6f} ({std_percent:.2f}%)")
    else:
        print("  No successful runs to summarize.")
    print()
    print(f"[SUMMARY] {results['verdict']}")
    print(f"   {results['interpretation']}")
    print("-" * 80)
    
    # Save results
    if output_file is None:
        output_file = Path(__file__).parent.parent / "results" / "02_multiseed_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}\n")
    
    return results


if __name__ == "__main__":
    try:
        parser = argparse.ArgumentParser(description="Run Phase 1.2 multi-seed GA validation.")
        parser.add_argument("--seeds", type=str, default="42,123,456,789,999", help="Comma-separated seeds")
        parser.add_argument("--generations", type=int, default=250, help="GA generations per seed")
        parser.add_argument("--workers", type=int, default=None, help="Parallel worker count")
        parser.add_argument("--sequential", action="store_true", help="Force sequential execution")
        parser.add_argument("--output", type=str, default=None, help="Output JSON file path")
        args = parser.parse_args()

        seeds = [int(seed.strip()) for seed in args.seeds.split(",") if seed.strip()]
        output_path = Path(args.output) if args.output else None

        results = multi_seed_test(
            seeds=seeds,
            generations=args.generations,
            output_file=output_path,
            parallel=not args.sequential,
            max_workers=args.workers
        )
        print("[DONE] Multi-seed test completed successfully\n")
    except Exception as e:
        print(f"\n[FAIL] Test failed with error:\n{e}\n")
        import traceback
        traceback.print_exc()
