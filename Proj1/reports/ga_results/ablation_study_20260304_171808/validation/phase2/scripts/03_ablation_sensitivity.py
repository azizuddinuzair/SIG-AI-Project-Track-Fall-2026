"""
Phase 2: Ablation & Sensitivity Analysis (Runtime-Optimized)

Efficient ablation study using:
- Quick GA mode (100 pop × 100 gen) for fast signal detection
- Fixed seed (42) across all experiments to isolate variable effects
- Pokémon subsets (top 100 by stats) to reduce overhead
- Archetype-constrained swaps for neighbor sampling
- Parallel execution (joblib) for 4-8 core speedup
- Minimal output (best fitness/team/diversity only)

Target runtime: < 30 minutes for all tests
"""

import sys
import json
import numpy as np
import pandas as pd
import random
import time
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Dict, List, Tuple, Optional
import copy

# Add source paths
sys.path.insert(0, r"c:\Users\rezas\GitHub\SIG-AI-Project-Track-Fall-2026\Proj1\src\models")
sys.path.insert(0, r"c:\Users\rezas\GitHub\SIG-AI-Project-Track-Fall-2026\Proj1\src")

from ga_optimization import PokemonGA, load_pokemon_data
from ga_config import get_config_c, modify_config
from ga_fitness import evaluate_fitness


FITNESS_CONSISTENCY_TOL = 1e-6


def assert_fitness_consistency(stored_fitness: float, team_df: pd.DataFrame, config: Dict, context: str):
    """Fail fast if stored fitness drifts from recomputed fitness."""
    recomputed_fitness, _ = evaluate_fitness(team_df, config)
    assert abs(stored_fitness - recomputed_fitness) < FITNESS_CONSISTENCY_TOL, (
        f"Fitness consistency check failed ({context}): "
        f"stored_fitness={stored_fitness:.12f}, "
        f"recomputed_fitness={recomputed_fitness:.12f}, "
        f"delta={abs(stored_fitness - recomputed_fitness):.12f}"
    )


# ============================================================================
# QUICK GA MODE
# ============================================================================

def create_quick_config(base_config: Dict, **overrides) -> Dict:
    """
    Create a configuration for fast Phase 2 exploration.
    Uses reduced population and generation count to preserve signal while saving time.
    """
    config = copy.deepcopy(base_config)
    
    # Set quick GA params
    config["population"]["size"] = 100  # Down from 150
    config["population"]["generations"] = 100  # Down from 250
    
    # Keep selection pressure via tournament
    config["population"]["tournament_k"] = 3
    config["population"]["elitism"] = 3  # 3% elitism
    
    # Apply any custom overrides (e.g., diversity_weight=0.25)
    for key, value in overrides.items():
        if "." in key:
            parts = key.split(".")
            target = config
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        else:
            if key in config:
                config[key] = value
            else:
                # Search nested dicts
                for section in config.values():
                    if isinstance(section, dict) and key in section:
                        section[key] = value
                        break
    
    return config


# ============================================================================
# EXPERIMENTS
# ============================================================================

def find_convergence_generation(fitness_history: List[Dict], tolerance: float = 1e-6, window: int = 5) -> int:
    """
    Find generation where fitness converged (doesn't improve by > tolerance for window generations).
    
    Args:
        fitness_history: List of dicts with 'generation' and 'max_fitness' keys
        tolerance: Minimum improvement to count as progress
        window: Number of generations to check for stagnation
        
    Returns:
        Generation number where convergence occurred (or last generation)
    """
    if len(fitness_history) < window + 1:
        return len(fitness_history) - 1
    
    max_fitnesses = [h["max_fitness"] for h in fitness_history]
    
    for i in range(window, len(max_fitnesses)):
        # Check if last 'window' generations improved by < tolerance
        recent_max = max(max_fitnesses[i-window:i+1])
        if recent_max - max_fitnesses[i] < tolerance:
            return fitness_history[i]["generation"]
    
    return fitness_history[-1]["generation"]


def run_entropy_sweep(pokemon_df: pd.DataFrame, seed: int = 42) -> Dict:
    """
    Entropy sweep: Test effect of diversity_weight on convergence.
    
    Tests: [0.5, 0.3, 0.2, 0.15, 0.1] (baseline ConfigC uses 0.15)
    Runs: 5 experiments
    Time per exp: ~25 sec × 5 = ~2 min total
    """
    print("\n" + "="*70)
    print("ENTROPY SWEEP: Diversity Weight Sensitivity")
    print("="*70)
    
    base_config = get_config_c()
    weights_to_test = [0.5, 0.35, 0.25, 0.15, 0.10]
    
    results = []
    
    for w in weights_to_test:
        config = create_quick_config(base_config, **{"fitness.diversity_weight": w})
        config["name"] = f"EntropyWeight_{w}"
        config["random_seed"] = seed
        
        print(f"\n  Testing diversity_weight={w}...", end=" ", flush=True)
        start = time.time()
        
        ga = PokemonGA(pokemon_df, config)
        ga.run()
        
        best_team_obj, best_fitness, breakdown = ga.get_best_teams(1)[0]
        assert_fitness_consistency(
            stored_fitness=float(best_fitness),
            team_df=best_team_obj,
            config=config,
            context=f"entropy_sweep diversity_weight={w}"
        )
        convergence_gen = find_convergence_generation(ga.fitness_history)
        elapsed = time.time() - start
        
        # Extract team names
        best_team = ",".join(best_team_obj["name"].tolist())
        
        result = {
            "experiment": "entropy_sweep",
            "diversity_weight": w,
            "best_fitness": float(best_fitness),
            "best_team": best_team,
            "convergence_generation": convergence_gen,
            "elapsed_seconds": elapsed,
            "final_population_std": float(np.std([s[0] for s in ga.fitness_scores[-10:]]))
        }
        results.append(result)
        
        print(f"fitness={best_fitness:.4f}, conv_gen={convergence_gen}")
    
    return {"experiment": "entropy_sweep", "runs": results}


def run_ablation_tests(pokemon_df: pd.DataFrame, seed: int = 42) -> Dict:
    """
    Ablation study: Disable diversity components one at a time.
    
    Baseline: ConfigC (all enabled)
    Test 1: No diversity_weight (entropy)
    Test 2: No imbalance_lambda (balance penalty)
    Test 3: No weakness_lambda (weakness penalty)
    Test 4: Uniform init instead of sqrt_weighted
    
    Runs: 4 experiments
    Time per exp: ~25 sec × 4 = ~1.5 min total
    """
    print("\n" + "="*70)
    print("ABLATION TEST: Individual Component Impact")
    print("="*70)
    
    base_config = get_config_c()
    
    test_cases = [
        ("baseline_full", {}),
        ("no_entropy", {"fitness.diversity_weight": 0.0}),
        ("no_balance_penalty", {"fitness.imbalance_lambda": 0.0}),
        ("no_weakness_penalty", {"fitness.weakness_lambda": 0.0}),
        ("uniform_init_no_diversity", {
            "initialization.method": "uniform",
            "fitness.diversity_weight": 0.0,
            "fitness.imbalance_lambda": 0.0,
            "fitness.weakness_lambda": 0.0
        }),
    ]
    
    results = []
    
    for test_name, config_overrides in test_cases:
        config = create_quick_config(base_config, **config_overrides)
        config["name"] = test_name
        config["random_seed"] = seed
        
        print(f"\n  Testing {test_name}...", end=" ", flush=True)
        start = time.time()
        
        ga = PokemonGA(pokemon_df, config)
        ga.run()
        
        best_team_obj, best_fitness, breakdown = ga.get_best_teams(1)[0]
        assert_fitness_consistency(
            stored_fitness=float(best_fitness),
            team_df=best_team_obj,
            config=config,
            context=f"ablation_test {test_name}"
        )
        convergence_gen = find_convergence_generation(ga.fitness_history)
        elapsed = time.time() - start
        
        best_team = ",".join(best_team_obj["name"].tolist())
        
        result = {
            "experiment": "ablation_test",
            "test_name": test_name,
            "best_fitness": float(best_fitness),
            "best_team": best_team,
            "convergence_generation": convergence_gen,
            "elapsed_seconds": elapsed,
            "final_population_std": float(np.std([s[0] for s in ga.fitness_scores[-10:]]))
        }
        results.append(result)
        
        print(f"fitness={best_fitness:.4f}, conv_gen={convergence_gen}")
    
    return {"experiment": "ablation_tests", "runs": results}


def run_init_method_sensitivity(pokemon_df: pd.DataFrame, seed: int = 42) -> Dict:
    """
    Initialization method sensitivity: uniform vs. inverse vs. sqrt_weighted
    
    Tests impact of archetype weighting during initialization & mutation.
    Uses ConfigC fitness function (all diversity components).
    
    Runs: 3 experiments
    Time per exp: ~25 sec × 3 = ~1.5 min total
    """
    print("\n" + "="*70)
    print("INITIALIZATION SENSITIVITY: Weighted vs. Uniform Selection")
    print("="*70)
    
    base_config = get_config_c()
    
    test_cases = [
        ("uniform_init", {"initialization.method": "uniform", "mutation.weighted": False}),
        ("inverse_weighted_init", {"initialization.method": "inverse", "mutation.weighted": True}),
        ("sqrt_weighted_init", {"initialization.method": "sqrt_weighted", "mutation.weighted": True}),
    ]
    
    results = []
    
    for test_name, config_overrides in test_cases:
        config = create_quick_config(base_config, **config_overrides)
        config["name"] = test_name
        config["random_seed"] = seed
        
        print(f"\n  Testing {test_name}...", end=" ", flush=True)
        start = time.time()
        
        ga = PokemonGA(pokemon_df, config)
        ga.run()
        
        best_team_obj, best_fitness, breakdown = ga.get_best_teams(1)[0]
        assert_fitness_consistency(
            stored_fitness=float(best_fitness),
            team_df=best_team_obj,
            config=config,
            context=f"init_sensitivity {test_name}"
        )
        convergence_gen = find_convergence_generation(ga.fitness_history)
        elapsed = time.time() - start
        
        best_team = ",".join(best_team_obj["name"].tolist())
        
        result = {
            "experiment": "init_sensitivity",
            "init_method": config_overrides.get("initialization.method", "sqrt_weighted"),
            "best_fitness": float(best_fitness),
            "best_team": best_team,
            "convergence_generation": convergence_gen,
            "elapsed_seconds": elapsed,
            "final_population_std": float(np.std([s[0] for s in ga.fitness_scores[-10:]]))
        }
        results.append(result)
        
        print(f"fitness={best_fitness:.4f}, conv_gen={convergence_gen}")
    
    return {"experiment": "init_sensitivity", "runs": results}


def run_neighbor_sampling(pokemon_df: pd.DataFrame, best_team_names: List[str], seed: int = 42) -> Dict:
    """
    Neighbor sampling: Probe fitness landscape around best team.
    
    For each of 6 team members:
      - Sample 50 random Pokémon from same archetype
      - Compute fitness of team with that member swapped
      - Track max delta and spread to estimate landscape sharpness
    
    This is embarrassingly parallel but kept in main process for simplicity.
    
    Runs: 6 members × 50 swaps = 300 fitness evaluations
    Time: ~3-5 minutes total (fitness eval is fast)
    """
    print("\n" + "="*70)
    print("NEIGHBOR SAMPLING: Fitness Landscape Around Best Team")
    print("="*70)
    
    config = get_config_c()
    config["random_seed"] = seed
    
    # Get archetype mapping
    archetype_map = pokemon_df.set_index("name")["archetype"].to_dict()
    
    # For each position in best team
    position_results = []
    
    for pos, pokemon_name in enumerate(best_team_names):
        print(f"\n  Position {pos} ({pokemon_name}): Sampling neighbors...", end=" ", flush=True)
        
        if pokemon_name not in archetype_map:
            print(f" [SKIPPED - not found]")
            continue
        
        archetype = archetype_map[pokemon_name]
        
        # Get all Pokémon from same archetype
        same_archetype = pokemon_df[pokemon_df["archetype"] == archetype]["name"].tolist()
        
        if len(same_archetype) < 2:
            print(f" [SKIPPED - only {len(same_archetype)} in archetype]")
            continue
        
        # Sample random neighbors
        sample_size = min(50, len(same_archetype))
        neighbors = np.random.choice(same_archetype, size=sample_size, replace=False)
        
        # Evaluate fitness with each neighbor swap
        neighbor_fitnesses = []
        original_fitness = None
        
        for neighbor in neighbors:
            # Create variant team with this swap
            variant_team = [pokemon_df[pokemon_df["name"] == n].iloc[0] for n in best_team_names]
            variant_team[pos] = pokemon_df[pokemon_df["name"] == neighbor].iloc[0]
            variant_df = pd.DataFrame(variant_team)
            
            # Evaluate
            fitness, _ = evaluate_fitness(variant_df, config)
            neighbor_fitnesses.append(fitness)
            
            if neighbor == pokemon_name:
                original_fitness = fitness
        
        neighbor_fitnesses = np.array(neighbor_fitnesses)
        
        # Stats
        max_neighbor_fitness = np.max(neighbor_fitnesses)
        min_neighbor_fitness = np.min(neighbor_fitnesses)
        mean_neighbor_fitness = np.mean(neighbor_fitnesses)
        
        # Estimate landscape sharpness: delta from original
        if original_fitness is None:
            # Re-evaluate original
            original_team = pd.DataFrame([pokemon_df[pokemon_df["name"] == n].iloc[0] for n in best_team_names])
            original_fitness, _ = evaluate_fitness(original_team, config)
        
        max_delta = max_neighbor_fitness - original_fitness
        peak_sharpness = "high" if max_delta < 0.01 else "medium" if max_delta < 0.05 else "low"
        
        result = {
            "position": pos,
            "pokemon": pokemon_name,
            "archetype": archetype,
            "sample_size": sample_size,
            "original_fitness": float(original_fitness),
            "max_neighbor_fitness": float(max_neighbor_fitness),
            "min_neighbor_fitness": float(min_neighbor_fitness),
            "mean_neighbor_fitness": float(mean_neighbor_fitness),
            "max_delta": float(max_delta),
            "peak_sharpness": peak_sharpness
        }
        position_results.append(result)
        
        print(f"delta={max_delta:.4f}, sharpness={peak_sharpness}")
    
    return {"experiment": "neighbor_sampling", "positions": position_results}


# ============================================================================
# MAIN EXECUTION
# ============================================================================

def run_phase2_ablation_study(output_dir: Optional[str] = None):
    """
    Execute all Phase 2 ablation & sensitivity tests.
    
    Args:
        output_dir: Directory to save results JSON. Defaults to phase2/ in reports.
    """
    
    print("\n" + "="*70)
    print("PHASE 2: ABLATION & SENSITIVITY ANALYSIS (Optimized for Speed)")
    print("="*70)
    
    start_time = time.time()
    
    # Load data
    print("\nLoading Pokémon dataset...", end=" ", flush=True)
    pokemon_df = load_pokemon_data()
    print(f"Loaded {len(pokemon_df)} Pokémon")
    
    # Set output directory
    if output_dir is None:
        output_dir = Path(__file__).parent.parent / "results"
    else:
        output_dir = Path(output_dir)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # UTF-8 safety for multiprocessing
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    
    # Run all experiments
    all_results = {
        "phase": "Phase 2 - Ablation & Sensitivity",
        "strategy": "Runtime-optimized (100 pop × 100 gen, fixed seed=42)",
        "date": pd.Timestamp.now().isoformat(),
        "num_pokemon": len(pokemon_df),
        "experiments": {}
    }
    
    # Experiment 1: Entropy sweep
    entropy_results = run_entropy_sweep(pokemon_df, seed=42)
    all_results["experiments"]["entropy_sweep"] = entropy_results
    
    # Experiment 2: Ablation tests
    ablation_results = run_ablation_tests(pokemon_df, seed=42)
    all_results["experiments"]["ablation_tests"] = ablation_results
    
    # Experiment 3: Initialization sensitivity
    init_results = run_init_method_sensitivity(pokemon_df, seed=42)
    all_results["experiments"]["init_sensitivity"] = init_results
    
    # Experiment 4: Neighbor sampling (use best team from Phase 1)
    # Phase 1 best: baxcalibur,groudon,flutter-mane,stakataka,miraidon,mewtwo
    phase1_best_team = ["baxcalibur", "groudon", "flutter-mane", "stakataka", "miraidon", "mewtwo"]
    neighbor_results = run_neighbor_sampling(pokemon_df, phase1_best_team, seed=42)
    all_results["experiments"]["neighbor_sampling"] = neighbor_results
    
    # Save results
    elapsed_total = time.time() - start_time
    all_results["elapsed_seconds"] = elapsed_total
    
    result_file = output_dir / "03_ablation_sensitivity_results.json"
    
    with open(result_file, "w", encoding="utf-8") as f:
        json.dump(all_results, f, indent=2)
    
    # Summary
    print("\n" + "="*70)
    print("PHASE 2 COMPLETE")
    print("="*70)
    print(f"Total elapsed time: {elapsed_total:.1f} seconds ({elapsed_total/60:.1f} minutes)")
    print(f"Results saved to: {result_file}")
    print("\nNext: Analyze results to identify which components matter most.")
    
    return all_results


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Phase 2: Ablation & Sensitivity Analysis")
    parser.add_argument("--output", type=str, default=None, help="Output directory for results")
    args = parser.parse_args()
    
    run_phase2_ablation_study(output_dir=args.output)
