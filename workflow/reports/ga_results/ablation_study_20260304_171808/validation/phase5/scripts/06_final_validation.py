"""
Phase 5: Final Validation
==========================
Quick check with recommended config:
- Inverse initialization
- Entropy enabled
- Weakness penalty = 0

Goals:
- Verify best fitness
- Check team structure (type diversity, base strength)
- Confirm stability across 3 seeds
- Runtime target: 10-15 minutes max
"""

import json
import sys
import os
from pathlib import Path
import time
import copy
import argparse
import io
import numpy as np
import pandas as pd

# Set UTF-8 encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

# Add parent directory to path
proj_root = Path(__file__).parent.parent.parent
sys.path.append(str(proj_root))
sys.path.append(str(proj_root / "src" / "models"))

from src.models.ga_optimization import PokemonGA, load_pokemon_data
from src.models.ga_fitness import evaluate_fitness

# ============================================================================
# CONSTANTS
# ============================================================================

FITNESS_CONSISTENCY_TOL = 1e-6

# ============================================================================
# CONFIG BUILDER
# ============================================================================

def build_config(population_size=60, generations=60):
    """Build the recommended configuration based on Phase 2 findings."""
    return {
        "name": "FinalValidation",
        "population": {
            "size": population_size,
            "generations": generations,
            "tournament_k": 3,
            "elitism": 5
        },
        "fitness": {
            "base_stats_weight": 0.40,
            "type_coverage_weight": 0.30,
            "synergy_weight": 0.15,
            "diversity_weight": 0.15,
            "imbalance_lambda": 0.20,
            "weakness_lambda": 0.0,  # <-- KEY: Set to 0 (Phase 2 found this harmful)
        },
        "initialization": {
            "method": "inverse",  # <-- KEY: Inverse init is best (0.7508 vs 0.7178)
        },
        "mutation": {
            "rate": 0.15,
            "weighted": True  # Enable archetype-aware mutation
        },
        "crossover": {
            "rate": 0.80,
        },
        "random_seed": 42  # Will be overridden per run
    }

# ============================================================================
# VALIDATION RUNNER
# ============================================================================

def run_validation_with_seed(seed, pokemon_df, config):
    """Run GA once with given seed, return best team and fitness breakdown."""
    cfg = copy.deepcopy(config)
    cfg['random_seed'] = seed
    cfg['name'] = f"FinalValidation_seed_{seed}"
    
    ga = PokemonGA(pokemon_df=pokemon_df, config=cfg)
    fitness_history = ga.run()
    
    # Get best teams
    best_teams = ga.get_best_teams(n=1)
    if not best_teams:
        raise ValueError(f"GA returned no teams for seed {seed}")
    
    best_team, best_fitness, breakdown = best_teams[0]
    
    # =====================================================================
    # CONSISTENCY CHECK: Recompute fitness from scratch
    # =====================================================================
    recomputed_fitness, recomputed_breakdown = evaluate_fitness(best_team, cfg)
    
    assert abs(best_fitness - recomputed_fitness) < FITNESS_CONSISTENCY_TOL, \
        f"Fitness mismatch: stored={best_fitness}, recomputed={recomputed_fitness}"
    
    # Extract team structure
    types_in_team = best_team['type1'].unique().tolist()
    mean_offensive_index = float(best_team['offensive_index'].mean())
    mean_defensive_index = float(best_team['defensive_index'].mean())
    
    return {
        'seed': seed,
        'best_fitness': best_fitness,
        'fitness_breakdown': breakdown,
        'team_names': best_team['name'].tolist(),
        'types_in_team': types_in_team,
        'num_types': len(types_in_team),
        'base_strength_component': float(breakdown.get('base_strength', 0.0)),
        'mean_offensive_index': mean_offensive_index,
        'mean_defensive_index': mean_defensive_index,
        'consistency_check': 'PASS',
    }

# ============================================================================
# MAIN
# ============================================================================

def main():
    parser = argparse.ArgumentParser(description="Phase 5 final validation")
    parser.add_argument("--population", type=int, default=60)
    parser.add_argument("--generations", type=int, default=60)
    parser.add_argument("--seeds", type=int, default=2)
    args = parser.parse_args()

    print("=" * 80)
    print("PHASE 5: FINAL VALIDATION")
    print("=" * 80)
    print()
    
    # Load pokemon data
    pokemon_df = load_pokemon_data()
    print(f"Loaded {len(pokemon_df)} Pokémon")
    print()
    
    config = build_config(population_size=args.population, generations=args.generations)
    
    print("CONFIG:")
    print(f"  Initialization: {config['initialization']['method']}")
    print(f"  Weakness Lambda: {config['fitness']['weakness_lambda']}")
    print(f"  Diversity Weight: {config['fitness']['diversity_weight']}")
    print(f"  Population Size: {config['population']['size']}")
    print(f"  Generations: {config['population']['generations']}")
    print()
    
    # Run across seeds for quick stability check
    seeds = [42 + i for i in range(args.seeds)]
    results = []
    best_fitnesses = []
    
    start_time = time.time()
    
    for i, seed in enumerate(seeds):
        print(f"[{i+1}/{len(seeds)}] Running seed {seed}...")
        seed_start = time.time()
        
        result = run_validation_with_seed(seed, pokemon_df, config)
        results.append(result)
        best_fitnesses.append(result['best_fitness'])
        
        elapsed = time.time() - seed_start
        print(f"  ✓ Best Fitness: {result['best_fitness']:.4f}")
        print(f"    Types in team: {result['types_in_team']}")
        print(f"    Base strength component: {result['base_strength_component']:.4f}")
        print(f"    Runtime: {elapsed:.1f}s")
        print()
    
    total_elapsed = time.time() - start_time
    
    # =====================================================================
    # SUMMARY
    # =====================================================================
    
    print("=" * 80)
    print("FINAL VALIDATION SUMMARY")
    print("=" * 80)
    print()
    
    fitness_array = np.array(best_fitnesses)
    mean_fitness = float(fitness_array.mean())
    std_fitness = float(fitness_array.std())
    cv_fitness = float(std_fitness / mean_fitness * 100) if mean_fitness > 0 else 0
    
    print(f"Best Fitness ({len(seeds)} seeds):")
    print(f"  Mean: {mean_fitness:.4f}")
    print(f"  Std Dev: {std_fitness:.6f}")
    print(f"  Coefficient of Variation: {cv_fitness:.2f}%")
    print()
    
    mean_base_strength = np.mean([r['base_strength_component'] for r in results])
    mean_num_types = np.mean([r['num_types'] for r in results])
    
    print(f"Team Structure:")
    print(f"  Avg base strength component: {mean_base_strength:.4f}")
    print(f"  Avg unique types: {mean_num_types:.1f}")
    print()
    
    # Check if consistent
    consistency_ok = all(r['consistency_check'] == 'PASS' for r in results)
    print(f"Fitness Consistency Check: {'✓ PASS' if consistency_ok else '✗ FAIL'}")
    print()
    
    print(f"Total Runtime: {total_elapsed:.1f}s ({total_elapsed/60:.1f} min)")
    print()
    
    # =====================================================================
    # SAVE RESULTS
    # =====================================================================
    
    output_dir = Path(__file__).parent.parent / 'results'
    output_dir.mkdir(parents=True, exist_ok=True)
    output_file = output_dir / '06_final_validation_results.json'
    output_data = {
        'config': config,
        'seeds': seeds,
        'results_per_seed': results,
        'summary': {
            'mean_best_fitness': mean_fitness,
            'std_best_fitness': std_fitness,
            'cv_best_fitness': cv_fitness,
            'mean_base_strength': mean_base_strength,
            'mean_num_types': mean_num_types,
            'consistency_check': 'PASS' if consistency_ok else 'FAIL',
            'total_runtime_seconds': total_elapsed,
        }
    }
    
    with open(output_file, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"Results saved to: {output_file}")
    print()
    print("=" * 80)
    print("PHASE 5 COMPLETE")
    print("=" * 80)

if __name__ == '__main__':
    main()
