"""
PHASE 1.1: Random Baseline Test

Tests whether ConfigC's best team (fitness = 0.7324) outperforms random selection.

Generates 100,000 random 6-Pokémon teams and calculates their fitness.
If ConfigC is in the top 0.5% of random teams, GA is optimizing.
If ConfigC is in the top 10%, GA is slightly better than random.
If ConfigC is in top 50%, GA isn't optimizing well.

Expected result: ConfigC should be in top 0.1-0.5% (top 50-500 of 100k)
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
from pathlib import Path
import pandas as pd
import numpy as np
import time
import json

# Add project paths
current_dir = Path(__file__).resolve().parent
proj_dir = current_dir.parents[4]  # Navigate to Proj1/
sys.path.insert(0, str(proj_dir / "src" / "models"))

from ga_optimization import load_pokemon_data, PokemonGA
from ga_config import get_config_c
from ga_fitness import evaluate_fitness

def random_baseline_test(num_samples=100000, output_file=None):
    """
    Generate random teams and compare to ConfigC best team.
    
    Args:
        num_samples: Number of random teams to generate (default 100k)
        output_file: Optional file to save results
        
    Returns:
        Dictionary with test results
    """
    print("\n" + "="*80)
    print("PHASE 1.1: Random Baseline Test")
    print("="*80)
    print(f"Generating {num_samples:,} random teams...\n")
    
    # Load data
    pokemon_df = load_pokemon_data()
    config = get_config_c()
    
    # ConfigC best team (from ablation_summary.csv)
    configc_best_fitness = 0.7323711133754189
    configc_best_team = "baxcalibur,groudon,flutter-mane,stakataka,miraidon,mewtwo"
    
    print(f"ConfigC best fitness: {configc_best_fitness:.6f}")
    print(f"ConfigC best team: {configc_best_team}\n")
    
    # Generate random teams
    start_time = time.time()
    random_fitnesses = []
    
    pokemon_names = pokemon_df['name'].values
    
    for i in range(num_samples):
        if (i + 1) % 10000 == 0:
            print(f"  {i+1:,} / {num_samples:,} teams generated...")
        
        # Randomly select 6 unique Pokémon
        team = list(np.random.choice(pokemon_names, size=6, replace=False))
        team_str = ",".join(team)
        
        # Get DataFrame for this team
        team_df = pokemon_df[pokemon_df['name'].isin(team)].copy()
        
        # Evaluate fitness
        fitness, _ = evaluate_fitness(team_df, config)
        random_fitnesses.append(fitness)
    
    elapsed = time.time() - start_time
    random_fitnesses = np.array(random_fitnesses)
    
    print(f"\nGeneration complete in {elapsed:.2f} seconds\n")
    
    # Analysis
    percentile = (random_fitnesses < configc_best_fitness).sum() / len(random_fitnesses) * 100
    better_count = (random_fitnesses > configc_best_fitness).sum()
    
    results = {
        "test": "Random Baseline",
        "num_samples": num_samples,
        "elapsed_seconds": elapsed,
        "configc_best_fitness": configc_best_fitness,
        "configc_best_team": configc_best_team,
        "random_mean_fitness": float(np.mean(random_fitnesses)),
        "random_max_fitness": float(np.max(random_fitnesses)),
        "random_min_fitness": float(np.min(random_fitnesses)),
        "random_std_fitness": float(np.std(random_fitnesses)),
        "configc_percentile": percentile,
        "teams_better_than_configc": int(better_count),
        "verdict": "",
        "interpretation": ""
    }
    
    # Verdict
    if percentile <= 0.5:
        results["verdict"] = "[PASS] GA optimizes significantly (top 0.5%)"
        results["interpretation"] = f"ConfigC is better than {percentile:.3f}% of random teams. GA is optimizing."
    elif percentile <= 5:
        results["verdict"] = "[PASS] GA optimizes (top 5%)"
        results["interpretation"] = f"ConfigC is better than {percentile:.2f}% of random teams. GA is working."
    elif percentile <= 20:
        results["verdict"] = "[WARNING] GA optimizes but weakly (top 20%)"
        results["interpretation"] = f"ConfigC is better than {percentile:.1f}% of random teams. Weak optimization."
    else:
        results["verdict"] = "[FAIL] GA not optimizing (below top 20%)"
        results["interpretation"] = f"ConfigC worse than {100-percentile:.1f}% of random teams. Check fitness function."
    
    # Print results
    print("-" * 80)
    print(f"Random team statistics (n={num_samples:,}):")
    print(f"  Mean fitness:    {results['random_mean_fitness']:.6f}")
    print(f"  Max fitness:     {results['random_max_fitness']:.6f}")
    print(f"  Min fitness:     {results['random_min_fitness']:.6f}")
    print(f"  Std deviation:   {results['random_std_fitness']:.6f}")
    print()
    print(f"ConfigC comparison:")
    print(f"  ConfigC fitness: {configc_best_fitness:.6f}")
    print(f"  Better than:     {percentile:.3f}% of random teams")
    print(f"  Rank:            {results['teams_better_than_configc']:,} / {num_samples:,}")
    print()
    print(f"RESULT: {results['verdict']}")
    print(f"        {results['interpretation']}")
    print("-" * 80)
    
    # Save results
    if output_file is None:
        output_file = Path(__file__).parent.parent / "results" / "01_random_baseline_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}\n")
    
    return results


if __name__ == "__main__":
    try:
        results = random_baseline_test(num_samples=100000)
        print("\n✅ Test completed successfully\n")
    except Exception as e:
        print(f"\n❌ Test failed with error:\n{e}\n")
        import traceback
        traceback.print_exc()
