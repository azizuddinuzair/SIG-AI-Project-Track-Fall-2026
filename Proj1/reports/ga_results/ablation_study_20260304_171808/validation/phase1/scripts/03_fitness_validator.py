"""
PHASE 1.3: Fitness Validator Test

Tests whether the fitness function is correctly implemented.

Takes ConfigC's best team and recomputes its fitness from scratch.
If computed fitness matches the reported fitness: no bugs
If they differ: indicates a bug in the fitness calculation

This is a quick sanity check to ensure the GA results are trustworthy.
"""

import sys
import io
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

from pathlib import Path
import json
import pandas as pd

# Add project paths
current_dir = Path(__file__).resolve().parent
proj_dir = current_dir.parents[4]  # Navigate to Proj1/
sys.path.insert(0, str(proj_dir / "src" / "models"))

from ga_optimization import load_pokemon_data, PokemonGA
from ga_config import get_config_c
from ga_fitness import evaluate_fitness

def fitness_validator_test(output_file=None):
    """
    Validate fitness computation for ConfigC's best team.
    
    Args:
        output_file: Optional file to save results
        
    Returns:
        Dictionary with validation results
    """
    print("\n" + "="*80)
    print("PHASE 1.3: Fitness Validator Test")
    print("="*80)
    
    # Load data
    pokemon_df = load_pokemon_data()
    config = get_config_c()
    
    # ConfigC best team from ablation study
    reported_fitness = 0.7323711133754189
    best_team_str = "baxcalibur,groudon,flutter-mane,stakataka,miraidon,mewtwo"
    
    print(f"\nValidating ConfigC best team:")
    print(f"  Team: {best_team_str}")
    print(f"  Reported fitness: {reported_fitness:.10f}\n")
    
    # Parse team
    team_names = best_team_str.split(",")
    print(f"Team members:")
    for name in team_names:
        poke = pokemon_df[pokemon_df['name'] == name].iloc[0]
        print(f"  - {name:20s} (HP:{poke['hp']}, ATK:{poke['attack']}, SP.ATK:{poke['special-attack']})")
    
    print()
    
    # Recompute fitness
    team_df = pokemon_df[pokemon_df['name'].isin(team_names)].copy()
    computed_fitness, breakdown = evaluate_fitness(team_df, config)
    
    print(f"Fitness computation:")
    print(f"  Reported fitness: {reported_fitness:.10f}")
    print(f"  Computed fitness: {computed_fitness:.10f}")
    print(f"  Difference:       {abs(computed_fitness - reported_fitness):.10f}")
    print()
    
    # Show breakdown
    if breakdown:
        print("Fitness component breakdown:")
        for component, value in breakdown.items():
            print(f"  {component:30s}: {value:.6f}")
        print()
    
    # Verdict
    tolerance = 1e-6  # Allow for floating point errors
    if abs(computed_fitness - reported_fitness) < tolerance:
        verdict = "[PASS] Fitness computation is correct"
        interpretation = "Computed and reported values match (within floating point tolerance)"
        status = "VALID"
    elif abs(computed_fitness - reported_fitness) < 0.001:
        verdict = "[WARNING] Minor discrepancy detected"
        interpretation = f"Difference of {abs(computed_fitness - reported_fitness):.6f}. Check rounding."
        status = "MINOR_ISSUE"
    else:
        verdict = "[FAIL] Significant discrepancy"
        interpretation = f"Difference of {abs(computed_fitness - reported_fitness):.6f}. Bug likely in fitness function."
        status = "BUG_DETECTED"
    
    results = {
        "test": "Fitness Validator",
        "best_team": best_team_str,
        "reported_fitness": reported_fitness,
        "computed_fitness": float(computed_fitness),
        "difference": float(abs(computed_fitness - reported_fitness)),
        "verdict": verdict,
        "interpretation": interpretation,
        "status": status
    }
    
    print("-" * 80)
    print(f"RESULT: {verdict}")
    print(f"        {interpretation}")
    print("-" * 80)
    
    # Save results
    if output_file is None:
        output_file = Path(__file__).parent.parent / "results" / "03_fitness_validator_results.json"
    
    with open(output_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\nResults saved to: {output_file}\n")
    
    return results


if __name__ == "__main__":
    try:
        results = fitness_validator_test()
        print("✅ Fitness validator test completed\n")
    except Exception as e:
        print(f"\n❌ Test failed with error:\n{e}\n")
        import traceback
        traceback.print_exc()
