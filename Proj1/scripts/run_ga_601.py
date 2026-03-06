"""
Phase 2 GA Optimization: Run with 601 Pokemon

Configuration for runtime:
- Population: 150
- Generations: 100
- Uses Config C (optimal from previous ablation study)

Saves results to: Proj1/reports/ga_results/run_601pokemon_<timestamp>/
"""

import sys
from pathlib import Path

# Add src to path for imports
sys.path.append(str(Path(__file__).parents[1]))

import pandas as pd
import numpy as np
import joblib
from datetime import datetime
import json

from src.ga import PokemonGA, load_pokemon_data, get_config_c


def main():
    print("=" * 80)
    print("PHASE 2: GA TEAM OPTIMIZATION (1 HOUR RUN)")
    print("Dataset: 601 Pokemon (including new forms)")
    print("=" * 80)
    
    # Load updated 601-Pokemon dataset
    print("\n[DATA] Loading data...")
    pokemon_df = load_pokemon_data()
    print(f"   ✓ Loaded {len(pokemon_df)} Pokemon")
    print(f"   ✓ Archetypes: {pokemon_df['archetype'].nunique()}")
    
    archetype_counts = pokemon_df['archetype'].value_counts()
    print("\n   Archetype distribution:")
    for arch, count in archetype_counts.items():
        print(f"     - {arch:25s}: {count:3d} ({count/len(pokemon_df)*100:5.1f}%)")
    
    # Configure for expanded diversity run (3-4 hours)
    print("\n[CONFIG] Configuring GA...")
    config = get_config_c()
    config['population']['size'] = 300
    config['population']['generations'] = 300
    config['name'] = "Config C - Extended (50+ Pokemon Diversity)"
    
    print(f"   Population size: {config['population']['size']}")
    print(f"   Generations: {config['population']['generations']}")
    print(f"   Estimated runtime: ~180-240 minutes (to maximize Pok\u00e9mon diversity)")
    
    # Create output directory before GA (for generation snapshots)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parents[1] / "reports" / "ga_results" / f"run_601pokemon_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize and run
    print("\n[GA] Starting GA evolution...")
    start_time = datetime.now()
    
    ga = PokemonGA(pokemon_df, config, output_dir=output_dir)
    history = ga.run()
    
    end_time = datetime.now()
    runtime = (end_time - start_time).total_seconds()
    
    print(f"\n✅ Evolution complete! Runtime: {runtime/60:.1f} minutes")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save top teams
    print(f"\n💾 Saving results to: {output_dir.name}/")
    
    top_10 = ga.get_best_teams(10)
    teams_data = []
    
    for rank, (team, fitness, breakdown) in enumerate(top_10, 1):
        team_dict = {
            'rank': rank,
            'fitness': fitness,
            'breakdown': breakdown,
            'pokemon': team[['name', 'archetype', 'type1', 'type2', 'hp', 'attack', 
                           'defense', 'special-attack', 'special-defense', 'speed']].to_dict('records')
        }
        teams_data.append(team_dict)
    
    with open(output_dir / "top_10_teams.json", 'w') as f:
        json.dump(teams_data, f, indent=2)
    
    # Save fitness history
    history_df = pd.DataFrame(history)
    history_df.to_csv(output_dir / "fitness_history.csv", index=False)
    
    # Save metadata
    metadata = {
        'dataset_size': len(pokemon_df),
        'archetypes': pokemon_df['archetype'].nunique(),
        'config': config,
        'runtime_seconds': runtime,
        'timestamp': timestamp,
        'best_fitness': float(top_10[0][1]),
        'best_breakdown': top_10[0][2]
    }
    
    with open(output_dir / "metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Print summary
    print("\n" + "=" * 80)
    print("TOP 5 TEAMS")
    print("=" * 80)
    
    for rank, (team, fitness, breakdown) in enumerate(top_10[:5], 1):
        print(f"\n[RANK {rank}] Fitness = {fitness:.4f}")
        print(team[['name', 'archetype', 'type1']].to_string(index=False))
        
        # Show archetype diversity
        archetype_counts = team['archetype'].value_counts()
        entropy = breakdown.get('archetype_entropy', 0)
        print(f"   Archetype entropy: {entropy:.3f} (diversity: {np.exp(entropy)/6*100:.1f}%)")
        for arch, count in archetype_counts.items():
            print(f"     - {arch}: {count}")
    
    print("\n" + "=" * 80)
    print("FINAL STATISTICS")
    print("=" * 80)
    
    max_fitness = history_df['max_fitness'].iloc[-1]
    mean_fitness = history_df['mean_fitness'].iloc[-1]
    improvement = (max_fitness - history_df['max_fitness'].iloc[0]) / history_df['max_fitness'].iloc[0] * 100
    
    print(f"Best fitness:      {max_fitness:.4f}")
    print(f"Mean fitness:      {mean_fitness:.4f}")
    print(f"Improvement:       {improvement:.1f}%")
    print(f"Runtime:           {runtime/60:.1f} minutes")
    print(f"Results saved to:  {output_dir}")
    
    print("\n✅ Phase 2 GA optimization complete!")
    
    return 0


if __name__ == "__main__":
    exit(main())
