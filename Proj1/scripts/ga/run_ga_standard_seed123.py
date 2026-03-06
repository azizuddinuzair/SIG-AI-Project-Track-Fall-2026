"""
Ultra-simple: Just call ga.run() which we know works.
No custom loops, no overrides, just measure results.
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parents[2]))

import pandas as pd
import numpy as np
import json
from datetime import datetime

from src.ga import PokemonGA, load_pokemon_data, get_config_c


print("\n" + "="*80)
print("STANDARD GA RUN v3 (With Different Seed)")
print("="*80)

pokemon_df = load_pokemon_data()
print(f"✓ {len(pokemon_df)} Pokemon loaded")

config = get_config_c()
config['population']['size'] = 300
config['population']['generations'] = 300
config['random_seed'] = 123  # Different seed

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = Path(__file__).parents[2] / "reports" / "ga_results" / f"run_standard_seed123_{timestamp}"
output_dir.mkdir(parents=True, exist_ok=True)

print(f"\n[GA] Running standard GA with seed=123...")

ga = PokemonGA(pokemon_df, config, output_dir=output_dir)

start = datetime.now()
fitness_hist = ga.run()
elapsed = (datetime.now() - start).total_seconds() / 60

print(f"\n✅ Done! ({elapsed:.1f} min)")

# Save top teams
best = ga.get_best_teams(10)
teams_data = []
for rank, (team, fitness, breakdown) in enumerate(best, 1):
    teams_data.append({
        'rank': rank,
        'fitness': fitness,
        'pokemon': team[['name', 'archetype', 'type1']].to_dict('records')
    })

with open(output_dir / "top_10_teams.json", 'w') as f:
    json.dump(teams_data, f, indent=2)

# Stats
all_pokemon = []
for team in ga.population:
    all_pokemon.extend(team['name'].tolist())

unique_final = len(set(all_pokemon))

print(f"\n{'='*80}")
print(f"RESULTS: {unique_final}/601 unique Pokemon ({unique_final/601*100:.1f}%)")
print(f"Target: 50+ (achievement: {unique_final}/50 = {unique_final/50*100:.1f}%)")

output_dir_str = str(output_dir)
print(f"\nOutput: {output_dir_str}")
