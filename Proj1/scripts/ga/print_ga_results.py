"""
Print summary of GA results from run_601pokemon_20260305_171543
"""

import pandas as pd
import json
from pathlib import Path

results_dir = Path("Proj1/reports/ga_results/run_601pokemon_20260305_171543")

print("=" * 80)
print("PHASE 2 GA RESULTS: 601 Pokemon Dataset")
print("=" * 80)

# Load history
history_df = pd.read_csv(results_dir / "fitness_history.csv")

print(f"\n📊 Evolution Progress:")
print(f"   Initial fitness (Gen 0):  {history_df['max_fitness'].iloc[0]:.4f}")
print(f"   Final fitness (Gen 100):  {history_df['max_fitness'].iloc[-1]:.4f}")
improvement = (history_df['max_fitness'].iloc[-1] - history_df['max_fitness'].iloc[0]) / history_df['max_fitness'].iloc[0] * 100
print(f"   Improvement:              +{improvement:.1f}%")

print(f"\n   Final mean fitness:       {history_df['mean_fitness'].iloc[-1]:.4f}")
print(f"   Final std:                {history_df['std_fitness'].iloc[-1]:.4f}")
print(f"   Final entropy:            {history_df['mean_entropy'].iloc[-1]:.3f}")
print(f"   Rare archetype %:         {history_df['rare_archetype_percent'].iloc[-1]:.1f}%")

# Load top teams
with open(results_dir / "top_10_teams.json") as f:
    teams = json.load(f)

print(f"\n🏆 BEST TEAM (Fitness = {teams[0]['fitness']:.4f})")
print("=" * 80)

# Calculate team BST
team_bst = 0
for poke in teams[0]['pokemon']:
    bst = poke['hp']+poke['attack']+poke['defense']+poke['special-attack']+poke['special-defense']+poke['speed']
    team_bst += bst
    type_str = poke['type1'] + (f"/{poke['type2']}" if poke['type2'] else "")
    print(f"  {poke['name']:25s} {poke['archetype']:25s} {type_str:20s} BST:{bst}")

print(f"\nTeam BST: {team_bst} (avg {team_bst/6:.1f}, cap: 3300)")
if team_bst > 3300:
    print(f"  ⚠️  OVER CAP by {team_bst - 3300} BST")
else:
    print(f"  ✓ Under cap by {3300 - team_bst} BST")

print(f"\nFitness Breakdown:")
for key, value in teams[0]['breakdown'].items():
    print(f"  {key:30s}: {value:.4f}")

# Count archetype distribution
archetype_counts = {}
for poke in teams[0]['pokemon']:
    arch = poke['archetype']
    archetype_counts[arch] = archetype_counts.get(arch, 0) + 1

print(f"\nArchetype Distribution:")
for arch, count in sorted(archetype_counts.items()):
    print(f"  {arch:25s}: {count}")

# Check if top teams are identical
print(f"\n📈 Team Diversity:")
unique_teams = set()
for team in teams[:10]:
    team_sig = tuple(sorted([p['name'] for p in team['pokemon']]))
    unique_teams.add(team_sig)

print(f"  Unique teams in top 10: {len(unique_teams)}")
if len(unique_teams) == 1:
    print("  ⚠️  WARNING: GA converged to single optimal team (all top 10 identical)")

# Load metadata
with open(results_dir / "metadata.json") as f:
    metadata = json.load(f)

print(f"\n⏱️  Runtime: {metadata['runtime_seconds']/60:.1f} minutes")

# Print relative path from project root if possible
proj_root = Path(__file__).resolve().parents[2]
try:
    rel_path = results_dir.resolve().relative_to(proj_root)
    print(f"📁 Results saved to: {rel_path}")
except ValueError:
    print(f"📁 Results saved to: {results_dir.name}")

print("\n" + "=" * 80)
print("✅ Phase 2 GA Complete!")
print("=" * 80)
