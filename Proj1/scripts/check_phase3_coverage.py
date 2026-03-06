"""Check which Pokemon from GAare missing from Phase 3 role priors."""

import pandas as pd

# Load the diversity analysis results
ga_pokemon = [
    "arctozolt", "aurorus", "blissey", "camerupt", "darmanitan-standard",
    "darmanitan-zen", "diggersby", "dracozolt", "eelektross", "florges",
    "hydrapple", "kingambit", "latias", "manaphy", "marshadow",
    "meowstic-male", "miraidon", "oinkologne-female", "raging-bolt",
    "shuckle", "sneasler", "stakataka", "stoutland",
    "urshifu-single-strike", "urshifu-single-strike-gmax",
    "victreebel", "xurkitree", "zekrom", "zeraora"
]

# Load Phase 3 results
phase3_df = pd.read_csv("reports/ga_results/run_601pokemon_20260305_133528/phase3_role_bootstrap_v1/pokemon_role_priors.csv")
phase3_pokemon = set(phase3_df["name"].str.lower())

print(f"\n{'='*80}")
print("PHASE 3 COVERAGE ANALYSIS")
print(f"{'='*80}")
print(f"GA discovered: {len(ga_pokemon)} unique Pokemon")
print(f"Phase 3 assigned roles: {len(phase3_pokemon)} Pokemon")
print(f"Missing: {len(ga_pokemon) - len(phase3_pokemon)} Pokemon")

missing = sorted(set(ga_pokemon) - phase3_pokemon)
print(f"\n{'='*80}")
print(f"MISSING POKEMON ({len(missing)})")
print(f"{'='*80}")
for i, pokemon in enumerate(missing, 1):
    print(f"  {i:2d}. {pokemon}")

# Check if some are just name mismatches
print(f"\n{'='*80}")
print("POKEMON WITH ROLES")
print(f"{'='*80}")
for pokemon in sorted(phase3_pokemon):
    role = phase3_df[phase3_df["name"].str.lower() == pokemon]["role_prior"].values[0]
    archetype = phase3_df[phase3_df["name"].str.lower() == pokemon]["archetype"].values[0]
    print(f"  {pokemon:30s} - {role:20s} ({archetype})")
