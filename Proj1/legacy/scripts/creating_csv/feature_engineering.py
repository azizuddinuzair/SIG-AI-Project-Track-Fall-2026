"""
Feature engineering for Pokemon archetype clustering.

Core features (4D):
- offensive_index: Attack + Special Attack
- defensive_index: (HP * 0.5) + Defense + Special Defense  
- speed_percentile: rank(speed) / N
- physical_special_bias: (attack - sp_attack) / (attack + sp_attack)

Type structure (18D):
- type_defense_{type}: Damage multiplier received from each attacking type

Total: 23D engineered feature space ready for StandardScaler → PCA → Clustering
"""

import pathlib
import pandas as pd
import numpy as np


data_folder = pathlib.Path(__file__).resolve().parents[2] / "data"
pokemon_stats_folder = data_folder / "test"
pokemon_stats_csv = pokemon_stats_folder / "fully_evolved_pokemon_stats.csv"
cluster_data_folder = data_folder / "cluster_data"


def build_type_effectiveness_chart():
    """
    Creates a 18x18 matrix where chart[attacking_type_idx][defending_type_idx] = damage multiplier.
    Values: 0.5 (resists), 1.0 (normal), 2.0 (weak to)
    """
    type_names = ['normal', 'fire', 'water', 'grass', 'electric', 'ice', 'fighting',
                  'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost',
                  'dragon', 'dark', 'steel', 'fairy']
    
    # Initialize 18x18 matrix (all 1.0 = normal damage)
    chart = [[1.0 for _ in range(18)] for _ in range(18)]
    
    # Resistances: defending_type resists attacking_type (0.5x damage taken)
    resistances = {
        'normal': [],
        'fire': ['fire', 'grass', 'ice', 'bug', 'steel', 'fairy'],
        'water': ['fire', 'water', 'ice', 'steel'],
        'grass': ['water', 'grass', 'ground', 'electric'],
        'electric': ['flying', 'steel', 'electric'],
        'ice': ['ice'],
        'fighting': ['rock', 'bug', 'dark'],
        'poison': ['fighting', 'poison', 'bug', 'grass'],
        'ground': ['poison', 'rock'],
        'flying': ['fighting', 'bug', 'grass'],
        'psychic': ['fighting', 'psychic'],
        'bug': ['fighting', 'grass', 'dark'],
        'rock': ['normal', 'flying', 'poison', 'fire'],
        'ghost': ['poison', 'bug'],
        'dragon': ['fire', 'water', 'grass', 'electric'],
        'dark': ['ghost', 'dark'],
        'steel': ['normal', 'flying', 'rock', 'bug', 'steel', 'grass', 'psychic', 'ice', 'dragon', 'fairy'],
        'fairy': ['fighting', 'bug', 'dark']
    }
    
    # Weaknesses: defending_type weak to attacking_type (2.0x damage taken)
    weaknesses = {
        'normal': ['fighting'],
        'fire': ['water', 'ground', 'rock'],
        'water': ['grass', 'electric'],
        'grass': ['fire', 'ice', 'poison', 'flying', 'bug'],
        'electric': ['ground'],
        'ice': ['fire', 'fighting', 'rock', 'steel'],
        'fighting': ['flying', 'psychic', 'fairy'],
        'poison': ['ground', 'psychic'],
        'ground': ['water', 'grass', 'ice'],
        'flying': ['electric', 'ice', 'rock'],
        'psychic': ['bug', 'ghost', 'dark'],
        'bug': ['fire', 'flying', 'rock'],
        'rock': ['water', 'grass', 'fighting', 'ground', 'steel'],
        'ghost': ['ghost', 'dark'],
        'dragon': ['ice', 'dragon', 'fairy'],
        'dark': ['fighting', 'bug', 'fairy'],
        'steel': ['fire', 'water', 'ground'],
        'fairy': ['poison', 'steel']
    }
    
    # Populate chart
    for i, attacking_type in enumerate(type_names):
        for j, defending_type in enumerate(type_names):
            if defending_type in resistances.get(attacking_type, []):
                chart[i][j] = 0.5
            elif defending_type in weaknesses.get(attacking_type, []):
                chart[i][j] = 2.0
    
    return type_names, chart


def calculate_defensive_type_vector(pokemon_row, type_chart):
    """
    For a Pokemon, calculate damage taken from each attacking type.
    Returns 18D vector where each dimension = damage multiplier from that type.
    
    For dual-typed Pokemon, damage multiplier = type1_multiplier × type2_multiplier
    """
    type_names, chart = type_chart
    
    pokemon_type1 = str(pokemon_row.get('type1', 'normal')).lower()
    pokemon_type2 = pokemon_row.get('type2')
    pokemon_type2 = str(pokemon_type2).lower() if pd.notna(pokemon_type2) else None
    
    defensive_vector = []
    
    for i, attacking_type in enumerate(type_names):
        attacking_idx = i
        type1_idx = type_names.index(pokemon_type1)
        type1_multiplier = chart[attacking_idx][type1_idx]
        
        if pokemon_type2 and pokemon_type2 != 'nan' and pokemon_type2 != 'none':
            type2_idx = type_names.index(pokemon_type2)
            type2_multiplier = chart[attacking_idx][type2_idx]
            combined_multiplier = type1_multiplier * type2_multiplier
        else:
            combined_multiplier = type1_multiplier
        
        defensive_vector.append(combined_multiplier)
    
    return defensive_vector


def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate complete archetype feature set:
    - 4 core stat features (magnitude + orientation)
    - 18 type defensive features (structure)
    
    Returns DataFrame with original columns + engineered features
    """
    result = df.copy()
    
    print("  Calculating core stat features...")
    
    # Calculate Offensive Index (Attack + Special Attack)
    result["offensive_index"] = result["attack"] + result["special-attack"]

    # Calculate Defensive Index (HP * 0.5) + Defense + Special Defense
    result["defensive_index"] = (result["hp"] * 0.5) + result["defense"] + result["special-defense"]

    # Calculate Speed Percentile: rank(speed) / N
    result["speed_percentile"] = result["speed"].rank(pct=True)

    # Calculate Physical Special Bias (Attack - Special Attack) / (Attack + Special Attack)
    total_offense = result["attack"] + result["special-attack"]
    result["physical_special_bias"] = (result["attack"] - result["special-attack"]) / total_offense
    result["physical_special_bias"] = result["physical_special_bias"].fillna(0)

    # Defensive asymmetry and role-shape features for better tank role separation
    result["physical_bulk"] = result["hp"] * result["defense"]
    result["special_bulk"] = result["hp"] * result["special-defense"]

    total_bulk = result["physical_bulk"] + result["special_bulk"]
    result["bulk_bias"] = (result["special_bulk"] - result["physical_bulk"]) / total_bulk
    result["bulk_bias"] = result["bulk_bias"].replace([np.inf, -np.inf], 0).fillna(0)

    result["offense_to_bulk_ratio"] = result["offensive_index"] / (result["defensive_index"] + 1e-6)
    result["speed_to_bulk_ratio"] = result["speed"] / (result["defensive_index"] + 1e-6)

    # Percentile forms reduce sensitivity to absolute stat scale outliers.
    result["special_bulk_percentile"] = result["special_bulk"].rank(pct=True)
    result["physical_bulk_percentile"] = result["physical_bulk"].rank(pct=True)
    
    print("  Building type effectiveness chart...")
    type_chart = build_type_effectiveness_chart()
    type_names, _ = type_chart
    
    print("  Calculating defensive type vectors...")
    defensive_vectors = []
    for idx, row in result.iterrows():
        vector = calculate_defensive_type_vector(row, type_chart)
        defensive_vectors.append(vector)
    
    # Add 18 type defense columns
    for i, type_name in enumerate(type_names):
        result[f"type_defense_{type_name}"] = [vec[i] for vec in defensive_vectors]
    
    return result


if __name__ == "__main__":
    """
    Generate engineered features for all Pokemon and save to cluster_data folder.
    """
    # Create output folder if it doesn't exist
    cluster_data_folder.mkdir(parents=True, exist_ok=True)
    
    # Load the full dataset
    print("Loading Pokemon data...")
    df = pd.read_csv(pokemon_stats_csv)
    print(f"  Loaded {len(df)} fully evolved Pokemon")
    
    # Calculate features for all Pokemon
    print("\nCalculating engineered features...")
    df_with_features = calculate_all_features(df)
    
    # Save to CSV
    output_path = cluster_data_folder / "pokemon_archetype_features.csv"
    df_with_features.to_csv(output_path, index=False)
    
    print(f"\n✅ Feature engineering complete!")
    # Print relative path from project root
    proj_root = pathlib.Path(__file__).resolve().parents[2]
    try:
        rel_path = output_path.relative_to(proj_root)
        print(f"💾 Saved to: {rel_path}\n")
    except ValueError:
        print(f"💾 Saved to: {output_path}\n")
    
    print(f"Dataset shape: {df_with_features.shape}")
    print(f"Rows: {len(df_with_features)} Pokemon")
    print(f"Columns: {len(df_with_features.columns)} (includes engineered features)\n")
    
    print("Features added:")
    print("  Core Stats (4D):")
    print("    - offensive_index")
    print("    - defensive_index")
    print("    - speed_percentile")
    print("    - physical_special_bias")
    print("  Type Defense (18D):")
    type_names, _ = build_type_effectiveness_chart()
    for type_name in type_names:
        print(f"    - type_defense_{type_name}")

    print("  Defensive Asymmetry / Role Shape:")
    print("    - physical_bulk")
    print("    - special_bulk")
    print("    - bulk_bias")
    print("    - offense_to_bulk_ratio")
    print("    - speed_to_bulk_ratio")
    print("    - special_bulk_percentile")
    print("    - physical_bulk_percentile")
    
    print(f"\n💡 Ready for: StandardScaler → PCA → GMM Clustering")
    print(f"💡 To test specific Pokemon, run: python test_pokemon_features.py")

