"""
Test script to inspect engineered features for specific Pokemon.
Useful for validating feature engineering logic and exploring archetypes.
"""

import pathlib
import pandas as pd
from Proj1.scripts.creating_csv.feature_engineering import calculate_all_features


def test_pokemon_features(pokemon_names, csv_path=None):
    """
    Load Pokemon data, calculate features, and display results for specified Pokemon.
    
    Args:
        pokemon_names: List of Pokemon names to test (lowercase)
        csv_path: Optional path to CSV file. Defaults to ../data/test/fully_evolved_pokemon_stats.csv
    """
    if csv_path is None:
        data_folder = pathlib.Path(__file__).resolve().parents[1] / "data"
        csv_path = data_folder / "test" / "fully_evolved_pokemon_stats.csv"
    
    # Load and process data
    df = pd.read_csv(csv_path)
    df_with_features = calculate_all_features(df)
    
    # Display header
    print("=" * 80)
    print(f"POKEMON FEATURE TEST - {len(pokemon_names)} Pokemon")
    print("=" * 80)
    
    found_count = 0
    
    for pokemon_name in pokemon_names:
        pokemon_data = df_with_features[df_with_features["name"] == pokemon_name.lower()]
        
        if pokemon_data.empty:
            print(f"\n⚠️  {pokemon_name.capitalize()} not found in dataset")
            continue
        
        found_count += 1
        pokemon = pokemon_data.iloc[0]
        
        print(f"\n{pokemon_name.upper()}")
        print("-" * 40)
        print(f"Type: {pokemon['type1']}", end="")
        if pd.notna(pokemon.get('type2')):
            print(f" / {pokemon['type2']}")
        else:
            print()
        
        print(f"\nBase Stats:")
        print(f"  HP:  {pokemon['hp']:3.0f}  |  Attack:     {pokemon['attack']:3.0f}")
        print(f"  Def: {pokemon['defense']:3.0f}  |  Sp. Attack: {pokemon['special-attack']:3.0f}")
        print(f"  SpD: {pokemon['special-defense']:3.0f}  |  Speed:      {pokemon['speed']:3.0f}")
        
        bst = (pokemon['hp'] + pokemon['attack'] + pokemon['defense'] + 
               pokemon['special-attack'] + pokemon['special-defense'] + pokemon['speed'])
        print(f"  Base Stat Total: {bst:.0f}")
        
        print(f"\nEngineered Features (Core Stats):")
        print(f"  Offensive Index:        {pokemon['offensive_index']:6.1f}")
        print(f"  Defensive Index:        {pokemon['defensive_index']:6.1f}")
        print(f"  Speed Percentile:       {pokemon['speed_percentile']:6.3f}")
        print(f"  Physical/Special Bias:  {pokemon['physical_special_bias']:6.3f}")
        
        # Show type defense profile (strengths and weaknesses)
        type_names = ['normal', 'fire', 'water', 'grass', 'electric', 'ice', 'fighting',
                      'poison', 'ground', 'flying', 'psychic', 'bug', 'rock', 'ghost',
                      'dragon', 'dark', 'steel', 'fairy']
        
        resists = []  # 0.5x damage
        weaks = []    # 2.0x damage
        
        for type_name in type_names:
            multiplier = pokemon[f'type_defense_{type_name}']
            if multiplier == 0.5:
                resists.append(type_name)
            elif multiplier == 2.0:
                weaks.append(type_name)
        
        print(f"\nType Defense Profile (18D):")
        print(f"  Resists ({len(resists)}):  {', '.join(resists) if resists else 'None'}")
        print(f"  Weak to ({len(weaks)}):    {', '.join(weaks) if weaks else 'None'}")
    
    # Summary
    print("\n" + "=" * 80)
    print(f"✅ Found {found_count}/{len(pokemon_names)} Pokemon")
    print("=" * 80)
    
    return df_with_features


def compare_pokemon(pokemon_names, csv_path=None):
    """
    Display Pokemon side-by-side for easy comparison.
    
    Args:
        pokemon_names: List of Pokemon names to compare
        csv_path: Optional path to CSV file
    """
    if csv_path is None:
        data_folder = pathlib.Path(__file__).resolve().parents[1] / "data"
        csv_path = data_folder / "test" / "fully_evolved_pokemon_stats.csv"
    
    df = pd.read_csv(csv_path)
    df_with_features = calculate_all_features(df)
    
    # Filter to requested Pokemon
    df_filtered = df_with_features[df_with_features["name"].isin([n.lower() for n in pokemon_names])]
    
    if df_filtered.empty:
        print("⚠️  None of the specified Pokemon were found")
        return
    
    # Select columns to display (core features only for readability)
    display_cols = [
        "name", "type1", "type2", "hp", "attack", "defense", 
        "special-attack", "special-defense", "speed",
        "offensive_index", "defensive_index", "speed_percentile",
        "physical_special_bias"
    ]
    
    pd.set_option('display.max_columns', None)
    pd.set_option('display.width', None)
    pd.set_option('display.max_colwidth', None)
    
    print("\n" + "=" * 120)
    print("POKEMON COMPARISON TABLE")
    print("=" * 120)
    print(df_filtered[display_cols].to_string(index=False))
    print("=" * 120)
    
    return df_filtered


if __name__ == "__main__":
    # Example 1: Test Eeveelutions (showcases archetype diversity)
    print("\n### TEST 1: Eeveelutions (Archetype Diversity) ###")
    eeveelutions = ["leafeon", "espeon", "umbreon", "jolteon", "flareon", "vaporeon", "glaceon", "sylveon"]
    test_pokemon_features(eeveelutions)
    
    print("\n\n")
    
    # Example 2: Test evolution chains (stat progression)
    print("### TEST 2: Machamp Evolution Line (Stat Progression) ###")
    machop_line = ["machop", "machoke", "machamp"]
    test_pokemon_features(machop_line)
    
    print("\n\n")
    
    # Example 3: Compare legendaries
    print("### TEST 3: Legendary Comparison ###")
    legendaries = ["mewtwo", "lugia", "rayquaza", "arceus"]
    compare_pokemon(legendaries)
    
    print("\n\n")
    
    # Example 4: Custom test - Uncomment and modify as needed
    # custom_pokemon = ["pikachu", "charizard", "blastoise", "venusaur"]
    # test_pokemon_features(custom_pokemon)
