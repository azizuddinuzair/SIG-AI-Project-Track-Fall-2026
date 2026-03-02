"""
File to create new features for the pokemon dataset. This includes:
- Offensive Index: Attack + Special Attack
- Defensive Index: (HP * 0.5) + Defense + Special Defense
- Speed Percentile: rank(speed) / N 
- Physical Special Bias: (attack - sp_attack) / (attack + sp_attack)
- Bulk to Speed Ratio: defense_index / speed


Testing feature engineering with different pokemon -->
Physical Offensive Index: Leafeon
Special Offensive Index: Espeon
Defense Index: Umbreon
Speed Percentile: Jolteon
Physical Special Bias: Flareon
Bulk to Speed Ratio: Vaporeon


"""

import pathlib
import pandas as pd


data_folder = pathlib.Path(__file__).resolve().parents[1] / "data"
pokemon_stats_csv = data_folder / "all_pokemon_stats.csv"


def calculate_all_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate derived features for all Pokemon in the dataframe.
    
    Args:
        df: DataFrame with base stats (hp, attack, defense, special-attack, special-defense, speed)
    
    Returns:
        DataFrame with original columns plus engineered features
    """
    result = df.copy()

    # Calculate Offensive Index (Attack + Special Attack)
    result["offensive_index"] = result["attack"] + result["special-attack"]

    # Calculate Defensive Index (HP * 0.5) + Defense + Special Defense
    result["defensive_index"] = (result["hp"] * 0.5) + result["defense"] + result["special-defense"]

    # Calculate Speed Percentile: rank(speed) / N
    result["speed_percentile"] = result["speed"].rank(pct=True)

    # Calculate Physical Special Bias (Attack - Special Attack) / (Attack + Special Attack)
    # Handle division by zero: if both are 0, set bias to 0
    total_offense = result["attack"] + result["special-attack"]
    result["physical_special_bias"] = (result["attack"] - result["special-attack"]) / total_offense
    result["physical_special_bias"] = result["physical_special_bias"].fillna(0)

    # Calculate Bulk to Speed Ratio (Defensive Index / Speed)
    # Handle division by zero: if speed is 0, set ratio to a large number
    result["bulk_to_speed_ratio"] = result["defensive_index"] / result["speed"].replace(0, 1e-6)
    
    return result


if __name__ == "__main__":
    """
    Generate engineered features for all Pokemon and save to CSV.
    For testing specific Pokemon, use test_pokemon_features.py instead.
    """
    # Load the full dataset
    df = pd.read_csv(pokemon_stats_csv)
    
    # Calculate features for all Pokemon
    print("Calculating engineered features for all Pokemon...")
    df_with_features = calculate_all_features(df)
    
    # Save to CSV
    output_path = data_folder / "pokemon_with_features.csv"
    df_with_features.to_csv(output_path, index=False)
    
    print(f"✅ Successfully engineered features for {len(df_with_features)} Pokemon")
    print(f"💾 Saved to: {output_path}")
    print(f"\nFeatures added:")
    print("  - offensive_index")
    print("  - defensive_index")
    print("  - speed_percentile")
    print("  - physical_special_bias")
    print("  - bulk_to_speed_ratio")
    print(f"\n💡 To test specific Pokemon, run: python test_pokemon_features.py")

