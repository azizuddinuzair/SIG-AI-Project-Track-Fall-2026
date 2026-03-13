"""
Enhanced Feature Engineering for Pokemon Clustering
Adds derived ratios, type coverage, and interaction features to improve generalist separation.

Input: pokemon_archetype_features.csv
Output: pokemon_archetype_features_enhanced.csv
"""

import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import RobustScaler
import warnings

warnings.filterwarnings('ignore')


def load_raw_features():
    """Load raw Pokemon data from cluster_data"""
    data_path = Path(__file__).parents[2] / "data" / "cluster_data"
    df = pd.read_csv(data_path / "pokemon_archetype_features.csv")
    return df


def compute_derived_features(df):
    """Compute derived ratios, composite stats, and interaction features"""
    
    print("Computing derived features...\n")
    
    # 1∩╕ÅΓâú RATIOS & COMPOSITE STATS (Raw scales, will normalize later)
    
    # Offense/Defense bias (0=pure special, 1=pure physical)
    df['Offense_Bias'] = df['attack'] / (df['attack'] + df['special-attack'] + 1e-6)
    
    # Defense lean (how much DEF emphasis vs speed)
    df['Defense_Lean'] = df['defense'] / (df['speed'] + 1e-6)
    
    # Physical bulk: HP ├ù DEF (raw, not percentage)
    df['Bulk_Physical'] = df['hp'] * df['defense']
    
    # Special bulk: HP ├ù SPA (raw, not percentage)
    df['Bulk_Special'] = df['hp'] * df['special-attack']
    
    # Sweep index: speed relative to offense (high = fast, low = slow)
    total_offense = df['attack'] + df['special-attack']
    df['Sweep_Index'] = df['speed'] / (total_offense + 1e-6)
    
    # 2∩╕ÅΓâú TYPE COVERAGE FEATURES
    
    # Count resistances from type_defense columns (< 1.0 = resistant)
    type_defense_cols = [col for col in df.columns if col.startswith('type_defense_')]
    df['Num_Resistances'] = (df[type_defense_cols] < 1.0).sum(axis=1)
    
    # Count weaknesses (> 1.0 = weak)
    df['Num_Weaknesses'] = (df[type_defense_cols] > 1.0).sum(axis=1)
    
    # Type synergy: resistances / (resistances + weaknesses)
    # High = good defensive typing, Low = vulnerable typing
    total_coverage = df['Num_Resistances'] + df['Num_Weaknesses']
    df['Type_Synergy'] = df['Num_Resistances'] / (total_coverage + 1e-6)
    
    # 3∩╕ÅΓâú INTERACTION FEATURES (Encode Role Semantics)
    
    # Pivot potential: balance of offense+defense with decent speed
    df['Pivot_Potential'] = (df['defense'] + df['special-attack']) * (df['speed_percentile'] + 0.1)
    
    # Wall potential: maximize bulk while being slow (good for switching)
    df['Wall_Potential'] = (df['hp'] + df['defense']) * (1.0 - df['speed_percentile'] + 0.1)
    
    # Balance metric: how evenly distributed are stats (high = lopsided, low = balanced)
    core_stats = df[['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']]
    df['Stat_Balance'] = core_stats.max(axis=1) / (core_stats.min(axis=1) + 1e-6)
    
    return df


def normalize_features(df_with_derived):
    """Normalize derived features to prevent PCA dominance by high-magnitude features"""
    
    print("Normalizing derived features...\n")
    
    # Features to normalize (keep original archetype features as-is)
    derived_features = [
        'Offense_Bias', 'Defense_Lean', 'Bulk_Physical', 'Bulk_Special',
        'Sweep_Index', 'Num_Resistances', 'Num_Weaknesses', 'Type_Synergy',
        'Pivot_Potential', 'Wall_Potential', 'Stat_Balance'
    ]
    
    # Apply RobustScaler to derived features (handled high-magnitude features like bulk)
    scaler = RobustScaler()
    df_normalized = df_with_derived.copy()
    df_normalized[derived_features] = scaler.fit_transform(df_with_derived[derived_features])
    
    return df_normalized


def save_enhanced_features(df_enhanced):
    """Save enhanced features to new CSV"""
    
    data_path = Path(__file__).parents[2] / "data" / "cluster_data"
    output_path = data_path / "pokemon_archetype_features_enhanced.csv"
    
    df_enhanced.to_csv(output_path, index=False)
    print(f"Γ£à Saved enhanced features to: pokemon_archetype_features_enhanced.csv")
    print(f"   Original features: 25 columns")
    print(f"   Enhanced features: {df_enhanced.shape[1]} columns (+11 derived)\n")
    
    return output_path


def print_feature_summary(df_enhanced):
    """Print statistics on new features"""
    
    print("="*80)
    print("ENHANCED FEATURE SUMMARY")
    print("="*80 + "\n")
    
    derived_features = [
        'Offense_Bias', 'Defense_Lean', 'Bulk_Physical', 'Bulk_Special',
        'Sweep_Index', 'Num_Resistances', 'Num_Weaknesses', 'Type_Synergy',
        'Pivot_Potential', 'Wall_Potential', 'Stat_Balance'
    ]
    
    print("Derived Features (Normalized):\n")
    
    print("1∩╕ÅΓâú  RATIOS & COMPOSITE STATS")
    print(f"   Offense_Bias:       {df_enhanced['Offense_Bias'].mean():.2f} ┬▒ {df_enhanced['Offense_Bias'].std():.2f}")
    print(f"                       (0=special, 1=physical)")
    print(f"   Defense_Lean:       {df_enhanced['Defense_Lean'].mean():.2f} ┬▒ {df_enhanced['Defense_Lean'].std():.2f}")
    print(f"                       (high=defensive focus)")
    print(f"   Bulk_Physical:      {df_enhanced['Bulk_Physical'].mean():.0f} ┬▒ {df_enhanced['Bulk_Physical'].std():.0f}")
    print(f"                       (HP ├ù DEF for survival)")
    print(f"   Bulk_Special:       {df_enhanced['Bulk_Special'].mean():.0f} ┬▒ {df_enhanced['Bulk_Special'].std():.0f}")
    print(f"                       (HP ├ù SPA for lasting power)")
    print(f"   Sweep_Index:        {df_enhanced['Sweep_Index'].mean():.2f} ┬▒ {df_enhanced['Sweep_Index'].std():.2f}")
    print(f"                       (high=sweeper, low=stall)\n")
    
    print("2∩╕ÅΓâú  TYPE COVERAGE")
    print(f"   Num_Resistances:    {df_enhanced['Num_Resistances'].mean():.1f} ┬▒ {df_enhanced['Num_Resistances'].std():.1f}")
    print(f"                       (avg defensive resistances)")
    print(f"   Num_Weaknesses:     {df_enhanced['Num_Weaknesses'].mean():.1f} ┬▒ {df_enhanced['Num_Weaknesses'].std():.1f}")
    print(f"                       (avg type weaknesses)")
    print(f"   Type_Synergy:       {df_enhanced['Type_Synergy'].mean():.2f} ┬▒ {df_enhanced['Type_Synergy'].std():.2f}")
    print(f"                       (0=vulnerable, 1=well-typed)\n")
    
    print("3∩╕ÅΓâú  INTERACTION FEATURES")
    print(f"   Pivot_Potential:    {df_enhanced['Pivot_Potential'].mean():.2f} ┬▒ {df_enhanced['Pivot_Potential'].std():.2f}")
    print(f"                       (DEF+SPA ├ù speed for switching)")
    print(f"   Wall_Potential:     {df_enhanced['Wall_Potential'].mean():.2f} ┬▒ {df_enhanced['Wall_Potential'].std():.2f}")
    print(f"                       (bulk ├ù low-speed = defensive role)")
    print(f"   Stat_Balance:       {df_enhanced['Stat_Balance'].mean():.2f} ┬▒ {df_enhanced['Stat_Balance'].std():.2f}")
    print(f"                       (1=balanced, high=lopsided)\n")
    
    print("="*80 + "\n")


def main():
    print("\n" + "="*80)
    print("ENHANCED FEATURE ENGINEERING FOR POKEMON CLUSTERING")
    print("="*80 + "\n")
    
    # Load raw features
    print("Loading raw Pokemon features...")
    df = load_raw_features()
    print(f"Γ£à Loaded {df.shape[0]} Pok├⌐mon with {df.shape[1]} initial features\n")
    
    # Compute derived features
    df_with_derived = compute_derived_features(df)
    print(f"Γ£à Computed 11 derived features\n")
    
    # Normalize derived features
    df_enhanced = normalize_features(df_with_derived)
    print(f"Γ£à Normalized derived features\n")
    
    # Print summary
    print_feature_summary(df_enhanced)
    
    # Save enhanced features
    output_path = save_enhanced_features(df_enhanced)
    
    print("="*80)
    print("NEXT STEPS")
    print("="*80 + "\n")
    print("1. Update clustering_pipeline.py to use pokemon_archetype_features_enhanced.csv")
    print("2. Run k-sweep test (k=5-12) to find optimal k with new features")
    print("3. Compare silhouette scores and archetype distributions")
    print("4. Verify generalist clusters split into more meaningful roles\n")


if __name__ == '__main__':
    main()
