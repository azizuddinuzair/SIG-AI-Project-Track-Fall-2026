"""
Diagnostic script: Analyze legendary Pokémon skew in feature space.

This script loads engineered features and visualizes the distribution of
offensive_index and defensive_index, highlighting how legendaries cluster
separately in the feature space. Used to determine if filtering is needed
before GMM clustering.

Output:
- Console: Summary statistics and recommendation
- PNG: legendary_skew_diagnostic.png showing scatter and histogram plots
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib


def main():
    # Load the engineered features
    data_folder = pathlib.Path(__file__).resolve().parents[2] / "data"
    features_csv = data_folder / "cluster_data" / "pokemon_archetype_features.csv"
    
    if not features_csv.exists():
        print(f"❌ Error: {features_csv} not found")
        print("   Run: python Proj1/scripts/creating_csv/feature_engineering.py first")
        return
    
    print("📊 Loading engineered features...")
    df = pd.read_csv(features_csv)
    
    # Calculate Base Stat Total (BST)
    df['bst'] = (df['hp'] + df['attack'] + df['defense'] + 
                 df['special-attack'] + df['special-defense'] + df['speed'])
    
    # Identify legendaries (BST > 580 is standard threshold)
    df['is_legendary'] = df['bst'] > 580
    
    non_legendary = df[~df['is_legendary']]
    legendary = df[df['is_legendary']]
    
    # Create visualizations
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Offensive vs Defensive (colored by legend status)
    ax = axes[0, 0]
    ax.scatter(non_legendary['offensive_index'], non_legendary['defensive_index'], 
              alpha=0.5, s=40, label='Regular Pokémon', c='blue')
    if len(legendary) > 0:
        ax.scatter(legendary['offensive_index'], legendary['defensive_index'], 
                  alpha=0.8, s=100, label='Legendary (BST > 580)', c='red', marker='^')
    ax.set_xlabel('Offensive Index', fontsize=11)
    ax.set_ylabel('Defensive Index', fontsize=11)
    ax.set_title('Offensive vs Defensive: Legendary Skew Analysis', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Offensive Index Distribution
    ax = axes[0, 1]
    ax.hist(non_legendary['offensive_index'], alpha=0.6, bins=30, label='Regular', color='blue')
    if len(legendary) > 0:
        ax.hist(legendary['offensive_index'], alpha=0.7, bins=10, label='Legendary', color='red')
    ax.set_xlabel('Offensive Index', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Offensive Index Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 3: Defensive Index Distribution
    ax = axes[1, 0]
    ax.hist(non_legendary['defensive_index'], alpha=0.6, bins=30, label='Regular', color='blue')
    if len(legendary) > 0:
        ax.hist(legendary['defensive_index'], alpha=0.7, bins=10, label='Legendary', color='red')
    ax.set_xlabel('Defensive Index', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Defensive Index Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    # Plot 4: BST Distribution
    ax = axes[1, 1]
    ax.hist(non_legendary['bst'], alpha=0.6, bins=40, label='Regular', color='blue')
    if len(legendary) > 0:
        ax.hist(legendary['bst'], alpha=0.7, bins=15, label='Legendary', color='red')
    ax.axvline(x=580, color='green', linestyle='--', linewidth=2, label='Legendary Threshold (580)')
    ax.set_xlabel('Base Stat Total', fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.set_title('Base Stat Total Distribution', fontsize=12, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    output_png = data_folder / "cluster_data" / "legendary_skew_diagnostic.png"
    plt.savefig(output_png, dpi=150, bbox_inches='tight')
    print(f"📈 Visualization saved: {output_png}")
    plt.show()
    
    # Print summary statistics
    print("\n" + "=" * 70)
    print("LEGENDARY SKEW DIAGNOSTIC REPORT")
    print("=" * 70)
    
    print(f"\n📊 Dataset Summary:")
    print(f"   Total Pokémon: {len(df)}")
    print(f"   Regular Pokémon: {len(non_legendary)}")
    print(f"   Legendary Pokémon (BST > 580): {len(legendary)}")
    print(f"   Legendary %: {len(legendary) / len(df) * 100:.1f}%")
    
    print(f"\n📈 Offensive Index Statistics:")
    print(f"   Regular:   μ={non_legendary['offensive_index'].mean():.1f}, σ={non_legendary['offensive_index'].std():.1f}, range=[{non_legendary['offensive_index'].min():.0f}, {non_legendary['offensive_index'].max():.0f}]")
    print(f"   Legendary: μ={legendary['offensive_index'].mean():.1f}, σ={legendary['offensive_index'].std():.1f}, range=[{legendary['offensive_index'].min():.0f}, {legendary['offensive_index'].max():.0f}]")
    print(f"   Separation: {(legendary['offensive_index'].mean() - non_legendary['offensive_index'].mean()) / non_legendary['offensive_index'].std():.2f} std devs")
    
    print(f"\n🛡️  Defensive Index Statistics:")
    print(f"   Regular:   μ={non_legendary['defensive_index'].mean():.1f}, σ={non_legendary['defensive_index'].std():.1f}, range=[{non_legendary['defensive_index'].min():.0f}, {non_legendary['defensive_index'].max():.0f}]")
    print(f"   Legendary: μ={legendary['defensive_index'].mean():.1f}, σ={legendary['defensive_index'].std():.1f}, range=[{legendary['defensive_index'].min():.0f}, {legendary['defensive_index'].max():.0f}]")
    print(f"   Separation: {(legendary['defensive_index'].mean() - non_legendary['defensive_index'].mean()) / non_legendary['defensive_index'].std():.2f} std devs")
    
    print(f"\n⚡ Feature Space Assessment:")
    std_offensive = non_legendary['offensive_index'].std()
    std_defensive = non_legendary['defensive_index'].std()
    sep_off = (legendary['offensive_index'].mean() - non_legendary['offensive_index'].mean()) / std_offensive if len(legendary) > 0 else 0
    sep_def = (legendary['defensive_index'].mean() - non_legendary['defensive_index'].mean()) / std_defensive if len(legendary) > 0 else 0
    
    print(f"\n💡 RECOMMENDATION:")
    if len(legendary) / len(df) > 0.15:
        print(f"   ⚠️  FILTER LEGENDARIES BEFORE CLUSTERING")
        print(f"      Legendaries are {len(legendary) / len(df) * 100:.1f}% of dataset")
        print(f"      Separation in offensive/defensive space: {max(sep_off, sep_def):.2f} std devs")
        print(f"      → Risk of GMM clustering by stat magnitude instead of archetype")
        print(f"\n   Action: Run clustering_pipeline.py with filter_legendaries=True")
    elif len(legendary) / len(df) > 0.10:
        print(f"   ⚠️  MONITOR CLUSTERING (Borderline case)")
        print(f"      Legendaries are {len(legendary) / len(df) * 100:.1f}% of dataset")
        print(f"      Separation: {max(sep_off, sep_def):.2f} std devs")
        print(f"\n   Action: Cluster both ways (with/without legendaries), compare silhouette scores")
    else:
        print(f"   ✅ SAFE TO CLUSTER JOINTLY")
        print(f"      Legendaries are only {len(legendary) / len(df) * 100:.1f}% of dataset")
        print(f"      Separation: {max(sep_off, sep_def):.2f} std devs")
        print(f"\n   Action: Proceed with full clustering_pipeline.py (no filtering needed)")
    
    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
