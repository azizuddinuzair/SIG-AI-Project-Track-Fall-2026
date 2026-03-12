"""
Cluster Analyzer: Interpret Pokemon archetypes from GMM clusters.

This script analyzes the feature profiles of each cluster to determine what
archetype it represents (e.g., "Speed Sweeper", "Physical Tank", etc.)

Outputs:
- Detailed cluster profile report (console + file)
- CSV with cluster assignments for all Pokemon
- Archetype interpretation and top representative Pokemon
"""

import pandas as pd
import numpy as np
import pathlib
import joblib


def load_data_and_models():
    """Load the original features, PCA model, scaler, and GMM model."""
    data_folder = pathlib.Path(__file__).resolve().parents[2] / "data"
    models_folder = pathlib.Path(__file__).resolve().parents[2] / "reports" / "clustering_analysis"
    
    # Load engineered features
    features_csv = data_folder / "cluster_data" / "pokemon_archetype_features.csv"
    df = pd.read_csv(features_csv)
    
    # Load models from models subfolder
    models_folder = models_folder / "models"
    scaler = joblib.load(models_folder / "scaler.pkl")
    pca = joblib.load(models_folder / "pca.pkl")
    gmm = joblib.load(models_folder / "gmm_full_k12.pkl")
    
    # Load PCA features
    X_pca = np.load(models_folder / "features_pca_full.npy")
    
    # Get cluster assignments
    cluster_labels = gmm.predict(X_pca)
    
    return df, scaler, pca, gmm, X_pca, cluster_labels


def compute_cluster_centroids_original_space(df, cluster_labels):
    """Compute cluster centroids in original 23D feature space."""
    feature_cols = [col for col in df.columns if col.startswith('type_defense_') or 
                   col in ['offensive_index', 'defensive_index', 'speed_percentile', 'physical_special_bias']]
    
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    centroids = {}
    for k in sorted(df_with_clusters['cluster'].unique()):
        cluster_data = df_with_clusters[df_with_clusters['cluster'] == k][feature_cols]
        centroids[k] = cluster_data.mean()
    
    return centroids, feature_cols


def interpret_archetype(centroid_series):
    """
    Interpret what archetype a cluster represents based on feature values.
    Returns: (archetype_label, interpretation_text)
    """
    off_idx = centroid_series.get('offensive_index', 0)
    def_idx = centroid_series.get('defensive_index', 0)
    spd_pct = centroid_series.get('speed_percentile', 0)
    phys_bias = centroid_series.get('physical_special_bias', 0)
    
    # Extract type resistances (count how many types this cluster resists vs. weak to)
    type_cols = [col for col in centroid_series.index if col.startswith('type_defense_')]
    type_values = centroid_series[type_cols]
    resists = (type_values < 1.0).sum()  # Count 0.5x resistances
    weaks = (type_values > 1.0).sum()    # Count 2.0x weaknesses
    
    # Normalize stats relative to full dataset mean (from diagnostic data)
    # Regular Pokemon: off_idx μ=170.7±32.2, def_idx μ=207.0±41.1, spd_pct μ=0.5
    off_score = (off_idx - 170.7) / 32.2  # z-score
    def_score = (def_idx - 207.0) / 41.1
    phys_score = phys_bias  # Already normalized [-1, 1]
    
    # Classification logic using direct thresholds (NOT z-scores for speed)
    archetype = None
    interpretation = []
    
    # Primary classification by offensive/defensive balance
    if spd_pct > 0.65 and off_idx > 175:  # High speed + decent offense
        archetype = "Speed Sweeper"
        interpretation.append(f"High speed ({spd_pct:.1%} percentile) + good offense ({off_idx:.0f})")
        interpretation.append("Designed to outspeed and KO opponents before they move")
    elif spd_pct > 0.55 and off_idx > 175:  # Moderately high speed + decent offense
        archetype = "Fast Attacker"
        interpretation.append(f"Good speed ({spd_pct:.1%} percentile) + solid offense ({off_idx:.0f})")
        interpretation.append("Can outspeed many Pokemon while dealing good damage")
    elif spd_pct > 0.55 and phys_score > 0.15:  # Decent speed + physical bias
        archetype = "Physical Sweeper"
        interpretation.append(f"Good speed ({spd_pct:.1%} percentile) + physical bias ({phys_score:.2f})")
        interpretation.append("Physical attacker with enough speed to threaten many threats")
    elif phys_score > 0.20 and off_idx > 180:  # Strong physical bias + high offense
        archetype = "Physical Attacker"
        interpretation.append(f"Strong physical bias ({phys_score:.2f}) with high offense ({off_idx:.0f})")
        interpretation.append("Focuses on powerful physical attacks to overwhelm opponents")
    elif phys_score < -0.20 and off_idx > 175:  # Strong special bias + decent offense
        archetype = "Special Attacker"
        interpretation.append(f"Strong special bias ({phys_score:.2f}) with good offense ({off_idx:.0f})")
        interpretation.append("Focuses on powerful special attacks to overwhelm opponents")
    elif spd_pct < 0.35 and def_idx > 220:  # Low speed + high defense
        archetype = "Defensive Wall"
        interpretation.append(f"High defense ({def_idx:.0f}) + low speed ({spd_pct:.1%} percentile)")
        interpretation.append("Built to absorb hits and stall opponents; uses priority moves or setup")
    elif spd_pct < 0.40 and def_idx > 200:  # Low speed + good defense
        archetype = "Defensive Tank"
        interpretation.append(f"Good defense ({def_idx:.0f}) + low speed ({spd_pct:.1%} percentile)")
        interpretation.append("Tanky Pokemon designed to absorb damage and switch strategically")
    elif spd_pct > 0.50 and def_idx > 210:  # Decent speed + good defense
        archetype = "Defensive Pivot"
        interpretation.append(f"Good defense ({def_idx:.0f}) + moderate speed ({spd_pct:.1%} percentile)")
        interpretation.append("Balances defensiveness with mobility for pivoting and support")
    elif off_idx > 175 and def_idx > 200 and spd_pct > 0.40:  # Balanced
        archetype = "Balanced All-Rounder"
        interpretation.append(f"Balanced stats: OFF={off_idx:.0f}, DEF={def_idx:.0f}, SPD={spd_pct:.1%}")
        interpretation.append("No glaring weaknesses; adaptable to multiple roles")
    else:
        archetype = "Generalist"
        interpretation.append(f"Varied stat profile: OFF={off_idx:.0f}, DEF={def_idx:.0f}, SPD={spd_pct:.1%}")
        interpretation.append("Flexible but not specialized in any area")
    
    # Add physical vs special bias info
    if phys_score > 0.2:
        interpretation.append(f"Leaning PHYSICAL attacker (bias={phys_score:.2f})")
    elif phys_score < -0.2:
        interpretation.append(f"Leaning SPECIAL attacker (bias={phys_score:.2f})")
    else:
        interpretation.append("Mixed physical/special capabilities")
    
    # Add typing info
    interpretation.append(f"Type profile: {resists} resistances, {weaks} weaknesses (defensive coverage)")
    
    return archetype, "\n      ".join(interpretation)


def get_cluster_representatives(df, cluster_labels, k, n_top=5):
    """Get top representative Pokemon for a cluster (closest to centroid)."""
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    
    cluster_pokemon = df_with_clusters[df_with_clusters['cluster'] == k].copy()
    
    # Calculate distance to cluster mean (simple: sum of abs deviations)
    feature_cols = [col for col in df.columns if col.startswith('type_defense_') or 
                   col in ['offensive_index', 'defensive_index', 'speed_percentile', 'physical_special_bias']]
    
    cluster_mean = cluster_pokemon[feature_cols].mean()
    distances = ((cluster_pokemon[feature_cols] - cluster_mean).abs().sum(axis=1))
    
    top_indices = distances.nsmallest(n_top).index
    representatives = cluster_pokemon.loc[top_indices, ['name', 'type1', 'type2', 'hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']]
    
    return representatives


def main():
    print("\n" + "=" * 90)
    print("CLUSTER ARCHETYPE ANALYSIS")
    print("=" * 90)
    
    # Load everything
    print("\n📂 Loading data and models...")
    df, scaler, pca, gmm, X_pca, cluster_labels = load_data_and_models()
    print(f"   Loaded {len(df)} Pokemon with {len(cluster_labels)} cluster assignments")
    
    # Compute centroids
    print("\n📊 Computing cluster centroids...")
    centroids, feature_cols = compute_cluster_centroids_original_space(df, cluster_labels)
    print(f"   Computed centroids for {len(centroids)} clusters")
    
    # Generate report
    report = []
    report.append("\n" + "=" * 90)
    report.append("POKEMON ARCHETYPE CLUSTER PROFILES")
    report.append("=" * 90)
    
    archetypes = {}
    
    for k in sorted(centroids.keys()):
        centroid = centroids[k]
        cluster_size = (cluster_labels == k).sum()
        
        archetype_label, interpretation = interpret_archetype(centroid)
        archetypes[k] = archetype_label
        
        report.append(f"\n{'─' * 90}")
        report.append(f"CLUSTER {k}: {archetype_label} ({cluster_size} Pokemon, {cluster_size/len(df)*100:.1f}%)")
        report.append(f"{'─' * 90}")
        
        # Feature profile
        report.append(f"\nFeature Profile (Centroid Values):")
        report.append(f"   Offensive Index:      {centroid['offensive_index']:.1f}")
        report.append(f"   Defensive Index:      {centroid['defensive_index']:.1f}")
        report.append(f"   Speed Percentile:     {centroid['speed_percentile']:.3f}")
        report.append(f"   Physical/Special Bias: {centroid['physical_special_bias']:.3f}")
        
        # Type profile
        type_cols = [col for col in centroid.index if col.startswith('type_defense_')]
        type_resists = [col.replace('type_defense_', '') for col in type_cols if centroid[col] < 1.0]
        type_weaks = [col.replace('type_defense_', '') for col in type_cols if centroid[col] > 1.0]
        
        report.append(f"\nType Profile:")
        report.append(f"   Resists ({len(type_resists)}): {', '.join(type_resists)}")
        report.append(f"   Weak to ({len(type_weaks)}): {', '.join(type_weaks)}")
        
        # Interpretation
        report.append(f"\nInterpretation:")
        report.append(f"   {interpretation}")
        
        # Top representatives
        report.append(f"\nTop 5 Representative Pokemon:")
        representatives = get_cluster_representatives(df, cluster_labels, k, n_top=5)
        for idx, (i, row) in enumerate(representatives.iterrows(), 1):
            type_str = f"{row['type1']}" + (f"/{row['type2']}" if pd.notna(row['type2']) else "")
            report.append(f"   {idx}. {row['name'].capitalize():20} ({type_str:20}) HP:{row['hp']:3} ATK:{row['attack']:3} DEF:{row['defense']:3} SPA:{row['special-attack']:3} SPD:{int(row['speed']):3}")
    
    # Summary
    report.append(f"\n" + "=" * 90)
    report.append("ARCHETYPE SUMMARY")
    report.append("=" * 90)
    report.append("\nCluster → Archetype Mapping:")
    for k in sorted(archetypes.keys()):
        cluster_size = (cluster_labels == k).sum()
        report.append(f"   Cluster {k}: {archetypes[k]:30} ({cluster_size} Pokemon)")
    
    report.append(f"\nDiversity Check:")
    unique_archetypes = set(archetypes.values())
    report.append(f"   Unique archetypes: {len(unique_archetypes)}")
    report.append(f"   Archetypes: {', '.join(sorted(unique_archetypes))}")
    
    report.append("\n" + "=" * 90)
    
    # Print and save
    report_text = "\n".join(report)
    print(report_text)
    
    # Save to file
    output_dir = pathlib.Path(__file__).resolve().parents[2] / "reports" / "clustering_analysis"
    reports_dir = output_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    output_path = reports_dir / "cluster_archetype_profiles.txt"
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(report_text)
    
    # Print relative paths from project root
    proj_root = pathlib.Path(__file__).resolve().parents[2]
    try:
        rel_path = output_path.relative_to(proj_root)
        print(f"\n✅ Report saved to: {rel_path}")
    except ValueError:
        print(f"\n✅ Report saved to: {output_path}")
    
    # Save cluster assignments
    df_with_clusters = df.copy()
    df_with_clusters['cluster'] = cluster_labels
    df_with_clusters['archetype'] = df_with_clusters['cluster'].map(archetypes)
    
    data_dir = output_dir / "data"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "pokemon_with_clusters.csv"
    df_with_clusters.to_csv(csv_path, index=False)
    
    try:
        rel_csv_path = csv_path.relative_to(proj_root)
        print(f"✅ Cluster assignments saved to: {rel_csv_path}")
    except ValueError:
        print(f"✅ Cluster assignments saved to: {csv_path}")
    
    print()


if __name__ == "__main__":
    main()
