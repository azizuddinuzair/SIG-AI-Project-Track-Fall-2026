"""
Clustering Pipeline with A/B Testing: Full Dataset vs. Filtered (No Legendaries)

This pipeline implements the complete clustering workflow with comparative analysis:

Phase 1: Standardization & Dimensionality Reduction
- RobustScaler: Normalize 23D feature space (robust to outliers)
- PCA: Reduce to 85% variance (expected ~10 dimensions)

Phase 2: A/B Testing
- Variant A: Cluster full dataset (535 Pokemon, includes legendaries)
- Variant B: Cluster filtered dataset (473 Pokemon, BST <= 580)
- For each: Test GMM with k=5,6,7,8; select best k by silhouette score

Phase 3: Comparison & Recommendation
- Print silhouette, Davies-Bouldin, BIC for all models
- Recommend: Use full or filtered based on silhouette delta
- Save winning model + cluster assignments

Output Files:
- scaler.pkl, pca.pkl: Transformers (needed for Phase 2 GA)
- gmm_full_k*.pkl, gmm_filtered_k*.pkl: All trained models
- cluster_labels_*.npy: Cluster assignments for each variant/k
- clustering_comparison_report.txt: Summary stats and recommendation
- pca_variance_plot.png: Cumulative variance explained
- silhouette_comparison.png: Side-by-side silhouette comparison
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pathlib
import joblib
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings
warnings.filterwarnings('ignore')


def load_features():
    """Load engineered features from CSV."""
    data_folder = pathlib.Path(__file__).resolve().parents[2] / "data"
    features_csv = data_folder / "cluster_data" / "pokemon_archetype_features.csv"
    
    if not features_csv.exists():
        raise FileNotFoundError(f"Missing {features_csv}")
    
    df = pd.read_csv(features_csv)
    
    # Extract 23D feature columns (skip metadata)
    feature_cols = [col for col in df.columns if col.startswith('type_defense_') or 
                   col in ['offensive_index', 'defensive_index', 'speed_percentile', 'physical_special_bias']]
    
    return df, feature_cols


def prepare_datasets(df, feature_cols):
    """Prepare full and filtered datasets."""
    # Calculate BST for filtering
    df['bst'] = (df['hp'] + df['attack'] + df['defense'] + 
                 df['special-attack'] + df['special-defense'] + df['speed'])
    
    # Full dataset
    X_full = df[feature_cols].values
    
    # Filtered dataset (Regular Pokemon: BST <= 580)
    df_filtered = df[df['bst'] <= 580].copy()
    X_filtered = df_filtered[feature_cols].values
    
    return X_full, X_filtered, df, df_filtered


def standardize_features(X_full, X_filtered):
    """Apply RobustScaler to both datasets using same scaler (fit on full)."""
    scaler = RobustScaler()
    scaler.fit(X_full)  # Fit on full dataset
    
    X_full_scaled = scaler.transform(X_full)
    X_filtered_scaled = scaler.transform(X_filtered)
    
    return X_full_scaled, X_filtered_scaled, scaler


def reduce_dimensionality(X_full_scaled, X_filtered_scaled):
    """Apply PCA to both datasets using same PCA (fit on full)."""
    pca = PCA(n_components=0.85)
    pca.fit(X_full_scaled)
    
    X_full_pca = pca.transform(X_full_scaled)
    X_filtered_pca = pca.transform(X_filtered_scaled)
    
    return X_full_pca, X_filtered_pca, pca


def fit_gmm_models(X_full_pca, X_filtered_pca, k_values=[5, 6, 7, 8, 9, 10, 11, 12]):
    """Train GMM models for each k on both datasets."""
    results = {
        'full': {},
        'filtered': {}
    }
    
    print("\n🤖 Training GMM models...")
    
    for k in k_values:
        # Full dataset
        print(f"   k={k} on full dataset...", end=' ', flush=True)
        gmm_full = GaussianMixture(n_components=k, random_state=42, n_init=10)
        labels_full = gmm_full.fit_predict(X_full_pca)
        sil_full = silhouette_score(X_full_pca, labels_full)
        db_full = davies_bouldin_score(X_full_pca, labels_full)
        print(f"✓ (SIL={sil_full:.3f})")
        
        results['full'][k] = {
            'model': gmm_full,
            'labels': labels_full,
            'silhouette': sil_full,
            'davies_bouldin': db_full,
            'bic': gmm_full.bic(X_full_pca),
            'aic': gmm_full.aic(X_full_pca)
        }
        
        # Filtered dataset
        print(f"   k={k} on filtered dataset...", end=' ', flush=True)
        gmm_filtered = GaussianMixture(n_components=k, random_state=42, n_init=10)
        labels_filtered = gmm_filtered.fit_predict(X_filtered_pca)
        sil_filtered = silhouette_score(X_filtered_pca, labels_filtered)
        db_filtered = davies_bouldin_score(X_filtered_pca, labels_filtered)
        print(f"✓ (SIL={sil_filtered:.3f})")
        
        results['filtered'][k] = {
            'model': gmm_filtered,
            'labels': labels_filtered,
            'silhouette': sil_filtered,
            'davies_bouldin': db_filtered,
            'bic': gmm_filtered.bic(X_filtered_pca),
            'aic': gmm_filtered.aic(X_filtered_pca)
        }
    
    return results


def select_best_k(results):
    """Select best k for each variant based on silhouette score."""
    best_full_k = max(results['full'].keys(), key=lambda k: results['full'][k]['silhouette'])
    best_filtered_k = max(results['filtered'].keys(), key=lambda k: results['filtered'][k]['silhouette'])
    
    return best_full_k, best_filtered_k


def create_comparison_report(results, X_full_pca, X_filtered_pca, pca, scaler):
    """Generate comparison report and recommendation."""
    best_full_k, best_filtered_k = select_best_k(results)
    
    sil_full_best = results['full'][best_full_k]['silhouette']
    sil_filtered_best = results['filtered'][best_filtered_k]['silhouette']
    sil_delta = sil_full_best - sil_filtered_best
    
    report = []
    report.append("\n" + "=" * 80)
    report.append("CLUSTERING A/B TEST REPORT: Full Dataset vs. Filtered (No Legendaries)")
    report.append("=" * 80)
    
    report.append(f"\nDataset Sizes:")
    report.append(f"   Full Dataset: {len(X_full_pca)} Pokemon")
    report.append(f"   Filtered Dataset: {len(X_filtered_pca)} Pokemon (BST <= 580, removed {len(X_full_pca) - len(X_filtered_pca)} legendaries)")
    
    report.append(f"\nPCA Dimensionality Reduction:")
    report.append(f"   Original dimensions: 23D")
    report.append(f"   Reduced dimensions: {X_full_pca.shape[1]}D")
    report.append(f"   Variance explained: {np.sum(pca.explained_variance_ratio_):.1%}")
    
    # Scaler statistics (varies by scaler type)
    scaler_name = scaler.__class__.__name__
    report.append(f"\n{scaler_name} Statistics (fit on full dataset):")
    
    if hasattr(scaler, 'mean_'):
        report.append(f"   Mean (original scale): {scaler.mean_[:4]}")
        report.append(f"   Std (original scale): {scaler.scale_[:4]}")
    elif hasattr(scaler, 'center_'):
        report.append(f"   Center: {scaler.center_[:4]}")
        report.append(f"   Scale: {scaler.scale_[:4]}")
    
    report.append(f"\n" + "-" * 80)
    report.append("GMM MODEL COMPARISON (by k)")
    report.append("-" * 80)
    
    report.append(f"\n{'k':<3} {'Variant':<10} {'Silhouette':<12} {'Davies-Bouldin':<16} {'BIC':<12}")
    report.append("-" * 55)
    
    for k in sorted(results['full'].keys()):
        sil_f = results['full'][k]['silhouette']
        db_f = results['full'][k]['davies_bouldin']
        bic_f = results['full'][k]['bic']
        report.append(f"{k:<3} {'Full':<10} {sil_f:<12.4f} {db_f:<16.4f} {bic_f:<12.1f}")
        
        sil_g = results['filtered'][k]['silhouette']
        db_g = results['filtered'][k]['davies_bouldin']
        bic_g = results['filtered'][k]['bic']
        report.append(f"{k:<3} {'Filtered':<10} {sil_g:<12.4f} {db_g:<16.4f} {bic_g:<12.1f}")
        report.append("")
    
    report.append(f"\n" + "-" * 80)
    report.append("BEST MODELS SELECTED (by silhouette score)")
    report.append("-" * 80)
    report.append(f"\nFull Dataset Best: k={best_full_k}")
    report.append(f"   Silhouette: {sil_full_best:.4f}")
    report.append(f"   Davies-Bouldin: {results['full'][best_full_k]['davies_bouldin']:.4f}")
    report.append(f"   BIC: {results['full'][best_full_k]['bic']:.1f}")
    
    report.append(f"\nFiltered Dataset Best: k={best_filtered_k}")
    report.append(f"   Silhouette: {sil_filtered_best:.4f}")
    report.append(f"   Davies-Bouldin: {results['filtered'][best_filtered_k]['davies_bouldin']:.4f}")
    report.append(f"   BIC: {results['filtered'][best_filtered_k]['bic']:.1f}")
    
    report.append(f"\nSilhouette Delta (Full - Filtered): {sil_delta:.4f}")
    
    report.append(f"\n" + "=" * 80)
    report.append("RECOMMENDATION")
    report.append("=" * 80)
    
    if sil_delta > 0.05:
        recommendation = "✅ USE FULL DATASET (includes legendaries)"
        reason = "Legendaries improve clustering quality (silhouette delta > 0.05)"
    elif sil_delta < -0.05:
        recommendation = "❌ FILTER LEGENDARIES BEFORE CLUSTERING"
        reason = "Removing legendaries improves clustering quality (silhouette delta < -0.05)"
    else:
        recommendation = "✅ USE FULL DATASET (essentially equivalent)"
        reason = "Silhouette scores are statistically equivalent; preserving data is preferred"
    
    report.append(f"\n{recommendation}")
    report.append(f"\nReason: {reason}")
    report.append(f"\nSelected Configuration:")
    report.append(f"   Dataset: Full {len(X_full_pca)} Pokemon")
    report.append(f"   k: {best_full_k}")
    report.append(f"   Expected cluster sizes: {len(X_full_pca) // best_full_k} ± 50 Pokemon per cluster")
    
    report.append(f"\nNext Steps:")
    report.append(f"   1. Analyze cluster centroids in PCA space")
    report.append(f"   2. Interpret clusters as Pokemon archetypes (sweeper, tank, etc.)")
    report.append(f"   3. Implement GA team optimization using cluster assignments")
    
    report.append("\n" + "=" * 80 + "\n")
    
    return "\n".join(report)


def save_outputs(output_dir, results, scaler, pca, X_full_pca, X_filtered_pca):
    """Save all models, scalers, and cluster assignments."""
    output_dir = pathlib.Path(output_dir)
    models_dir = output_dir / "models"
    models_dir.mkdir(parents=True, exist_ok=True)
    
    print("\n💾 Saving outputs...")
    
    # Save transformers
    joblib.dump(scaler, models_dir / "scaler.pkl")
    joblib.dump(pca, models_dir / "pca.pkl")
    print(f"   ✓ models/scaler.pkl, models/pca.pkl")
    
    # Save all GMM models
    for k in results['full'].keys():
        joblib.dump(results['full'][k]['model'], models_dir / f"gmm_full_k{k}.pkl")
        joblib.dump(results['filtered'][k]['model'], models_dir / f"gmm_filtered_k{k}.pkl")
    print(f"   ✓ models/gmm_full_k*.pkl, models/gmm_filtered_k*.pkl ({len(results['full'])} models each)")
    
    # Save cluster labels
    for k in results['full'].keys():
        np.save(models_dir / f"cluster_labels_full_k{k}.npy", results['full'][k]['labels'])
        np.save(models_dir / f"cluster_labels_filtered_k{k}.npy", results['filtered'][k]['labels'])
    print(f"   ✓ models/cluster_labels_*.npy")
    
    # Save PCA features
    np.save(models_dir / "features_pca_full.npy", X_full_pca)
    np.save(models_dir / "features_pca_filtered.npy", X_filtered_pca)
    print(f"   ✓ models/features_pca_*.npy")


def plot_pca_variance(pca, output_dir):
    """Plot cumulative explained variance."""
    output_dir = pathlib.Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    cumsum = np.cumsum(pca.explained_variance_ratio_)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(cumsum) + 1), cumsum, 'bo-', linewidth=2, markersize=6)
    ax.axhline(y=0.90, color='r', linestyle='--', linewidth=2, label='90% variance target')
    ax.fill_between(range(1, len(cumsum) + 1), cumsum, alpha=0.3)
    ax.set_xlabel('Number of Components', fontsize=12)
    ax.set_ylabel('Cumulative Explained Variance', fontsize=12)
    ax.set_title('PCA: Variance Explained vs. Number of Components', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=11)
    ax.set_xticks(range(0, len(cumsum) + 1, 2))
    
    plt.tight_layout()
    plt.savefig(plots_dir / "pca_variance_plot.png", dpi=150, bbox_inches='tight')
    print(f"   ✓ plots/pca_variance_plot.png")
    plt.close()


def plot_silhouette_comparison(results, output_dir):
    """Plot silhouette comparison across k and variants."""
    output_dir = pathlib.Path(output_dir)
    plots_dir = output_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    k_values = sorted(results['full'].keys())
    sil_full = [results['full'][k]['silhouette'] for k in k_values]
    sil_filtered = [results['filtered'][k]['silhouette'] for k in k_values]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = np.arange(len(k_values))
    width = 0.35
    
    bars1 = ax.bar(x - width/2, sil_full, width, label='Full Dataset (535 Pokemon)', color='steelblue', alpha=0.8)
    bars2 = ax.bar(x + width/2, sil_filtered, width, label='Filtered (473 Pokemon, BST<=580)', color='coral', alpha=0.8)
    
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('GMM Silhouette Score: Full vs. Filtered Dataset', fontsize=13, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels([f'k={k}' for k in k_values])
    ax.legend(fontsize=11)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.3f}', ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(plots_dir / "silhouette_comparison.png", dpi=150, bbox_inches='tight')
    print(f"   ✓ plots/silhouette_comparison.png")
    plt.close()


def main():
    print("\n" + "=" * 80)
    print("CLUSTERING PIPELINE: A/B Testing Full vs. Filtered Dataset")
    print("=" * 80)
    
    # Load and prepare data
    print("\n📂 Loading engineered features...")
    df, feature_cols = load_features()
    print(f"   Loaded {len(df)} Pokemon with {len(feature_cols)} features")
    
    print("\n📊 Preparing datasets...")
    X_full, X_filtered, df_full, df_filtered = prepare_datasets(df, feature_cols)
    print(f"   Full dataset: {X_full.shape[0]} x {X_full.shape[1]} (23D)")
    print(f"   Filtered dataset: {X_filtered.shape[0]} x {X_filtered.shape[1]} (23D)")
    print(f"   Removed: {X_full.shape[0] - X_filtered.shape[0]} legendaries (BST > 580)")
    
    # Standardize
    print("\n⚖️  Normalizing features (RobustScaler)...")
    X_full_scaled, X_filtered_scaled, scaler = standardize_features(X_full, X_filtered)
    print(f"   ✓ Both datasets normalized")
    
    # PCA
    print("\n📉 Dimensionality reduction (PCA → 85% variance)...")
    X_full_pca, X_filtered_pca, pca = reduce_dimensionality(X_full_scaled, X_filtered_scaled)
    print(f"   23D → {X_full_pca.shape[1]}D")
    print(f"   Cumulative variance explained: {np.sum(pca.explained_variance_ratio_):.1%}")
    
    # Train GMM models
    print("\n")
    results = fit_gmm_models(X_full_pca, X_filtered_pca, k_values=[5, 6, 7, 8, 9, 10, 11, 12])
    
    # Generate report
    report = create_comparison_report(results, X_full_pca, X_filtered_pca, pca, scaler)
    print(report)
    
    # Save outputs
    output_dir = pathlib.Path(__file__).resolve().parents[2] / "reports" / "clustering_analysis"
    save_outputs(output_dir, results, scaler, pca, X_full_pca, X_filtered_pca)
    
    # Generate visualizations
    print("\n📈 Creating visualizations...")
    plot_pca_variance(pca, output_dir)
    plot_silhouette_comparison(results, output_dir)
    
    # Save report to file
    report_path = output_dir / "reports" / "clustering_comparison_report.txt"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, 'w', encoding='utf-8') as f:
        f.write(report)
    print(f"   ✓ reports/clustering_comparison_report.txt")
    
    print(f"\n✅ Pipeline complete! Outputs saved to: {output_dir}\n")


if __name__ == "__main__":
    main()
