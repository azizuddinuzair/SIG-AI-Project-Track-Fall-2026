"""
K-Sweep Test with Enhanced Features
Tests k=5-12 using the enriched feature set (with derived ratios, type coverage, interaction features)
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import RobustScaler
from sklearn.decomposition import PCA
from sklearn.mixture import GaussianMixture
from sklearn.metrics import silhouette_score, davies_bouldin_score
import warnings

warnings.filterwarnings('ignore')


def load_enhanced_features():
    """Load enhanced feature data with derived features"""
    proj_path = Path(__file__).parents[2]  # Proj1/
    data_path = proj_path / "data" / "cluster_data"
    df = pd.read_csv(data_path / "pokemon_archetype_features_enhanced.csv")
    
    # Select feature columns (skip name, type1, type2)
    feature_cols = [col for col in df.columns 
                   if col.lower() not in ['name', 'type1', 'type2']]
    X = df[feature_cols].values.astype(float)
    
    return X, df, feature_cols


def test_k_values_enhanced(X, k_values=range(5, 13)):
    """Test different k values with RobustScaler + 85% PCA using enhanced features"""
    
    print("\n" + "="*80)
    print("K-SWEEP TEST: Enhanced Features (with derived ratios & type coverage)")
    print("="*80 + "\n")
    
    # Apply RobustScaler
    print("Applying RobustScaler to enhanced features...")
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Apply PCA to 85% variance
    print("Applying PCA (85% variance)...")
    pca = PCA(n_components=0.85)
    X_pca = pca.fit_transform(X_scaled)
    n_components = X_pca.shape[1]
    variance_explained = np.sum(pca.explained_variance_ratio_)
    
    print(f"✓ {X.shape[0]} Pokémon reduced to {n_components}D (variance: {variance_explained*100:.1f}%)\n")
    
    results = []
    
    print("Testing k values...\n")
    print(f"{'k':<3} {'Silhouette':<12} {'Davies-Bouldin':<16} {'BIC':<12} {'Improvement':<12}")
    print("-" * 60)
    
    prev_silhouette = None
    
    for k in k_values:
        print(f"{k}  ", end='', flush=True)
        
        # Train GMM
        gmm = GaussianMixture(n_components=k, random_state=42, n_init=10)
        labels = gmm.fit_predict(X_pca)
        
        # Calculate metrics
        silhouette = silhouette_score(X_pca, labels)
        davies_bouldin = davies_bouldin_score(X_pca, labels)
        bic = gmm.bic(X_pca)
        
        # Calculate improvement from previous k
        if prev_silhouette is not None:
            improvement = ((silhouette - prev_silhouette) / prev_silhouette) * 100
            improvement_str = f"{improvement:+.2f}%"
        else:
            improvement_str = "-"
        
        print(f"{silhouette:.4f}     {davies_bouldin:.4f}          {bic:>10.1f}   {improvement_str:>10}")
        
        # Store result
        results.append({
            'k': k,
            'silhouette': silhouette,
            'davies_bouldin': davies_bouldin,
            'bic': bic,
            'n_components_used': n_components,
            'variance_explained': variance_explained,
            'labels': labels,
            'gmm': gmm,
            'X_pca': X_pca
        })
        
        prev_silhouette = silhouette
    
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80 + "\n")
    
    # Find best by silhouette
    best_by_sil = max(results, key=lambda x: x['silhouette'])
    print(f"🏆 Best Silhouette Score: k={best_by_sil['k']}")
    print(f"   Silhouette: {best_by_sil['silhouette']:.4f}")
    print(f"   Davies-Bouldin: {best_by_sil['davies_bouldin']:.4f} (lower is better)")
    print(f"   BIC: {best_by_sil['bic']:.1f}")
    
    print(f"\nComparison to baseline (k=12 with original features):")
    print(f"   Original k=12:      Silhouette = 0.3020")
    print(f"   Enhanced k={best_by_sil['k']:2d}:     Silhouette = {best_by_sil['silhouette']:.4f}")
    
    improvement_from_baseline = ((best_by_sil['silhouette'] - 0.3020) / 0.3020) * 100
    if improvement_from_baseline > 0:
        print(f"   ✅ Improvement: {improvement_from_baseline:+.2f}%")
    else:
        print(f"   ⚠️  Change: {improvement_from_baseline:.2f}%")
    
    # Create visualization
    print("\nGenerating visualization...\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Silhouette vs k
    ax = axes[0]
    k_values_list = [r['k'] for r in results]
    silhouettes = [r['silhouette'] for r in results]
    
    ax.plot(k_values_list, silhouettes, marker='o', linewidth=2, markersize=8, color='steelblue', label='Enhanced Features')
    ax.axhline(y=0.3020, color='red', linestyle='--', alpha=0.5, label='Baseline (k=12, original features)')
    ax.axvline(x=best_by_sil['k'], color='green', linestyle='--', alpha=0.7)
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Enhanced Feature: Silhouette Score vs Cluster Count', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(k_values_list)
    
    # Plot 2: Davies-Bouldin Index vs k
    ax = axes[1]
    db_scores = [r['davies_bouldin'] for r in results]
    
    ax.plot(k_values_list, db_scores, marker='s', linewidth=2, markersize=8, color='coral')
    ax.axvline(x=best_by_sil['k'], color='green', linestyle='--', alpha=0.7, label=f'Best (k={best_by_sil["k"]})')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Davies-Bouldin Index (lower is better)', fontsize=12)
    ax.set_title('Enhanced Feature: Cluster Separation Quality', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(k_values_list)
    
    plt.tight_layout()
    
    test_results_path = Path(__file__).parents[2] / "reports" / "clustering_analysis" / "test_results"
    test_results_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(test_results_path / 'k_sweep_enhanced_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: k_sweep_enhanced_comparison.png\n")
    
    # Save detailed results to CSV
    summary_data = []
    for result in results:
        summary_data.append({
            'k': result['k'],
            'Silhouette_Score': result['silhouette'],
            'Davies_Bouldin': result['davies_bouldin'],
            'BIC': result['bic'],
            'PCA_Components': result['n_components_used'],
            'Variance_Explained': result['variance_explained']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(test_results_path / 'k_sweep_enhanced_results.csv', index=False)
    print("✅ Saved: k_sweep_enhanced_results.csv\n")
    
    print("="*80 + "\n")
    
    return results, best_by_sil


if __name__ == '__main__':
    X, df, feature_cols = load_enhanced_features()
    print(f"\n✅ Loaded {len(feature_cols)} features (original + derived)")
    results, best = test_k_values_enhanced(X, k_values=range(5, 13))
