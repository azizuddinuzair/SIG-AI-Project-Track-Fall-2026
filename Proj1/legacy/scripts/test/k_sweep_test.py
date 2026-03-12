"""
Quick k-sweep test script
Tests different k values (5-12) with RobustScaler + 85% PCA to find optimal clusters
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


def load_features():
    """Load raw feature data"""
    proj_path = Path(__file__).parents[2]  # Proj1/
    data_path = proj_path / "data" / "cluster_data"
    df = pd.read_csv(data_path / "pokemon_archetype_features.csv")
    
    # Select feature columns
    feature_cols = [col for col in df.columns 
                   if col.lower() not in ['name', 'type1', 'type2']]
    X = df[feature_cols].values.astype(float)
    
    return X, df


def test_k_values(X, k_values=range(5, 13)):
    """Test different k values with fixed RobustScaler + 85% PCA"""
    
    print("\n" + "="*80)
    print("K-SWEEP TEST: Finding optimal cluster count")
    print("="*80 + "\n")
    
    # Apply RobustScaler
    print("Applying RobustScaler...")
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
    
    print(f"\nCurrent pipeline uses k=8:")
    current_result = [r for r in results if r['k'] == 8][0]
    print(f"   Silhouette: {current_result['silhouette']:.4f}")
    print(f"   Improvement if switching to k={best_by_sil['k']}: {((best_by_sil['silhouette'] - current_result['silhouette'])/current_result['silhouette']*100):+.2f}%")
    
    # Check if improvement is worth it (>5% gain)
    improvement_pct = ((best_by_sil['silhouette'] - current_result['silhouette']) / current_result['silhouette']) * 100
    
    if improvement_pct > 5:
        print(f"\n✅ Significant improvement detected! Recommend switching to k={best_by_sil['k']}")
    elif improvement_pct > 0:
        print(f"\n⚠️  Minor improvement ({improvement_pct:.2f}%). k=8 is reasonable, but k={best_by_sil['k']} is slightly better.")
    else:
        print(f"\n✓ k=8 is optimal or near-optimal. No significant improvement found.")
    
    # Create visualization
    print("\nGenerating visualization...\n")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Plot 1: Silhouette vs k
    ax = axes[0]
    k_values_list = [r['k'] for r in results]
    silhouettes = [r['silhouette'] for r in results]
    
    ax.plot(k_values_list, silhouettes, marker='o', linewidth=2, markersize=8, color='steelblue')
    ax.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='Current (k=8)')
    ax.axvline(x=best_by_sil['k'], color='green', linestyle='--', alpha=0.7, label=f'Best (k={best_by_sil["k"]})')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Silhouette Score', fontsize=12)
    ax.set_title('Silhouette Score vs Cluster Count', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(k_values_list)
    
    # Plot 2: Davies-Bouldin Index vs k
    ax = axes[1]
    db_scores = [r['davies_bouldin'] for r in results]
    
    ax.plot(k_values_list, db_scores, marker='s', linewidth=2, markersize=8, color='coral')
    ax.axvline(x=8, color='red', linestyle='--', alpha=0.7, label='Current (k=8)')
    ax.axvline(x=best_by_sil['k'], color='green', linestyle='--', alpha=0.7, label=f'Best (k={best_by_sil["k"]})')
    ax.set_xlabel('Number of Clusters (k)', fontsize=12)
    ax.set_ylabel('Davies-Bouldin Index (lower is better)', fontsize=12)
    ax.set_title('Cluster Separation Quality vs Cluster Count', fontsize=13, fontweight='bold')
    ax.grid(True, alpha=0.3)
    ax.legend()
    ax.set_xticks(k_values_list)
    
    plt.tight_layout()
    
    test_results_path = Path(__file__).parents[2] / "reports" / "clustering_analysis" / "test_results"
    test_results_path.mkdir(parents=True, exist_ok=True)
    
    plt.savefig(test_results_path / 'k_sweep_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: k_sweep_comparison.png\n")
    
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
    summary_df.to_csv(test_results_path / 'k_sweep_results.csv', index=False)
    print("✅ Saved: k_sweep_results.csv\n")
    
    print("="*80 + "\n")
    
    return results, best_by_sil


if __name__ == '__main__':
    X, df = load_features()
    results, best = test_k_values(X, k_values=range(5, 13))
