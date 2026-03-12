"""
Clustering Optimization Test Script
Tests different normalization and dimensionality reduction strategies to find
the configuration that produces the most distinct clusters (fewest Generalist repeats).

Stores results in Proj1/reports/clustering_analysis/test_results/
"""

import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler, RobustScaler, MinMaxScaler, Normalizer
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
    
    # Select feature columns (skip Name, Type1, Type2, Legendaries columns)
    feature_cols = [col for col in df.columns 
                   if col.lower() not in ['name', 'type1', 'type2']]
    X = df[feature_cols].values
    
    # Ensure all values are numeric
    X = X.astype(float)
    
    return X, df


def test_scaler_and_pca_combo(X, scaler_name, scaler, pca_variance_threshold):
    """
    Test a specific scaler + PCA configuration
    
    Returns:
        dict with metrics, labels, and metadata
    """
    try:
        # Apply scaler
        X_scaled = scaler.fit_transform(X)
        
        # Apply PCA to achieve target variance threshold
        pca = PCA(n_components=0.99)  # Start with high variance
        X_pca = pca.fit_transform(X_scaled)
        
        # Find number of components for target variance
        cumsum = np.cumsum(pca.explained_variance_ratio_)
        n_components = np.argmax(cumsum >= pca_variance_threshold) + 1
        
        # Re-fit PCA with optimal components
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(X_scaled)
        
        variance_explained = np.sum(pca.explained_variance_ratio_)
        
        # Fit GMM with k=8
        gmm = GaussianMixture(n_components=8, random_state=42, n_init=10)
        labels = gmm.fit_predict(X_pca)
        
        # Calculate metrics
        silhouette = silhouette_score(X_pca, labels)
        davies_bouldin = davies_bouldin_score(X_pca, labels)
        bic = gmm.bic(X_pca)
        
        # Count unique archetypes (we'll use a placeholder for now)
        # In the real analysis, we'd need to call interpret_archetype()
        unique_archetypes_approximation = len(np.unique(labels))
        
        return {
            'scaler': scaler_name,
            'pca_variance_target': pca_variance_threshold,
            'n_components': n_components,
            'variance_explained': variance_explained,
            'silhouette_score': silhouette,
            'davies_bouldin': davies_bouldin,
            'bic': bic,
            'cluster_labels': labels,
            'pca_model': pca,
            'scaler_model': scaler,
            'gmm_model': gmm,
            'X_pca': X_pca,
            'unique_clusters': unique_archetypes_approximation
        }
    except Exception as e:
        print(f"  ❌ Failed: {str(e)}")
        return None


def analyze_cluster_distribution(labels):
    """Analyze cluster size distribution and balance"""
    unique, counts = np.unique(labels, return_counts=True)
    distribution = dict(zip(unique, counts))
    
    mean_size = np.mean(counts)
    std_size = np.std(counts)
    imbalance_ratio = np.max(counts) / np.min(counts)
    
    return {
        'distribution': distribution,
        'mean_size': mean_size,
        'std_size': std_size,
        'imbalance_ratio': imbalance_ratio
    }


def main():
    print("\n" + "="*80)
    print("CLUSTERING OPTIMIZATION TEST")
    print("Testing different normalization and PCA configurations")
    print("="*80 + "\n")
    
    # Load data
    print("Loading feature data...")
    X, df = load_features()
    print(f"✅ Loaded {X.shape[0]} Pokémon with {X.shape[1]} features\n")
    
    # Define test configurations
    scalers = {
        'StandardScaler': StandardScaler(),
        'RobustScaler': RobustScaler(),
        'MinMaxScaler': MinMaxScaler(),
        'Normalizer (L2)': Normalizer(norm='l2')
    }
    
    pca_variances = [0.85, 0.90, 0.91, 0.92, 0.95]
    
    results = []
    
    # Run tests
    print("Running tests (this may take a minute)...\n")
    total = len(scalers) * len(pca_variances)
    current = 0
    
    for scaler_name, scaler in scalers.items():
        for pca_var in pca_variances:
            current += 1
            print(f"[{current}/{total}] {scaler_name} + PCA@{pca_var*100:.0f}%...", end=" ")
            
            result = test_scaler_and_pca_combo(X, scaler_name, scaler, pca_var)
            
            if result is not None:
                # Get cluster distribution
                dist_analysis = analyze_cluster_distribution(result['cluster_labels'])
                result.update(dist_analysis)
                results.append(result)
                
                print(f"✅ Silhouette={result['silhouette_score']:.4f}")
            else:
                print("⚠️ Skipped")
    
    print(f"\n✅ Completed {len(results)} tests\n")
    
    # Sort results by silhouette score (higher is better)
    results.sort(key=lambda x: x['silhouette_score'], reverse=True)
    
    # Display results
    print("="*80)
    print("TOP 10 CONFIGURATIONS (by Silhouette Score)")
    print("="*80 + "\n")
    
    print(f"{'Rank':<5} {'Scaler':<20} {'PCA Var':<10} {'N-Comp':<7} "
          f"{'Silhouette':<12} {'Davies-Bouldin':<15} {'Imbalance':<10}")
    print("-" * 80)
    
    for i, result in enumerate(results[:10], 1):
        print(f"{i:<5} {result['scaler']:<20} {result['pca_variance_target']*100:>5.0f}% "
              f"{result['n_components']:>4d}   {result['silhouette_score']:>10.4f}   "
              f"{result['davies_bouldin']:>13.4f}   {result['imbalance_ratio']:>8.2f}x")
    
    print("\n" + "="*80)
    print("DETAILED ANALYSIS OF TOP 3 CONFIGURATIONS")
    print("="*80 + "\n")
    
    for i, result in enumerate(results[:3], 1):
        print(f"\n--- Configuration #{i} ---")
        print(f"Scaler: {result['scaler']}")
        print(f"PCA Variance Target: {result['pca_variance_target']*100:.0f}%")
        print(f"Components Used: {result['n_components']}")
        print(f"Actual Variance Explained: {result['variance_explained']*100:.2f}%")
        print(f"\nMetrics:")
        print(f"  Silhouette Score:    {result['silhouette_score']:.4f}")
        print(f"  Davies-Bouldin Index: {result['davies_bouldin']:.4f}")
        print(f"  BIC:                  {result['bic']:.2f}")
        print(f"\nCluster Distribution:")
        for cluster_id in sorted(result['distribution'].keys()):
            count = result['distribution'][cluster_id]
            pct = (count / len(df)) * 100
            print(f"  Cluster {cluster_id}: {count:>3d} Pokémon ({pct:>5.1f}%)")
        print(f"  Mean size: {result['mean_size']:.1f} ± {result['std_size']:.1f}")
        print(f"  Max/Min ratio (imbalance): {result['imbalance_ratio']:.2f}x")
    
    # Save detailed results to CSV
    result_path = Path(__file__).parents[2] / "reports" / "clustering_analysis" / "test_results"
    result_path.mkdir(parents=True, exist_ok=True)
    
    # Create summary dataframe
    summary_data = []
    for result in results:
        summary_data.append({
            'Scaler': result['scaler'],
            'PCA_Variance_Target': result['pca_variance_target'],
            'N_Components': result['n_components'],
            'Variance_Explained': result['variance_explained'],
            'Silhouette_Score': result['silhouette_score'],
            'Davies_Bouldin': result['davies_bouldin'],
            'BIC': result['bic'],
            'Mean_Cluster_Size': result['mean_size'],
            'Cluster_Imbalance_Ratio': result['imbalance_ratio']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_df.to_csv(result_path / "optimization_test_results.csv", index=False)
    print(f"\n✅ Detailed results saved to: optimization_test_results.csv\n")
    
    # Create comparison visualization
    print("Generating comparison plots...\n")
    
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Clustering Optimization: Scaler & PCA Configuration Comparison', 
                 fontsize=14, fontweight='bold')
    
    # Plot 1: Silhouette Score by Scaler
    ax = axes[0, 0]
    for scaler_name in scalers.keys():
        scaler_results = [r for r in results if r['scaler'] == scaler_name]
        variances = [r['pca_variance_target']*100 for r in scaler_results]
        silhouettes = [r['silhouette_score'] for r in scaler_results]
        ax.plot(variances, silhouettes, marker='o', label=scaler_name, linewidth=2)
    ax.set_xlabel('PCA Variance Target (%)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Silhouette Score vs PCA Variance')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Davies-Bouldin Index by Scaler
    ax = axes[0, 1]
    for scaler_name in scalers.keys():
        scaler_results = [r for r in results if r['scaler'] == scaler_name]
        variances = [r['pca_variance_target']*100 for r in scaler_results]
        db_scores = [r['davies_bouldin'] for r in scaler_results]
        ax.plot(variances, db_scores, marker='s', label=scaler_name, linewidth=2)
    ax.set_xlabel('PCA Variance Target (%)')
    ax.set_ylabel('Davies-Bouldin Index (lower is better)')
    ax.set_title('Cluster Separation Quality')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Components Required by Scaler
    ax = axes[1, 0]
    for scaler_name in scalers.keys():
        scaler_results = [r for r in results if r['scaler'] == scaler_name]
        variances = [r['pca_variance_target']*100 for r in scaler_results]
        n_comps = [r['n_components'] for r in scaler_results]
        ax.plot(variances, n_comps, marker='^', label=scaler_name, linewidth=2)
    ax.set_xlabel('PCA Variance Target (%)')
    ax.set_ylabel('Number of PCA Components')
    ax.set_title('Dimensionality vs Variance Retention')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Cluster Imbalance Ratio
    ax = axes[1, 1]
    imbalance_scores = [r['imbalance_ratio'] for r in results]
    silhouette_scores = [r['silhouette_score'] for r in results]
    scaler_colors = {scaler: ['C0', 'C1', 'C2', 'C3'][i] 
                     for i, scaler in enumerate(scalers.keys())}
    colors = [scaler_colors[r['scaler']] for r in results]
    ax.scatter(imbalance_scores, silhouette_scores, c=colors, s=100, alpha=0.6)
    
    # Add legend
    for scaler_name, color in scaler_colors.items():
        ax.scatter([], [], c=color, s=100, label=scaler_name)
    
    ax.set_xlabel('Cluster Imbalance Ratio (Max/Min)')
    ax.set_ylabel('Silhouette Score')
    ax.set_title('Balance vs Quality Trade-off')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(result_path / 'optimization_comparison.png', dpi=150, bbox_inches='tight')
    print("✅ Saved: optimization_comparison.png\n")
    
    # Recommendation
    print("="*80)
    print("RECOMMENDATION")
    print("="*80 + "\n")
    
    best = results[0]
    print(f"🏆 Best Configuration:")
    print(f"   Scaler: {best['scaler']}")
    print(f"   PCA Variance: {best['pca_variance_target']*100:.0f}% → {best['n_components']} components")
    print(f"   Silhouette Score: {best['silhouette_score']:.4f}")
    print(f"   Davies-Bouldin: {best['davies_bouldin']:.4f} (lower is better)")
    print(f"   Cluster Imbalance: {best['imbalance_ratio']:.2f}x")
    print(f"\nNext Steps:")
    print(f"1. Update clustering_pipeline.py to use {best['scaler']} with PCA@{best['pca_variance_target']*100:.0f}%")
    print(f"2. Re-run pipeline to generate new cluster assignments")
    print(f"3. Run cluster_analyzer.py to see if archetypes are more distinct")
    print(f"4. Compare to current results (should have fewer Generalist repeats)")
    
    print("\n" + "="*80 + "\n")


if __name__ == '__main__':
    main()
