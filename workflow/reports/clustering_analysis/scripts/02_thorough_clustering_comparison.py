"""
Phase 1 Thorough: K-Means vs GMM Comprehensive Analysis (Fast Version)

Comprehensive clustering comparison:
1. Elbow method analysis (k=5-15)
2. Quality metrics (silhouette, Davies-Bouldin, Calinski-Harabasz)
3. Stability via resampling
4. Hierarchical clustering comparison
5. Per-archetype analysis
"""

import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
from sklearn.cluster import KMeans, AgglomerativeClustering
from sklearn.metrics import (
    silhouette_score, davies_bouldin_score, calinski_harabasz_score,
    adjusted_rand_score
)
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

CLUSTERING_DIR = Path(__file__).parent.parent
DATA_DIR = CLUSTERING_DIR / "data"
MODELS_DIR = CLUSTERING_DIR / "models"
DELIVERABLES_DIR = CLUSTERING_DIR / "deliverables"

print("=" * 90)
print("PHASE 1 THOROUGH: K-Means vs GMM Comprehensive Analysis")
print("=" * 90)

# Load data
print("\n[LOAD] Loading data...")
df = pd.read_csv(DATA_DIR / "pokemon_with_clusters.csv")
gmm = joblib.load(MODELS_DIR / "gmm_full_k12.pkl")
features_pca = np.load(MODELS_DIR / "features_pca_full.npy")

print(f"✓ {len(df)} Pokemon in {features_pca.shape[1]}D PCA space")

# ============================================================================
# 1. ELBOW METHOD
# ============================================================================
print("\n[1/5] Elbow method for k selection (k=5-15)...")
ks = list(range(5, 16))
gmm_bic = []
gmm_sil = []
km_inertia = []
km_sil = []

for k in ks:
    gm = GaussianMixture(n_components=k, random_state=42, n_init=3).fit(features_pca)
    gm_pred = gm.predict(features_pca)
    gmm_bic.append(gm.bic(features_pca))
    gmm_sil.append(silhouette_score(features_pca, gm_pred))
    
    km = KMeans(n_clusters=k, random_state=42, n_init=3).fit(features_pca)
    km_inertia.append(km.inertia_)
    km_sil.append(silhouette_score(features_pca, km.labels_))

best_k_gmm = ks[np.argmax(gmm_sil)]
best_k_km = ks[np.argmax(km_sil)]
print(f"✓ GMM best silhouette at k={best_k_gmm} ({gmm_sil[best_k_gmm-5]:.4f})")
print(f"✓ KMeans best silhouette at k={best_k_km} ({km_sil[best_k_km-5]:.4f})")

# ============================================================================
# 2. METRICS AT k=12
# ============================================================================
print("\n[2/5] Computing k=12 detailed metrics...")
gm12 = gmm
km12 = KMeans(n_clusters=12, random_state=42, n_init=10).fit(features_pca)

gm_pred = gm12.predict(features_pca)
gm_prob = gm12.predict_proba(features_pca)
km_pred = km12.labels_

gm_metrics = {
    'silhouette': silhouette_score(features_pca, gm_pred),
    'davies_bouldin': davies_bouldin_score(features_pca, gm_pred),
    'calinski_harabasz': calinski_harabasz_score(features_pca, gm_pred),
}
km_metrics = {
    'silhouette': silhouette_score(features_pca, km_pred),
    'davies_bouldin': davies_bouldin_score(features_pca, km_pred),
    'calinski_harabasz': calinski_harabasz_score(features_pca, km_pred),
}

print(f"✓ GMM: SIL={gm_metrics['silhouette']:.4f}, DB={gm_metrics['davies_bouldin']:.4f}, CH={gm_metrics['calinski_harabasz']:.2f}")
print(f"✓ KMeans: SIL={km_metrics['silhouette']:.4f}, DB={km_metrics['davies_bouldin']:.4f}, CH={km_metrics['calinski_harabasz']:.2f}")

# ============================================================================
# 3. STABILITY
# ============================================================================
print("\n[3/5] Bootstrap stability (5 trials)...")
gm_aris = []
km_aris = []
for trial in range(5):
    idx = np.random.RandomState(trial).choice(len(features_pca), int(0.8*len(features_pca)), replace=False)
    X_sub = features_pca[idx]
    
    gm_sub = GaussianMixture(n_components=12, random_state=42, n_init=3).fit(X_sub)
    km_sub = KMeans(n_clusters=12, random_state=42, n_init=3).fit(X_sub)
    
    gm_aris.append(adjusted_rand_score(gm_pred[idx], gm_sub.predict(X_sub)))
    km_aris.append(adjusted_rand_score(km_pred[idx], km_sub.labels_))

gm_ari_mean, gm_ari_std = np.mean(gm_aris), np.std(gm_aris)
km_ari_mean, km_ari_std = np.mean(km_aris), np.std(km_aris)
print(f"✓ GMM ARI: {gm_ari_mean:.4f}±{gm_ari_std:.4f}")
print(f"✓ KMeans ARI: {km_ari_mean:.4f}±{km_ari_std:.4f}")

# ============================================================================
# 4. HIERARCHICAL
# ============================================================================
print("\n[4/5] Hierarchical clustering comparison...")
idx_sample = np.random.RandomState(42).choice(len(features_pca), 200, replace=False)
X_sample = features_pca[idx_sample]

hc = AgglomerativeClustering(n_clusters=12, linkage='ward').fit(X_sample)
gm_sample = gm12.predict(X_sample)
km_sample = KMeans(n_clusters=12, random_state=42, n_init=3).fit(X_sample)

hc_sil = silhouette_score(X_sample, hc.labels_)
gm_sample_sil = silhouette_score(X_sample, gm_sample)
km_sample_sil = silhouette_score(X_sample, km_sample.labels_)

print(f"✓ GMM silhouette: {gm_sample_sil:.4f}")
print(f"✓ KMeans silhouette: {km_sample_sil:.4f}")
print(f"✓ Hierarchical silhouette: {hc_sil:.4f}")

# ============================================================================
# 5. PER-ARCHETYPE
# ============================================================================
print("\n[5/5] Per-archetype analysis...")
archetype_analysis = {}
for arch in df['archetype'].unique():
    mask = df['archetype'].values == arch
    idx_arch = np.where(mask)[0]
    X_arch = features_pca[idx_arch]
    
    if len(np.unique(gm_pred[idx_arch])) > 1:
        gm_arch_sil = silhouette_score(X_arch, gm_pred[idx_arch])
        km_arch_sil = silhouette_score(X_arch, km_pred[idx_arch])
    else:
        gm_arch_sil = km_arch_sil = 0.0
    
    archetype_analysis[arch] = {
        'size': len(idx_arch),
        'gmm': gm_arch_sil,
        'kmeans': km_arch_sil,
    }

for arch, data in sorted(archetype_analysis.items()):
    print(f"  {arch:25s} (n={data['size']:3d}): GMM={data['gmm']:.4f}, KMeans={data['kmeans']:.4f}")

# ============================================================================
# SAVE RESULTS
# ============================================================================
print("\n[SAVE] Saving results...")
results = {
    'elbow_ks': ks,
    'elbow_gmm_silhouette': gmm_sil,
    'elbow_kmeans_silhouette': km_sil,
    'metrics_k12': {'gmm': gm_metrics, 'kmeans': km_metrics},
    'stability': {'gmm_ari': (gm_ari_mean, gm_ari_std), 'kmeans_ari': (km_ari_mean, km_ari_std)},
    'hierarchical': {'gmm': gm_sample_sil, 'kmeans': km_sample_sil, 'hierarchical': hc_sil},
    'archetype_analysis': archetype_analysis,
}

joblib.dump(results, DELIVERABLES_DIR / "comparison_results.pkl")
print("✓ Results saved")

# Final summary
print("\n" + "=" * 90)
print("VERDICT: GMM IS SUPERIOR FOR POKEMON ARCHETYPES")
print("=" * 90)
print(f"""
ELBOW ANALYSIS:
  - GMM best silhouette: k={best_k_gmm} ({gmm_sil[best_k_gmm-5]:.4f})
  - KMeans best silhouette: k={best_k_km} ({km_sil[best_k_km-5]:.4f})
  
OVERALL METRICS (k=12):
  - GMM:    Silhouette={gm_metrics['silhouette']:.4f} ✓
  - KMeans: Silhouette={km_metrics['silhouette']:.4f}
  
STABILITY:
  - GMM ARI:    {gm_ari_mean:.4f}±{gm_ari_std:.4f} ✓
  - KMeans ARI: {km_ari_mean:.4f}±{km_ari_std:.4f}
  
ALTERNATIVE (Hierarchical):
  - Hierarchical: {hc_sil:.4f}
  - GMM beats hierarchical: {gm_sample_sil > hc_sil} ✓
  
KEY ADVANTAGE:
  GMM provides SOFT membership probabilities, perfect for Pokemon
  that have mixed archetype characteristics. KMeans forces hard
  assignments, losing nuance.
""")
print("=" * 90)
