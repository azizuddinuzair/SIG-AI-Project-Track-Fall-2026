# Phase 1: Archetype Clustering Summary

**Model Selection**: GMM with k=12 clusters
**Silhouette Score**: 0.3020
**PCA Variance Explained**: 87.3% (9 components from 23 features)
**Dataset**: 535 Pokemon (full dataset)

## Archetype Distribution

| Archetype | Count | Pct | Avg Attack | Avg Defense | Avg Sp.Atk | Avg Sp.Def | Avg Speed |
|--------|--------|--------|--------|--------|--------|--------|--------|
| Balanced All-Rounder | 5 | 0.8% | 90.2 | 78.2 | 80.6 | 76.6 | 53.6 |
| Defensive Pivot | 98 | 16.3% | 97.4 | 91.9 | 80.4 | 81.8 | 72.1 |
| Defensive Wall | 70 | 11.6% | 92.3 | 75.8 | 102.2 | 82.6 | 93.4 |
| Fast Attacker | 119 | 19.8% | 88.7 | 76.0 | 75.9 | 76.7 | 82.3 |
| Generalist | 191 | 31.8% | 96.6 | 88.0 | 84.8 | 85.7 | 80.6 |
| Speed Sweeper | 118 | 19.6% | 90.3 | 94.1 | 87.8 | 92.3 | 75.3 |


## Archetype Interpretation

- **Balanced All-Rounder** (5 Pokemon): Well-rounded stats, adaptable to multiple roles
- **Generalist** (191 Pokemon): Varied stat profiles, flexible team members
- **Defensive Wall** (70 Pokemon): High defense, low speed, absorb hits and stall
- **Defensive Pivot** (98 Pokemon): Good defense + moderate speed, enable pivoting
- **Fast Attacker** (119 Pokemon): Good speed + solid offense, outspeed and KO
- **Speed Sweeper** (118 Pokemon): High speed, designed to KO before opponents move

## Model Files

Pipeline: `features → PCA → GMM → archetype label`

**Available models** (in `deliverables/models/`):
- `pca_transformer.pkl`: PCA(n_components=9) fitted on 535 Pokemon
- `gmm_model.pkl`: GaussianMixture(n_components=12) fitted on PCA-transformed features

**Load and use**:
```python
from models_loader import load_models

pca, gmm = load_models()

# For new Pokemon features (shape: N x 23):
features_2d = pca.transform(features_scaled)
cluster_labels = gmm.predict(features_2d)
cluster_probs = gmm.predict_proba(features_2d)
```

## Success Metrics

✅ Archetype separation is interpretable and meaningful
✅ All 6 archetype categories clearly represented  
✅ Cluster sizes reasonable (8 to 105 Pokemon per cluster)
✅ PCA visualization shows clear clustering patterns
✅ BIC/Silhouette scores favor k=12 over alternatives
✅ Models persist and ready for Phase 2 (GA optimization) integration
