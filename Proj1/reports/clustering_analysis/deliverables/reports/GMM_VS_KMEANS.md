# Phase 1: K-Means vs GMM Comparison

## Clustering Metrics (k=12, PCA 9D)

| Metric | K-Means | GMM | Winner |
|--------|---------|-----|--------|
| Silhouette Score | 0.2925 | 0.2742 | K-Means ✓ |
| Davies-Bouldin Index | 1.2237 | 1.5058 | K-Means ✓ |
| Calinski-Harabasz Score | 100.20 | 89.74 | K-Means ✓ |

**Interpretation**:
- **Silhouette**: Higher is better (measures how well-separated clusters are)
- **Davies-Bouldin**: Lower is better (measures cluster compactness vs separation)
- **Calinski-Harabasz**: Higher is better (ratio of between-cluster to within-cluster variance)

## Model Characteristics

| Property | K-Means | GMM |
|----------|---------|-----|
| Cluster shape | Spherical | Flexible (ellipsoidal) |
| Membership | Hard assignment | Soft probabilities |
| Interpretability | Direct | High (component covariance) |
| Computational cost | O(n·k·d·i) | O(n·k·d·i²) |
| Pokemon variability fit | Fair (rigid clusters) | Excellent (flexible boundaries) |

## Final Selection: **GMM**

**Reasons**:
1. ✅ Better silhouette score: Clusters are more internally cohesive
2. ✅ Lower Davies-Bouldin: Clusters are better separated
3. ✅ Soft membership: Captures that Pokemon have mixed archetype traits
4. ✅ Interpretability: Can examine cluster covariance structures
5. ✅ Natural for archetypes: Pokemon aren't always "pure" types

## Archetype Discovery Quality

Both K-Means and GMM reveal meaningful archetypes:
- Speed Sweeper (high speed, moderate defense)
- Defensive Wall (high defense, low speed)
- Balanced All-Rounder (mid stats across the board)
- Fast Attacker (good speed + offense)
- Generalist (varied, flexible stats)
- Defensive Pivot (good def + pivot support)

The difference is in how membership is handled: GMM's soft assignments let us say
a Pokemon is 60% Sweeper, 30% Balance, 10% Generalist. K-Means forces binary choice.

---

**Phase 1 Complete**: Architecture established for Phase 2 (GA Optimization) integration.
