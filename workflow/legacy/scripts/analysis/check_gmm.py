import joblib
import numpy as np

gmm = joblib.load('Proj1/reports/clustering_analysis/models/gmm_full_k12.pkl')
print('GMM n_components:', gmm.n_components)
print('Converged:', gmm.converged_)
print('n_iter:', gmm.n_iter_)

weights = gmm.weights_
print('\nCluster weights (should be ~0.083 each for balanced):')
for i, w in enumerate(weights):
    print(f'  Cluster {i:2d}: {w:.4f} ({w*601:.1f} expected Pokemon)')

print(f'\nClusters with >1% weight: {(weights > 0.01).sum()}')
print(f'Clusters with >5% weight: {(weights > 0.05).sum()}')
