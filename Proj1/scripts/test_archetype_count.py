"""
Test if fewer archetypes improve Phase 6 accuracy
(maybe 6 is too granular, 3 or 4 would generalize better)
"""

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "models"))

import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

# Load pokémon data
pokemon_csv = Path(__file__).parents[1] / "reports" / "clustering_analysis" / "data" / "pokemon_with_clusters.csv"
pokemon_df = pd.read_csv(pokemon_csv)

print("\nTesting different archetype counts...")
print("="*60)

# Phase 6 test set
test_pokemon = ["alakazam", "dragapult", "cloyster", "volcarona", "gyarados", 
                "blissey", "toxapex", "umbreon", "salamence", "garchomp",
                "venusaur", "lapras", "machamp", "arcanine", "rotom_wash", "ferrothorn"]

test_df = pokemon_df[pokemon_df['name'].isin(test_pokemon)]

# Features for archetype clustering
feature_cols = ['hp', 'attack', 'defense', 'special-defense', 'special-attack', 'speed',
                'offensive_index', 'defensive_index', 'speed_percentile']

for n_archetypes in [2, 3, 4, 5, 6, 8]:
    print(f"\nTesting with {n_archetypes} archetypes...")
    
    # Cluster all pokemon
    X = pokemon_df[feature_cols].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    gmm = GaussianMixture(n_components=n_archetypes, random_state=42)
    gmm.fit(X_scaled)
    
    # Get test set archetypes
    test_X = test_df[feature_cols].values
    test_X_scaled = scaler.transform(test_X)
    
    test_probs = gmm.predict_proba(test_X_scaled)
    test_pred_archetypes = np.argmax(test_probs, axis=1)
    
    # Check entropy of predictions
    pred_entropy = np.mean([-np.sum(p[p > 0] * np.log(p[p > 0])) for p in test_probs])
    
    print(f"  Average entropy: {pred_entropy:.4f} (lower = more confident)")
    print(f"  Predictions: {test_pred_archetypes[:5]}")
    
print("\n" + "="*60)
print("Best archetype count should have moderate entropy (not too confident, not too uncertain)")
