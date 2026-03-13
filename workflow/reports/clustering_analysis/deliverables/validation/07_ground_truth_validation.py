"""
Phase 6: Ground-Truth Archetype Validation

Validates learned archetypes against competitive Pokémon roles.
Uses 18 clear examples from Smogon OU tier.

Metrics: Accuracy, Confusion Matrix, Cohen's Kappa
"""

import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
import json
from pathlib import Path


# ============================================================================
# GROUND TRUTH ANNOTATIONS
# ============================================================================
# 18 Pokémon with clear competitive roles
# Format: pokemon_name → ground_truth_role

GROUND_TRUTH = {
    # Sweepers (5)
    "alakazam": "Sweeper",
    "dragapult": "Sweeper",
    "cloyster": "Sweeper",
    "volcarona": "Sweeper",
    "gyarados": "Sweeper",
    
    # Defensive Walls (3)
    "blissey": "Wall",
    "toxapex": "Wall",
    "skarmory": "Wall",
    
    # Pivots (3)
    "corviknight": "Pivot",
    "rotom": "Pivot",  # Base rotom (forms weren't in SKIPPED_POKEMON)
    "landorus-therian": "Pivot",  # Using competitive Therian form
    
    # Mixed / Balanced (3)
    "garchomp": "Mixed",
    "greninja": "Mixed",
    "iron-valiant": "Mixed",
    
    # Tanks (2)
    "ferrothorn": "Tank",
    "heatran": "Tank",
}

# Mapping clusters to ground truth roles
# GMM 6 archetypes → ground truth 5 roles
CLUSTER_TO_ROLE = {
    # Our archetypes: Generalist, Balanced All-Rounder, Fast Attacker, 
    #                 Defensive Pivot, Defensive Wall, Speed Sweeper
    "Generalist": ["Mixed", "Tank"],
    "Balanced All-Rounder": ["Mixed", "Tank"],
    "Fast Attacker": ["Sweeper"],
    "Defensive Pivot": ["Pivot", "Tank"],
    "Defensive Wall": ["Wall", "Tank"],
    "Speed Sweeper": ["Sweeper"],
}

# Reverse mapping for predictions
ROLE_PRED_MAP = {
    "Sweeper": ["Speed Sweeper", "Fast Attacker"],
    "Wall": ["Defensive Wall"],
    "Pivot": ["Defensive Pivot"],
    "Mixed": ["Balanced All-Rounder", "Generalist"],
    "Tank": ["Defensive Wall", "Balanced All-Rounder", "Generalist"],
}


# ============================================================================
# LOAD MODELS & DATA
# ============================================================================

def load_models_and_data():
    """Load PCA, GMM, and Pokémon data."""
    models_dir = Path(__file__).parent.parent / "models"
    data_dir = Path(__file__).parent.parent.parent / "data"
    
    # Load models
    pca = joblib.load(models_dir / "pca_transformer.pkl")
    gmm = joblib.load(models_dir / "gmm_model.pkl")
    
    # Load Pokémon data
    pokemon_df = pd.read_csv(data_dir / "pokemon_with_clusters.csv")
    
    return pca, gmm, pokemon_df


# ============================================================================
# ARCHETYPE CLUSTER MAPPING
# ============================================================================

def get_archetype_names():
    """Map 12 clusters to 6 archetypes (from Phase 1)."""
    return {
        0: "Generalist",
        1: "Balanced All-Rounder",
        2: "Generalist",
        3: "Generalist",
        4: "Fast Attacker",
        5: "Defensive Wall",
        6: "Speed Sweeper",
        7: "Balanced All-Rounder",
        8: "Speed Sweeper",
        9: "Defensive Wall",
        10: "Defensive Pivot",
        11: "Balanced All-Rounder",
    }


# ============================================================================
# VALIDATION
# ============================================================================

def predict_role(archetype_name):
    """
    Map archetype name to predicted role.
    Uses many-to-many mapping (archetype can map to multiple roles).
    For hard label, pick most likely.
    """
    role_mapping = {
        "Speed Sweeper": "Sweeper",
        "Fast Attacker": "Sweeper",
        "Defensive Wall": "Wall",
        "Defensive Pivot": "Pivot",
        "Balanced All-Rounder": "Mixed",
        "Generalist": "Mixed",
    }
    return role_mapping.get(archetype_name, "Mixed")


def validate_archetypes():
    """Run full validation."""
    print("=" * 80)
    print("PHASE 6: GROUND-TRUTH ARCHETYPE VALIDATION")
    print("=" * 80)
    print()
    
    # Load data
    print("Loading models and data...")
    pca, gmm, pokemon_df = load_models_and_data()
    archetype_map = get_archetype_names()
    
    # Extract test set
    test_pokemon_names = list(GROUND_TRUTH.keys())
    print(f"Test set size: {len(test_pokemon_names)} Pokémon")
    print()
    
    # Get predictions
    predictions = []
    soft_probs = []
    found_count = 0
    
    print("-" * 80)
    print(f"{'Pokémon':<20} {'Ground Truth':<12} {'Predicted':<20} {'Confidence':<8}")
    print("-" * 80)
    
    # Load scaler for feature scaling
    from sklearn.preprocessing import RobustScaler
    
    # Extract all engineered features (the same ones PCA was trained on)
    # 22 features total: 4 derived stats + 18 type defense vectors
    feature_cols = [
        'offensive_index', 'defensive_index', 'speed_percentile', 'physical_special_bias',
        'type_defense_normal', 'type_defense_fire', 'type_defense_water', 'type_defense_grass',
        'type_defense_electric', 'type_defense_ice', 'type_defense_fighting', 'type_defense_poison',
        'type_defense_ground', 'type_defense_flying', 'type_defense_psychic', 'type_defense_bug',
        'type_defense_rock', 'type_defense_ghost', 'type_defense_dragon', 'type_defense_dark',
        'type_defense_steel', 'type_defense_fairy'
    ]
    
    scaler = RobustScaler()
    all_features = pokemon_df[feature_cols].values
    scaler.fit(all_features)
    
    for poke_name in test_pokemon_names:
        # Find in dataset
        poke_row = pokemon_df[pokemon_df['name'].str.lower() == poke_name.lower()]
        
        if len(poke_row) == 0:
            print(f"{poke_name:<20} (NOT FOUND)")
            continue
        
        found_count += 1
        
        # Get features
        features = poke_row.iloc[0][feature_cols].values
        features = features.reshape(1, -1)
        
        # Scale and transform
        features_scaled = scaler.transform(features)
        features_pca = pca.transform(features_scaled)
        
        # Get cluster prediction
        cluster_id = gmm.predict(features_pca)[0]
        archetype = archetype_map[cluster_id]
        predicted_role = predict_role(archetype)
        
        # Get soft probabilities
        probs = gmm.predict_proba(features_pca)[0]
        confidence = np.max(probs)
        
        ground_truth = GROUND_TRUTH[poke_name]
        
        print(f"{poke_name:<20} {ground_truth:<12} {archetype:<20} {confidence:.2f}")
        
        predictions.append({
            'poke': poke_name,
            'ground_truth': ground_truth,
            'predicted_role': predicted_role,
            'predicted_archetype': archetype,
            'confidence': confidence,
            'cluster_id': cluster_id,
            'soft_probs': probs,
        })
        soft_probs.append(probs)
    
    print()
    print(f"Successfully matched: {found_count}/{len(test_pokemon_names)}")
    print()
    
    if found_count < len(test_pokemon_names):
        print("⚠️  Some Pokémon not found in dataset. Check names.")
        print()
    
    # ========================================================================
    # METRICS
    # ========================================================================
    
    ground_truth_labels = [p['ground_truth'] for p in predictions]
    predicted_labels = [p['predicted_role'] for p in predictions]
    
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    kappa = cohen_kappa_score(ground_truth_labels, predicted_labels)
    conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels)
    
    print("=" * 80)
    print("RESULTS")
    print("=" * 80)
    print()
    print(f"Accuracy:      {accuracy:.2%}")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print()
    
    # Interpretation
    if accuracy >= 0.70:
        acc_rating = "✓ STRONG (≥70%)"
    elif accuracy >= 0.60:
        acc_rating = "○ MODERATE (60-70%)"
    else:
        acc_rating = "✗ WEAK (<60%)"
    
    if kappa >= 0.75:
        kappa_rating = "✓ EXCELLENT (>0.75)"
    elif kappa >= 0.60:
        kappa_rating = "✓ GOOD (0.60-0.75)"
    elif kappa >= 0.40:
        kappa_rating = "○ MODERATE (0.40-0.60)"
    else:
        kappa_rating = "✗ POOR (<0.40)"
    
    print(f"Accuracy Rating: {acc_rating}")
    print(f"Kappa Rating:    {kappa_rating}")
    print()
    
    # Confusion matrix
    classes = sorted(set(ground_truth_labels + predicted_labels))
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print()
    print(f"{'Actual \\ Pred':<15}", end="")
    for cls in classes:
        print(f"{cls:<12}", end="")
    print()
    
    for i, actual_cls in enumerate(classes):
        print(f"{actual_cls:<15}", end="")
        for j, pred_cls in enumerate(classes):
            count = np.sum((np.array(ground_truth_labels) == actual_cls) & 
                          (np.array(predicted_labels) == pred_cls))
            print(f"{count:<12}", end="")
        print()
    
    print()
    
    # ========================================================================
    # SOFT PROBABILITIES (Key Insight)
    # ========================================================================
    
    print("=" * 80)
    print("SOFT PROBABILITIES (GMM Confidence Across Archetypes)")
    print("=" * 80)
    print()
    print("This shows how confident the model is about each Pokémon's role.")
    print("High values for multiple archetypes indicate hybrid roles.")
    print()
    
    archetype_names = list(archetype_map.values())
    archetype_names = sorted(list(set(archetype_names)))
    
    for pred in predictions:
        poke = pred['poke']
        probs = pred['soft_probs']
        
        print(f"{poke}:")
        for i, cluster_id in enumerate(sorted(archetype_map.keys())):
            archetype = archetype_map[cluster_id]
            prob = probs[cluster_id]
            if prob > 0.05:  # Only show >5% probability
                print(f"  {archetype:<25} {prob:.2%}")
        print()
    
    # ========================================================================
    # SAVE RESULTS
    # ========================================================================
    
    results = {
        'accuracy': float(accuracy),
        'kappa': float(kappa),
        'test_count': found_count,
        'predictions': [
            {
                'pokemon': p['poke'],
                'ground_truth': p['ground_truth'],
                'predicted_role': p['predicted_role'],
                'predicted_archetype': p['predicted_archetype'],
                'confidence': float(p['confidence']),
            }
            for p in predictions
        ],
        'interpretation': {
            'accuracy_rating': acc_rating,
            'kappa_rating': kappa_rating,
        }
    }
    
    output_dir = Path(__file__).parent / "results"
    output_dir.mkdir(exist_ok=True)
    
    with open(output_dir / "validation_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("=" * 80)
    print(f"Results saved to: {output_dir / 'validation_results.json'}")
    print("=" * 80)
    
    return results


if __name__ == "__main__":
    results = validate_archetypes()
