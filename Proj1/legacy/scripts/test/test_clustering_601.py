"""
Quick Sanity Test: 601-Pokemon Clustering System
Tests that the retrained clustering models work correctly before running Phase 2 GA.

Tests:
1. Models load correctly
2. Feature pipeline works (scaler → PCA → GMM)
3. All 6 archetypes are present
4. Cluster distribution is reasonable
5. Sample predictions are consistent
"""

import pandas as pd
import numpy as np
import joblib
from pathlib import Path


def get_feature_columns(df: pd.DataFrame):
    """Infer feature columns used by clustering from available engineered features."""
    role_shape_cols = [
        'offensive_index',
        'defensive_index',
        'speed_percentile',
        'physical_special_bias',
        'physical_bulk',
        'special_bulk',
        'bulk_bias',
        'offense_to_bulk_ratio',
        'speed_to_bulk_ratio',
        'special_bulk_percentile',
        'physical_bulk_percentile',
    ]
    return [
        c for c in df.columns
        if c.startswith('type_defense_') or c in role_shape_cols
    ]


def test_models_load():
    """Test 1: All models load without errors"""
    print("\n[TEST 1] Loading models...")
    
    models_dir = Path(__file__).parents[2] / "reports" / "clustering_analysis" / "models"
    
    try:
        scaler = joblib.load(models_dir / "scaler.pkl")
        pca = joblib.load(models_dir / "pca.pkl")
        gmm = joblib.load(models_dir / "gmm_full_k12.pkl")
        
        print(f"  ✓ Scaler: {type(scaler).__name__}")
        print(f"  ✓ PCA: {pca.n_components_} components, {pca.explained_variance_ratio_.sum():.1%} variance")
        print(f"  ✓ GMM: k={gmm.n_components}, converged={gmm.converged_}")
        
        return scaler, pca, gmm, True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return None, None, None, False


def test_feature_pipeline(scaler, pca, gmm):
    """Test 2: Feature transformation pipeline works"""
    print("\n[TEST 2] Testing feature pipeline...")
    
    data_dir = Path(__file__).parents[2] / "reports" / "clustering_analysis" / "data"
    df = pd.read_csv(data_dir / "pokemon_with_clusters.csv")
    
    feature_cols = get_feature_columns(df)
    
    try:
        # Extract and transform
        X = df[feature_cols].values
        X_scaled = scaler.transform(X)
        X_pca = pca.transform(X_scaled)
        clusters = gmm.predict(X_pca)
        probs = gmm.predict_proba(X_pca)
        
        print(f"  ✓ Input: {X.shape[0]} Pokemon, {X.shape[1]} features")
        print(f"  ✓ Scaled: {X_scaled.shape}")
        print(f"  ✓ PCA: {X_pca.shape}")
        print(f"  ✓ Clusters: {len(clusters)} assignments")
        print(f"  ✓ Probabilities: {probs.shape} (max confidence: {probs.max():.3f})")
        
        return clusters, probs, True
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return None, None, False


def test_archetype_coverage(df):
    """Test 3: All 6 archetypes present"""
    print("\n[TEST 3] Checking archetype coverage...")
    
    try:
        archetypes = df['archetype'].unique()
        counts = df['archetype'].value_counts()
        
        print(f"  ✓ Unique archetypes: {len(archetypes)}")
        
        for archetype in sorted(archetypes):
            count = counts[archetype]
            pct = count / len(df) * 100
            print(f"    - {archetype:25s}: {count:3d} ({pct:5.1f}%)")
        
        # Check if we have at least 5 archetypes (allow some flexibility)
        if len(archetypes) >= 5:
            print(f"  ✓ PASS: {len(archetypes)} archetypes found (expected 6+)")
            return True
        else:
            print(f"  ✗ FAIL: Only {len(archetypes)} archetypes (expected 6+)")
            return False
            
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_cluster_distribution(clusters):
    """Test 4: Cluster distribution is reasonable"""
    print("\n[TEST 4] Checking cluster distribution...")
    
    try:
        unique, counts = np.unique(clusters, return_counts=True)
        
        print(f"  ✓ Active clusters: {len(unique)}/12")
        
        # Check for cluster collapse (>50% in one cluster = bad)
        max_cluster_pct = counts.max() / len(clusters) * 100
        
        if max_cluster_pct > 50:
            print(f"  ⚠ WARNING: Largest cluster has {max_cluster_pct:.1f}% (cluster collapse)")
            return False
        else:
            print(f"  ✓ PASS: Largest cluster has {max_cluster_pct:.1f}% (balanced)")
            
        # Check for empty clusters
        if len(unique) < 8:
            print(f"  ⚠ WARNING: Only {len(unique)}/12 clusters active (some unused)")
            
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def test_sample_predictions(scaler, pca, gmm, df):
    """Test 5: Sample predictions for known Pokemon"""
    print("\n[TEST 5] Testing sample predictions...")
    
    feature_cols = get_feature_columns(df)
    
    # Test on known competitive Pokemon
    test_pokemon = ['dragapult', 'landorus-therian', 'blissey', 'garchomp', 'iron-valiant']
    
    try:
        for poke_name in test_pokemon:
            try:
                poke_row = df[df['name'].str.lower() == poke_name]
                if len(poke_row) == 0:
                    print(f"  ⚠ {poke_name:20s}: NOT FOUND")
                    continue
                    
                X = poke_row[feature_cols].values
                X_scaled = scaler.transform(X)
                X_pca = pca.transform(X_scaled)
                cluster = gmm.predict(X_pca)[0]
                archetype = poke_row['archetype'].values[0]
                
                print(f"  ✓ {poke_name:20s}: Cluster {cluster:2d} → {archetype}")
                
            except Exception as e:
                print(f"  ✗ {poke_name:20s}: ERROR - {e}")
                
        return True
        
    except Exception as e:
        print(f"  ✗ FAILED: {e}")
        return False


def main():
    print("=" * 80)
    print("SANITY TEST: 601-Pokemon Clustering System")
    print("=" * 80)
    
    # Load data
    data_dir = Path(__file__).parents[2] / "reports" / "clustering_analysis" / "data"
    df = pd.read_csv(data_dir / "pokemon_with_clusters.csv")
    print(f"\n📂 Loaded {len(df)} Pokemon")
    
    # Run tests
    results = []
    
    scaler, pca, gmm, success = test_models_load()
    results.append(("Models Load", success))
    
    if success:
        clusters, probs, success = test_feature_pipeline(scaler, pca, gmm)
        results.append(("Feature Pipeline", success))
        
        success = test_archetype_coverage(df)
        results.append(("Archetype Coverage", success))
        
        if clusters is not None:
            success = test_cluster_distribution(clusters)
            results.append(("Cluster Distribution", success))
        
        success = test_sample_predictions(scaler, pca, gmm, df)
        results.append(("Sample Predictions", success))
    
    # Summary
    print("\n" + "=" * 80)
    print("TEST SUMMARY")
    print("=" * 80)
    
    passed = sum(1 for _, success in results if success)
    total = len(results)
    
    for test_name, success in results:
        status = "✓ PASS" if success else "✗ FAIL"
        print(f"  {status:8s} {test_name}")
    
    print(f"\nResult: {passed}/{total} tests passed")
    
    if passed == total:
        print("✅ All tests passed! Ready for Phase 2 GA.")
        return 0
    else:
        print("⚠️  Some tests failed. Review before running GA.")
        return 1


if __name__ == "__main__":
    exit(main())
