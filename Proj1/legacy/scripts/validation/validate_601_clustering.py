"""
Phase 6: Ground-Truth Validation (601 Pokemon)

Validates learned archetypes against competitive Pokémon roles.
Uses updated dataset with all form variants.
"""

import pandas as pd
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, cohen_kappa_score
from pathlib import Path


# ============================================================================
# GROUND TRUTH ANNOTATIONS (70+ Pokemon) - Broader Coverage
# ============================================================================

GROUND_TRUTH = {
    # Sweepers / Attackers
    "alakazam": "Sweeper",
    "dragapult": "Sweeper",
    "cloyster": "Sweeper",
    "volcarona": "Sweeper",
    "gyarados": "Sweeper",
    "weavile": "Sweeper",
    "gengar": "Sweeper",
    "infernape": "Sweeper",
    "lucario": "Sweeper",
    "excadrill": "Sweeper",
    "aegislash": "Sweeper",
    "thundurus": "Sweeper",
    "aerodactyl": "Sweeper",
    "jolteon": "Sweeper",
    "manectric": "Sweeper",
    "starmie": "Sweeper",
    "hydreigon": "Sweeper",
    "chandelure": "Sweeper",
    "sceptile": "Sweeper",
    "charizard": "Sweeper",
    "haxorus": "Sweeper",

    # Defensive Walls
    "blissey": "Wall",
    "toxapex": "Wall",
    "skarmory": "Wall",
    "ferrothorn": "Wall",
    "hippowdon": "Wall",
    "umbreon": "Wall",
    "cresselia": "Wall",
    "chansey": "Wall",
    "slowbro": "Wall",
    "snorlax": "Wall",
    "milotic": "Wall",
    "vaporeon": "Wall",
    "suicune": "Wall",
    "mamoswine": "Wall",
    "gastrodon": "Wall",
    "quagsire": "Wall",
    "swampert": "Wall",
    "dusknoir": "Wall",
    "registeel": "Wall",

    # Pivots / Utility
    "corviknight": "Pivot",
    "rotom": "Pivot",
    "landorus": "Pivot",
    "heatran": "Pivot",
    "togekiss": "Pivot",
    "mandibuzz": "Pivot",
    "scizor": "Pivot",
    "gliscor": "Pivot",
    "magnezone": "Pivot",
    "empoleon": "Pivot",
    "zapdos": "Pivot",
    "raikou": "Pivot",
    "rotom-wash": "Pivot",

    # Mixed / Balanced
    "greninja": "Mixed",
    "iron-valiant": "Mixed",
    "urshifu-single-strike": "Mixed",
    "garchomp": "Mixed",
    "salamence": "Mixed",
    "dragonite": "Mixed",
    "metagross": "Mixed",
    "defense-forme": "Wall",  # Deoxys-Defense
    "tyranitar": "Mixed",
    "latios": "Mixed",
    "latias": "Mixed",
    "mew": "Mixed",
    "celebi": "Mixed",
    "jirachi": "Mixed",
    "victini": "Mixed",
    "arcanine": "Mixed",
    "scrafty": "Mixed",
    "aggron": "Mixed",
    "lapras": "Mixed",
    "kingdra": "Mixed",
    "abomasnow": "Mixed",
    "gallade": "Mixed",
    "machamp": "Mixed",
}

# Enhanced role mapping using archetype + engineered features
def predict_role_enhanced(row):
    """
    Role prediction using archetype-based mapping.
    Archetypes already capture relevant statis patterns.
    """
    archetype = row.get('archetype', 'Generalist')
    
    # Simple archetype → role mapping
    mapping = {
        "Speed Sweeper": "Sweeper",
        "Fast Attacker": "Sweeper",
        "Physical Attacker": "Sweeper",
        "Defensive Tank": "Wall",
        "Balanced All-Rounder": "Mixed",
        "Generalist": "Mixed",
    }
    return mapping.get(archetype, "Mixed")


def main():
    print("=" * 80)
    print("PHASE 6: GROUND-TRUTH VALIDATION (601 POKEMON)")
    print("=" * 80)
    print()
    
    # Load data with archetype assignments
    data_path = Path("reports/clustering_analysis/data/pokemon_with_clusters.csv")
    pokemon_df = pd.read_csv(data_path)
    
    print(f"[DATA] Loaded {len(pokemon_df)} Pokemon with archetype assignments")
    print(f"[ARCHETYPES] {pokemon_df['archetype'].nunique()} archetypes found")
    
    # Try to load role priors from Phase 3 if available
    role_priors_path = None
    for attempt in [
        Path("reports/ga_results/run_601pokemon_20260305_171543/phase3_role_bootstrap/pokemon_role_priors.csv"),
        None,  # Sentinel to check globs
    ]:
        if attempt and attempt.exists():
            role_priors_path = attempt
            break
    
    role_priors_df = None
    if role_priors_path is None:
        # Try finding latest by globbing
        import glob
        candidates = glob.glob("reports/ga_results/*/phase3_role_bootstrap/pokemon_role_priors.csv")
        if candidates:
            role_priors_path = Path(max(candidates, key=lambda p: Path(p).parent.parent.name))
    
    if role_priors_path:
        try:
            role_priors_df = pd.read_csv(role_priors_path)
            print(f"[ROLES] Loaded {len(role_priors_df)} role priors from Phase 3")
            role_priors_df.columns = ['name' if c == 'name' else c for c in role_priors_df.columns]
        except Exception as e:
            print(f"[WARN] Could not load role priors: {e}")
    else:
        print(f"[ROLES] No Phase 3 role priors found (using archetype heuristics)")
    
    print()
    
    archetype_counts = pokemon_df['archetype'].value_counts()
    print("Archetype distribution:")
    for arch, count in archetype_counts.items():
        print(f"  {arch:25s}: {count:3d} ({count/len(pokemon_df)*100:5.1f}%)")
    print()
    
    # Validate test set
    test_pokemon = list(GROUND_TRUTH.keys())
    print(f"[TEST] Test set: {len(test_pokemon)} Pokemon")
    print()
    
    print("-" * 95)
    print(f"{'Pokemon':<20} {'True Role':<12} {'Archetype':<25} {'Pred Role':<12} {'Match':<6} {'Source'}")
    print("-" * 95)
    
    predictions = []
    found_count = 0
    
    for poke_name in test_pokemon:
        # Find in dataset (case-insensitive)
        poke_row = pokemon_df[pokemon_df['name'].str.lower() == poke_name.lower()]
        
        if len(poke_row) == 0:
            print(f"{poke_name:<20} {'(NOT FOUND)':>12}")
            continue
        
        found_count += 1
        
        row = poke_row.iloc[0]
        archetype = row['archetype']
        ground_truth = GROUND_TRUTH[poke_name]
        
        # Try role priors first, fall back to enhanced heuristic
        predicted_role = None
        source = "archetype"
        
        if role_priors_df is not None:
            role_row = role_priors_df[role_priors_df['name'].str.lower() == poke_name.lower()]
            if len(role_row) > 0:
                role_prior = role_row.iloc[0].get('role_prior', None)
                if role_prior and role_prior != 'unknown':
                    # Map role_prior back to ground truth role type
                    role_map = {
                        'physical_sweeper': 'Sweeper',
                        'special_sweeper': 'Sweeper',
                        'defensive_wall': 'Wall',
                        'fast_pivot': 'Pivot',
                        'bulky_pivot': 'Pivot',
                        'breaker': 'Sweeper',
                        'bulky_wincon': 'Wall',
                        'mixed_balanced': 'Mixed',
                        'setup_sweeper': 'Sweeper',
                    }
                    predicted_role = role_map.get(str(role_prior).lower(), 'Mixed')
                    source = "role_prior"
        
        if predicted_role is None:
            predicted_role = predict_role_enhanced(row)
        
        match = "[YES]" if predicted_role == ground_truth else "[NO]"
        
        print(f"{poke_name:<20} {ground_truth:<12} {archetype:<25} {predicted_role:<12} {match:<6} {source}")
        
        predictions.append({
            'pokemon': poke_name,
            'ground_truth': ground_truth,
            'predicted_role': predicted_role,
            'archetype': archetype,
            'source': source,
        })
    
    print()
    print(f"[RESULT] Found: {found_count}/{len(test_pokemon)} Pokemon")
    
    if found_count < len(test_pokemon):
        missing = [p for p in test_pokemon if p not in [pred['pokemon'] for pred in predictions]]
        print(f"\n[WARN] Missing Pokemon: {', '.join(missing)}")
    
    print()
    
    # ========================================================================
    # METRICS
    # ========================================================================
    
    if len(predictions) == 0:
        print("❌ No predictions to evaluate")
        return
    
    ground_truth_labels = [p['ground_truth'] for p in predictions]
    predicted_labels = [p['predicted_role'] for p in predictions]
    
    accuracy = accuracy_score(ground_truth_labels, predicted_labels)
    kappa = cohen_kappa_score(ground_truth_labels, predicted_labels)
    
    print("=" * 80)
    print("VALIDATION RESULTS")
    print("=" * 80)
    print()
    print(f"Accuracy:      {accuracy:.2%} ({sum(gt == pred for gt, pred in zip(ground_truth_labels, predicted_labels))}/{len(predictions)})")
    print(f"Cohen's Kappa: {kappa:.4f}")
    print()
    
    # Interpretation - higher threshold for competitive domain
    if accuracy >= 0.75:
        acc_rating = "[EXCEL] Excellent (75%+) - Strong competitive alignment"
    elif accuracy >= 0.65:
        acc_rating = "[GOOD] Good (65-75%) - Solid predictions"
    elif accuracy >= 0.50:
        acc_rating = "[OK] Acceptable (50-65%) - Some alignment, needs improvement"
    else:
        acc_rating = "[POOR] Weak (<50%) - High error rate"
    
    if kappa >= 0.75:
        kappa_rating = "[EXCEL] Near-perfect agreement (>0.75)"
    elif kappa >= 0.60:
        kappa_rating = "[GOOD] Substantial agreement (0.60-0.75)"
    elif kappa >= 0.40:
        kappa_rating = "[FAIR] Fair agreement (0.40-0.60)"
    else:
        kappa_rating = "[POOR] Slight agreement (<0.40)"
    
    print(f"Accuracy Rating: {acc_rating}")
    print(f"Kappa Rating:    {kappa_rating}")
    print()
    
    # Confusion matrix
    classes = sorted(set(ground_truth_labels + predicted_labels))
    conf_matrix = confusion_matrix(ground_truth_labels, predicted_labels, labels=classes)
    
    print("Confusion Matrix (rows=actual, cols=predicted):")
    print()
    print(f"{'Actual \\ Pred':<15}", end="")
    for cls in classes:
        print(f"{cls:<12}", end="")
    print()
    print("-" * (15 + 12 * len(classes)))
    
    for i, actual_cls in enumerate(classes):
        print(f"{actual_cls:<15}", end="")
        for j in range(len(classes)):
            print(f"{conf_matrix[i, j]:<12}", end="")
        print()
    
    print()
    
    # Per-role analysis
    print("Per-Role Performance:")
    print()
    for role in classes:
        role_gt = [i for i, gt in enumerate(ground_truth_labels) if gt == role]
        if len(role_gt) == 0:
            continue
        
        role_correct = sum(1 for i in role_gt if predicted_labels[i] == role)
        role_acc = role_correct / len(role_gt)
        
        print(f"  {role:<12}: {role_acc:.2%} ({role_correct}/{len(role_gt)} correct)")
    
    print()
    
    # ========================================================================
    # INTERPRETATION
    # ========================================================================
    
    print("=" * 80)
    print("INTERPRETATION & NEXT STEPS")
    print("=" * 80)
    print()
    
    if accuracy >= 0.75:
        print("[SUCCESS!] Predictions align with competitive knowledge!")
        print()
        print("Next steps:")
        print("  [PASS] Role assignments are reliable for team composition guidance")
        print("  [PASS] Consider using roles in meta-analysis pipelines")
        print("  [PASS] Expand ground truth for broader validation")
    
    elif accuracy >= 0.50:
        print("[PARTIAL] Predictions show promise but have room for improvement")
        print()
        print("What's working:")
        for role in classes:
            role_gt = [i for i, gt in enumerate(ground_truth_labels) if gt == role]
            if len(role_gt) > 0:
                role_correct = sum(1 for i in role_gt if predicted_labels[i] == role)
                role_acc = role_correct / len(role_gt)
                if role_acc >= 0.60:
                    print(f"  [PASS] {role:12s}: {role_acc:.0%} accuracy")
        
        print()
        print("What needs improvement:")
        for role in classes:
            role_gt = [i for i, gt in enumerate(ground_truth_labels) if gt == role]
            if len(role_gt) > 0:
                role_correct = sum(1 for i in role_gt if predicted_labels[i] == role)
                role_acc = role_correct / len(role_gt)
                if role_acc < 0.60:
                    print(f"  [FAIL] {role:12s}: {role_acc:.0%} accuracy - needs refinement")
        
        print()
        print("Recommended improvements:")
        print("  • Refine role priors with additional move data")
        print("  • Expand ground truth with more competitive examples")
        print("  • Adjust heuristics based on per-role confusion patterns")
        
    else:
        print("[INVESTIGATE] Predictions don't align with domain knowledge")
        print()
        print("Possible reasons:")
        print("  • Stat-based clustering doesn't capture competitive styles")
        print("  • Role ground truth may need re-calibration")
        print("  • May need to incorporate movepool/tier data")
    
    print()
    print("=" * 80)
    print()


if __name__ == "__main__":
    main()
