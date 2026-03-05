# -*- coding: utf-8 -*-
"""
Phase 3: System Validation
===========================

Critical validation tests to determine if the GA is actually working:
1. Random search baseline - Does GA beat brute force?
2. Multi-swap neighbor test - Can 2-swaps escape local optimum?
3. Entropy overfitting check - Does diversity come at cost of strength?
4. Component scale analysis - Are fitness terms balanced?
5. GA stability test - Is the optimizer consistent?

Runtime estimate: ~90-120 minutes
"""

import sys
import os
import json
import io

# Set UTF-8 encoding for Windows console
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
if sys.stderr.encoding != 'utf-8':
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')
import time
import copy
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Tuple
import itertools
from collections import Counter

# Add parent directory to path
proj_root = Path(__file__).parent.parent.parent
sys.path.append(str(proj_root))
sys.path.append(str(proj_root / "src" / "models"))

from src.models.ga_optimization import PokemonGA, load_pokemon_data
from src.models.ga_config import get_config_c
from src.models.ga_fitness import evaluate_fitness

# ============================================================================
# Configuration
# ============================================================================

RANDOM_BASELINE_SAMPLES = 10000  # 10k random teams
MULTI_SWAP_SAMPLES = 1000        # 1k 2-swap tests
GA_STABILITY_RUNS = 20           # 20 different seeds
GA_STABILITY_SEED_START = 100    # Start at seed 100
FITNESS_CONSISTENCY_TOL = 1e-6   # Strict consistency check tolerance

# Reduced config for stability test - matches PokemonGA structure
STABILITY_CONFIG = {
    "name": "StabilityTest",
    "population": {
        "size": 50,          # Reduced from 150
        "generations": 50,   # Reduced from 250
        "tournament_k": 3,
        "elitism": 5
    },
    "fitness": {
        "base_stats_weight": 0.40,
        "type_coverage_weight": 0.30,
        "synergy_weight": 0.15,
        "diversity_weight": 0.15,
        "imbalance_lambda": 0.20,
        "weakness_lambda": 0.10
    },
    "initialization": {
        "method": "inverse"  # Use inverse weighting
    },
    "mutation": {
        "rate": 0.15,
        "weighted": True
    },
    "crossover": {
        "rate": 0.80,
        "type": "two_point"
    },
    "random_seed": 42  # Will be overridden per run
}


def resolve_phase2_results_path() -> Path:
    """Resolve Phase 2 results from common locations after directory reorganization."""
    candidates = [
        # New local organized path (after consolidation to ablation_study)
        Path(__file__).parent.parent.parent / "phase2" / "results" / "03_ablation_sensitivity_results.json",
        # Legacy local path (before consolidation)
        Path(__file__).parent.parent / "phase2" / "results" / "03_ablation_sensitivity_results.json",
    ]

    for path in candidates:
        if path.exists():
            return path

    raise FileNotFoundError(
        "Could not locate Phase 2 results file in known locations. "
        "Expected 03_ablation_sensitivity_results.json under a phase2 results directory."
    )

# ============================================================================
# Test 1: Random Search Baseline
# ============================================================================

def test_random_baseline(pokemon_df: pd.DataFrame, config: Dict, n_samples: int = 10000) -> Dict:
    """
    Generate N random teams and compare with GA performance.
    Critical test: GA must beat random search to be useful.
    """
    print("\n" + "="*70)
    print("TEST 1: Random Search Baseline")
    print("="*70)
    print(f"\nGenerating {n_samples:,} random teams...")
    
    start_time = time.time()
    
    # Get archetype groups for stratified sampling
    archetype_groups = {}
    for archetype in pokemon_df['archetype'].unique():
        archetype_groups[archetype] = pokemon_df[pokemon_df['archetype'] == archetype].index.tolist()
    
    fitness_scores = []
    best_fitness = -np.inf
    best_team = None
    
    # Generate random teams
    for i in range(n_samples):
        if (i + 1) % 2000 == 0:
            print(f"  Generated {i+1:,}/{n_samples:,} teams...")
        
        # Random team: sample 6 Pokémon uniformly
        team_indices = np.random.choice(len(pokemon_df), size=6, replace=False)
        team_df = pokemon_df.iloc[team_indices].copy()
        
        # Evaluate fitness
        fitness, breakdown = evaluate_fitness(team_df, config)
        fitness_scores.append(fitness)
        
        if fitness > best_fitness:
            best_fitness = fitness
            best_team = team_df['name'].tolist()
    
    elapsed = time.time() - start_time
    
    # Compute statistics
    fitness_scores = np.array(fitness_scores)
    results = {
        'experiment': 'random_baseline',
        'n_samples': n_samples,
        'mean_fitness': float(np.mean(fitness_scores)),
        'median_fitness': float(np.median(fitness_scores)),
        'std_fitness': float(np.std(fitness_scores)),
        'min_fitness': float(np.min(fitness_scores)),
        'max_fitness': float(np.max(fitness_scores)),
        'percentile_25': float(np.percentile(fitness_scores, 25)),
        'percentile_75': float(np.percentile(fitness_scores, 75)),
        'percentile_95': float(np.percentile(fitness_scores, 95)),
        'percentile_99': float(np.percentile(fitness_scores, 99)),
        'best_team': best_team,
        'elapsed_seconds': elapsed
    }
    
    # Print summary
    print(f"\n📊 Random Search Statistics:")
    print(f"   Mean fitness:      {results['mean_fitness']:.4f}")
    print(f"   Median fitness:    {results['median_fitness']:.4f}")
    print(f"   Std deviation:     {results['std_fitness']:.4f}")
    print(f"   95th percentile:   {results['percentile_95']:.4f}")
    print(f"   99th percentile:   {results['percentile_99']:.4f}")
    print(f"   Best random team:  {results['max_fitness']:.4f}")
    print(f"   Best team: {','.join(best_team)}")
    print(f"\n⏱️  Elapsed: {elapsed:.1f}s")
    
    # Compare with GA baseline from Phase 2
    ga_baseline = 0.7334  # From Phase 2 ConfigC
    print(f"\n🔬 Comparison:")
    print(f"   GA best (Phase 2):    {ga_baseline:.4f}")
    print(f"   Random best:          {results['max_fitness']:.4f}")
    print(f"   GA advantage:         {ga_baseline - results['max_fitness']:.4f} ({(ga_baseline/results['max_fitness'] - 1)*100:.1f}%)")
    print(f"   GA vs 95th percentile: {ga_baseline - results['percentile_95']:.4f}")
    
    return results


# ============================================================================
# Test 2: Multi-Swap Neighbor Test
# ============================================================================

def test_multi_swap_neighbors(pokemon_df: pd.DataFrame, config: Dict, 
                               best_team_names: List[str], n_samples: int = 1000) -> Dict:
    """
    Test if 2-member swaps can escape the local optimum.
    If many 2-swaps improve fitness, GA converged prematurely.
    """
    print("\n" + "="*70)
    print("TEST 2: Multi-Swap Neighbor Test")
    print("="*70)
    print(f"\nTesting {n_samples:,} 2-member swaps around best team...")
    
    start_time = time.time()
    
    # Get best team
    best_team_df = pokemon_df[pokemon_df['name'].isin(best_team_names)].copy()
    original_fitness, original_breakdown = evaluate_fitness(best_team_df, config)
    
    print(f"\n🎯 Original team fitness: {original_fitness:.4f}")
    print(f"   Team: {','.join(best_team_names)}")
    
    # Get archetype distribution
    original_archetypes = best_team_df['archetype'].tolist()
    archetype_counts = Counter(original_archetypes)
    
    print(f"   Archetypes: {dict(archetype_counts)}")
    
    improvements = []
    better_teams = []
    
    # Sample 2-swap neighbors
    for i in range(n_samples):
        if (i + 1) % 200 == 0:
            print(f"  Tested {i+1:,}/{n_samples:,} swaps...")
        
        # Select 2 random positions to swap
        positions = np.random.choice(6, size=2, replace=False)
        
        # Create variant
        variant_df = best_team_df.copy()
        
        # For each position, sample a replacement from same or different archetype
        for pos in positions:
            original_pokemon = variant_df.iloc[pos]['name']
            original_archetype = variant_df.iloc[pos]['archetype']
            
            # 50% chance: same archetype, 50% chance: different archetype
            if np.random.rand() < 0.5:
                # Same archetype
                candidates = pokemon_df[
                    (pokemon_df['archetype'] == original_archetype) & 
                    (~pokemon_df['name'].isin(variant_df['name']))
                ]
            else:
                # Any archetype
                candidates = pokemon_df[~pokemon_df['name'].isin(variant_df['name'])]
            
            if len(candidates) > 0:
                replacement = candidates.sample(n=1).iloc[0]
                variant_df.iloc[pos] = replacement
        
        # Evaluate variant
        variant_fitness, _ = evaluate_fitness(variant_df, config)
        
        delta = variant_fitness - original_fitness
        if delta > 0:
            improvements.append(delta)
            better_teams.append({
                'team': variant_df['name'].tolist(),
                'fitness': variant_fitness,
                'improvement': delta
            })
    
    elapsed = time.time() - start_time
    
    # Compute statistics
    results = {
        'experiment': 'multi_swap_neighbors',
        'n_samples': n_samples,
        'swap_size': 2,
        'original_fitness': original_fitness,
        'n_improvements': len(improvements),
        'improvement_rate': len(improvements) / n_samples,
        'max_improvement': float(np.max(improvements)) if improvements else 0.0,
        'mean_improvement': float(np.mean(improvements)) if improvements else 0.0,
        'best_neighbor': better_teams[np.argmax([t['improvement'] for t in better_teams])] if better_teams else None,
        'elapsed_seconds': elapsed
    }
    
    # Print summary
    print(f"\n📊 Multi-Swap Results:")
    print(f"   Improvements found: {results['n_improvements']}/{n_samples} ({results['improvement_rate']*100:.1f}%)")
    print(f"   Max improvement:    {results['max_improvement']:.4f}")
    if results['max_improvement'] > 0:
        print(f"   Mean improvement:   {results['mean_improvement']:.4f}")
        print(f"   Best neighbor fitness: {results['best_neighbor']['fitness']:.4f}")
        print(f"   Best neighbor team: {','.join(results['best_neighbor']['team'])}")
    
    print(f"\n⏱️  Elapsed: {elapsed:.1f}s")
    
    # Interpretation
    print(f"\n🔬 Interpretation:")
    if results['improvement_rate'] < 0.01:
        print("   ✅ Local optimum is VERY SHARP - excellent convergence")
    elif results['improvement_rate'] < 0.05:
        print("   ✅ Local optimum is SHARP - good convergence")
    elif results['improvement_rate'] < 0.10:
        print("   ⚠️  Some escape routes exist - consider longer runs")
    else:
        print("   ❌ Many improvements possible - GA may have stopped early")
    
    return results


# ============================================================================
# Test 3: Entropy Overfitting Check
# ============================================================================

def test_entropy_overfitting(pokemon_df: pd.DataFrame) -> Dict:
    """
    Analyze Phase 2 entropy sweep to check if high diversity_weight
    sacrifices base strength for diversity.
    """
    print("\n" + "="*70)
    print("TEST 3: Entropy Overfitting Analysis")
    print("="*70)
    print("\nAnalyzing entropy sweep from Phase 2...")
    
    # Load Phase 2 results
    phase2_path = resolve_phase2_results_path()
    
    with open(phase2_path, 'r') as f:
        phase2_data = json.load(f)
    
    entropy_runs = phase2_data['experiments']['entropy_sweep']['runs']
    
    # For each run, get the team and compute components
    analysis = []
    
    print("\n📊 Entropy Sweep Component Breakdown:")
    print(f"\n{'Weight':<8} {'Phase2':<10} {'Recalc':<10} {'Archetypes':<12} {'EntropyBonus':<13}")
    print("-" * 70)
    
    for run in entropy_runs:
        weight = run['diversity_weight']
        fitness = run['best_fitness']
        team_names = run['best_team'].split(',')
        
        # Get team
        team_df = pokemon_df[pokemon_df['name'].isin(team_names)].copy()
        
        # Recompute fitness with breakdown
        config_temp = get_config_c()
        config_temp['fitness']['diversity_weight'] = weight
        full_fitness, breakdown = evaluate_fitness(team_df, config_temp)

        # Integrity check: stored and recomputed fitness must match exactly
        assert abs(fitness - full_fitness) < FITNESS_CONSISTENCY_TOL, (
            f"Fitness consistency check failed for diversity_weight={weight}: "
            f"stored_fitness={fitness:.12f}, recomputed_fitness={full_fitness:.12f}, "
            f"delta={abs(fitness - full_fitness):.12f}"
        )
        
        # Count unique archetypes
        unique_archetypes = len(team_df['archetype'].unique())
        
        analysis.append({
            'diversity_weight': weight,
            'fitness': fitness,
            'recomputed_fitness': full_fitness,
            'unique_archetypes': unique_archetypes,
            'team': team_names,
            'breakdown': breakdown
        })
        
        # Print row (approximated since we don't have exact breakdown storage)
        print(f"{weight:<8.2f} {fitness:<10.4f} {full_fitness:<10.4f} {unique_archetypes:<12d} {breakdown['entropy_bonus']:<13.4f}")
    
    results = {
        'experiment': 'entropy_overfitting',
        'analysis': analysis
    }
    
    # Check for overfitting symptoms
    print(f"\n🔬 Overfitting Check:")
    
    # Check 1: Does archetype diversity saturate quickly?
    archetypes_at_low_weight = analysis[0]['unique_archetypes']  # weight=0.5
    archetypes_at_mid_weight = analysis[2]['unique_archetypes']  # weight=0.25
    archetypes_at_baseline = analysis[3]['unique_archetypes']    # weight=0.15
    
    print(f"   Unique archetypes:")
    print(f"     weight=0.50: {analysis[0]['unique_archetypes']}")
    print(f"     weight=0.35: {analysis[1]['unique_archetypes']}")
    print(f"     weight=0.25: {analysis[2]['unique_archetypes']}")
    print(f"     weight=0.15: {analysis[3]['unique_archetypes']}")
    print(f"     weight=0.10: {analysis[4]['unique_archetypes']}")
    
    # Check 2: Is fitness gain proportional to weight increase?
    fitness_ratio_high = analysis[0]['fitness'] / analysis[3]['fitness']  # 0.5 vs 0.15
    weight_ratio = 0.5 / 0.15
    
    print(f"\n   Proportionality check:")
    print(f"     Weight increase: {weight_ratio:.2f}x (0.15 → 0.50)")
    print(f"     Fitness increase: {fitness_ratio_high:.2f}x")
    
    if fitness_ratio_high > 1.3:
        print(f"     ⚠️  Fitness grew faster than weight - possible overfitting")
    else:
        print(f"     ✅ Fitness growth reasonable")
    
    return results


# ============================================================================
# Test 4: Component Scale Analysis
# ============================================================================

def test_component_scales(pokemon_df: pd.DataFrame, config: Dict) -> Dict:
    """
    Check if fitness components are properly scaled.
    Dominant terms can distort the optimization.
    """
    print("\n" + "="*70)
    print("TEST 4: Component Scale Analysis")
    print("="*70)
    print("\nAnalyzing component magnitudes for best teams...")
    
    # Load Phase 2 results to get best teams
    phase2_path = resolve_phase2_results_path()
    
    with open(phase2_path, 'r') as f:
        phase2_data = json.load(f)
    
    # Get best teams from different experiments
    test_teams = []
    
    # Baseline team
    baseline = phase2_data['experiments']['ablation_tests']['runs'][0]
    test_teams.append({
        'name': 'baseline_full',
        'team': baseline['best_team'].split(',')
    })
    
    # Entropy sweep teams
    for run in phase2_data['experiments']['entropy_sweep']['runs']:
        test_teams.append({
            'name': f"entropy_{run['diversity_weight']}",
            'team': run['best_team'].split(',')
        })
    
    # Analyze components for each team
    print(f"\n📊 Component Contributions:")
    print(f"\n{'Team':<25} {'Total':<8} | Base components (actual values)")
    print("-" * 90)
    
    all_breakdowns = []
    
    for test in test_teams:
        team_df = pokemon_df[pokemon_df['name'].isin(test['team'])].copy()
        fitness, breakdown = evaluate_fitness(team_df, config)
        
        all_breakdowns.append({
            'team_name': test['name'],
            'total_fitness': fitness,
            'breakdown': breakdown
        })
        
        # Print (note: breakdown structure depends on ga_fitness.py implementation)
        print(f"{test['name']:<25} {fitness:<8.4f} | {breakdown}")
    
    # Compute average component magnitudes
    print(f"\n🔬 Scale Analysis:")
    print(f"   Components should be roughly same order of magnitude")
    print(f"   If one term >> others, it dominates optimization")
    
    results = {
        'experiment': 'component_scales',
        'teams_analyzed': all_breakdowns
    }
    
    return results


# ============================================================================
# Test 5: GA Stability Test
# ============================================================================

def test_ga_stability(pokemon_df: pd.DataFrame, n_runs: int = 20) -> Dict:
    """
    Run GA multiple times with different seeds.
    High variance = unstable optimizer.
    """
    print("\n" + "="*70)
    print("TEST 5: GA Stability Test")
    print("="*70)
    print(f"\nRunning GA {n_runs} times with different seeds...")
    print(f"Config: {STABILITY_CONFIG['population']['size']} pop × {STABILITY_CONFIG['population']['generations']} gen")
    
    start_time = time.time()
    
    results_list = []
    
    for i in range(n_runs):
        seed = GA_STABILITY_SEED_START + i
        print(f"\n  Run {i+1}/{n_runs} (seed={seed})...")
        
        # Create config (deep copy for nested dict)
        config = copy.deepcopy(STABILITY_CONFIG)
        config['random_seed'] = seed
        config['name'] = f"stability_seed_{seed}"
        
        # Run GA
        ga = PokemonGA(pokemon_df=pokemon_df, config=config)
        
        fitness_history = ga.run()
        
        # Get best team
        best_teams = ga.get_best_teams(n=1)
        best_team, best_fitness, breakdown = best_teams[0]
        
        # Calculate convergence generation (when best fitness first reached within 0.1% of final)
        convergence_gen = 0
        if len(fitness_history) > 0 and 'best_fitness' in fitness_history.columns:
            threshold = best_fitness * 0.999  # Within 0.1% of final
            converged_gens = fitness_history[fitness_history['best_fitness'] >= threshold]
            if len(converged_gens) > 0:
                convergence_gen = int(converged_gens.iloc[0]['generation'])
        
        results_list.append({
            'seed': seed,
            'best_fitness': best_fitness,
            'best_team': best_team['name'].tolist(),
            'convergence_generation': convergence_gen
        })
        
        print(f"    ✓ Fitness: {best_fitness:.4f}, Conv gen: {convergence_gen}")
    
    elapsed = time.time() - start_time
    
    # Compute statistics
    fitnesses = [r['best_fitness'] for r in results_list]
    conv_gens = [r['convergence_generation'] for r in results_list]
    
    results = {
        'experiment': 'ga_stability',
        'n_runs': n_runs,
        'config': STABILITY_CONFIG,
        'runs': results_list,
        'statistics': {
            'mean_fitness': float(np.mean(fitnesses)),
            'std_fitness': float(np.std(fitnesses)),
            'min_fitness': float(np.min(fitnesses)),
            'max_fitness': float(np.max(fitnesses)),
            'median_fitness': float(np.median(fitnesses)),
            'cv_fitness': float(np.std(fitnesses) / np.mean(fitnesses)),  # Coefficient of variation
            'mean_conv_gen': float(np.mean(conv_gens)),
            'std_conv_gen': float(np.std(conv_gens))
        },
        'elapsed_seconds': elapsed
    }
    
    # Print summary
    print(f"\n📊 Stability Statistics:")
    print(f"   Mean fitness:  {results['statistics']['mean_fitness']:.4f}")
    print(f"   Std deviation: {results['statistics']['std_fitness']:.4f}")
    print(f"   Min fitness:   {results['statistics']['min_fitness']:.4f}")
    print(f"   Max fitness:   {results['statistics']['max_fitness']:.4f}")
    print(f"   CV (std/mean): {results['statistics']['cv_fitness']:.4f}")
    print(f"\n   Mean conv gen: {results['statistics']['mean_conv_gen']:.1f}")
    print(f"   Std conv gen:  {results['statistics']['std_conv_gen']:.1f}")
    print(f"\n⏱️  Total elapsed: {elapsed:.1f}s ({elapsed/60:.1f} min)")
    
    # Interpretation
    print(f"\n🔬 Interpretation:")
    cv = results['statistics']['cv_fitness']
    if cv < 0.02:
        print("   ✅ EXCELLENT stability (CV < 2%)")
    elif cv < 0.05:
        print("   ✅ GOOD stability (CV < 5%)")
    elif cv < 0.10:
        print("   ⚠️  MODERATE stability (CV < 10%) - acceptable but could improve")
    else:
        print("   ❌ POOR stability (CV > 10%) - optimizer is unreliable")
    
    return results


# ============================================================================
# Main Phase 3 Orchestration
# ============================================================================

def main():
    print("\n" + "="*70)
    print("PHASE 3: SYSTEM VALIDATION")
    print("="*70)
    print("\nValidation suite to determine if GA is genuinely effective.")
    print("\nTests:")
    print("  1. Random baseline - Does GA beat brute force?")
    print("  2. Multi-swap neighbors - Can 2-swaps escape local optimum?")
    print("  3. Entropy overfitting - Does diversity hurt base strength?")
    print("  4. Component scales - Are fitness terms balanced?")
    print("  5. GA stability - Is optimizer consistent across seeds?")
    print("\nEstimated runtime: 90-120 minutes")
    print("="*70)
    
    start_time = time.time()
    
    # Load data
    print("\n📂 Loading Pokémon data...")
    pokemon_df = load_pokemon_data()
    print(f"   Loaded {len(pokemon_df)} Pokémon")
    
    # Get config
    config = get_config_c()
    
    # Get best team from Phase 2
    best_team_names = ['venusaur', 'groudon', 'flutter-mane', 'stakataka', 'zekrom', 'mewtwo']
    
    # Run tests
    results = {
        'phase': 'Phase 3 - System Validation',
        'date': datetime.now().isoformat(),
        'num_pokemon': len(pokemon_df),
        'tests': {}
    }
    
    # Test 1: Random baseline
    results['tests']['random_baseline'] = test_random_baseline(
        pokemon_df, config, n_samples=RANDOM_BASELINE_SAMPLES
    )
    
    # Test 2: Multi-swap neighbors
    results['tests']['multi_swap_neighbors'] = test_multi_swap_neighbors(
        pokemon_df, config, best_team_names, n_samples=MULTI_SWAP_SAMPLES
    )
    
    # Test 3: Entropy overfitting
    results['tests']['entropy_overfitting'] = test_entropy_overfitting(pokemon_df)
    
    # Test 4: Component scales
    results['tests']['component_scales'] = test_component_scales(pokemon_df, config)
    
    # Test 5: GA stability
    results['tests']['ga_stability'] = test_ga_stability(
        pokemon_df, n_runs=GA_STABILITY_RUNS
    )
    
    # Save results
    total_elapsed = time.time() - start_time
    results['total_elapsed_seconds'] = total_elapsed
    
    output_dir = Path(__file__).parent.parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "04_system_validation_results.json"
    
    with open(output_path, 'w') as f:
        json.dump(results, f, indent=2)
    
    print("\n" + "="*70)
    print("PHASE 3 COMPLETE")
    print("="*70)
    print(f"Total elapsed time: {total_elapsed:.1f} seconds ({total_elapsed/60:.1f} minutes)")
    print(f"Results saved to: {output_path}")
    print("\n🎯 Key Validation Metrics:")
    print(f"   GA vs Random best: {config.get('baseline_ga_fitness', 0.7334) - results['tests']['random_baseline']['max_fitness']:.4f}")
    print(f"   2-swap escape rate: {results['tests']['multi_swap_neighbors']['improvement_rate']*100:.1f}%")
    print(f"   GA stability (CV): {results['tests']['ga_stability']['statistics']['cv_fitness']:.4f}")
    
    print("\nNext: Review results to validate system before making changes.")


if __name__ == "__main__":
    main()
