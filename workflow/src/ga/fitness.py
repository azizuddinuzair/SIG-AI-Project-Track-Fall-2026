"""
GA Fitness Function Components

Modular fitness evaluation for Pokémon team optimization:
1. Base strength (stats normalization)
2. Type coverage (offensive matchup quality)
3. Team synergy (placeholder for advanced metrics)
4. Entropy-based diversity bonus
5. Quadratic archetype balance penalty
6. Quadratic shared weakness penalty

All components normalized to [0, 1] range for consistent weighting.
"""

import numpy as np
import pandas as pd
from collections import Counter
from typing import List, Dict, Tuple


# Gen 9 Type Effectiveness Chart (18x18 matrix)
# Rows: Attacking type, Columns: Defending type
# Values: 2.0 (super effective), 1.0 (neutral), 0.5 (not very effective), 0.0 (immune)
TYPE_NAMES = [
    'normal', 'fire', 'water', 'electric', 'grass', 'ice',
    'fighting', 'poison', 'ground', 'flying', 'psychic', 'bug',
    'rock', 'ghost', 'dragon', 'dark', 'steel', 'fairy'
]

# Hardcoded Gen 9 type chart (attacking type × defending type)
TYPE_CHART_RAW = [
    # nor  fir  wat  ele  gra  ice  fig  poi  gro  fly  psy  bug  roc  gho  dra  dar  ste  fai
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 0.0, 1.0, 1.0, 0.5, 1.0],  # normal
    [1.0, 0.5, 0.5, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 2.0, 1.0],  # fire
    [1.0, 2.0, 0.5, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0, 1.0],  # water
    [1.0, 1.0, 2.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0],  # electric
    [1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 1.0, 0.5, 2.0, 0.5, 1.0, 0.5, 2.0, 1.0, 0.5, 1.0, 0.5, 1.0],  # grass
    [1.0, 0.5, 0.5, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0],  # ice
    [2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 0.5, 0.5, 0.5, 2.0, 0.0, 1.0, 2.0, 2.0, 0.5],  # fighting
    [1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 1.0, 0.5, 0.5, 1.0, 1.0, 0.0, 2.0],  # poison
    [1.0, 2.0, 1.0, 2.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.0, 1.0, 0.5, 2.0, 1.0, 1.0, 1.0, 2.0, 1.0],  # ground
    [1.0, 1.0, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 0.5, 1.0],  # flying
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 0.0, 0.5, 1.0],  # psychic
    [1.0, 0.5, 1.0, 1.0, 2.0, 1.0, 0.5, 0.5, 1.0, 0.5, 2.0, 1.0, 1.0, 0.5, 1.0, 2.0, 0.5, 0.5],  # bug
    [1.0, 2.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 0.5, 2.0, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0],  # rock
    [0.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 1.0],  # ghost
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 0.5, 0.0],  # dragon
    [1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.5, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 2.0, 1.0, 0.5, 1.0, 0.5],  # dark
    [1.0, 0.5, 0.5, 0.5, 1.0, 2.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 1.0, 1.0, 1.0, 0.5, 2.0],  # steel
    [1.0, 0.5, 1.0, 1.0, 1.0, 1.0, 2.0, 0.5, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 2.0, 2.0, 0.5, 1.0],  # fairy
]

TYPE_CHART = np.array(TYPE_CHART_RAW)
TYPE_INDEX = {name: idx for idx, name in enumerate(TYPE_NAMES)}


def get_type_effectiveness(attacking_type: str, defending_types: List[str]) -> float:
    """
    Calculate type effectiveness multiplier.
    
    Args:
        attacking_type: Attacking move type
        defending_types: List of 1-2 defending types
        
    Returns:
        Multiplier (e.g., 2.0, 1.0, 0.5, 0.25, 0.0)
    """
    if attacking_type not in TYPE_INDEX:
        return 1.0
    
    atk_idx = TYPE_INDEX[attacking_type]
    multiplier = 1.0
    
    for def_type in defending_types:
        if def_type and def_type in TYPE_INDEX:
            def_idx = TYPE_INDEX[def_type]
            multiplier *= TYPE_CHART[atk_idx, def_idx]
    
    return multiplier


# ============================================================================
# FITNESS COMPONENT 1: Base Strength
# ============================================================================

def compute_base_strength(team: pd.DataFrame) -> float:
    """
    Compute normalized base strength from raw stats.
    
    Components:
    - Offensive index (ATK + SPA)
    - Defensive index (HP*0.5 + DEF + SPD)
    - Speed percentile
    
    Normalization:
    - Expected total team stats: ~1200-3600
    - Normalize to [0, 1] using empirical bounds
    
    Args:
        team: DataFrame with 6 Pokémon (columns: hp, attack, etc.)
        
    Returns:
        Normalized strength score [0, 1]
    """
    # Calculate team aggregate stats
    team_offense = team['offensive_index'].sum()
    team_defense = team['defensive_index'].sum()
    team_speed = team['speed_percentile'].mean()  # Average speed tier
    
    # Normalize components (empirical bounds from dataset)
    # Offensive: 600-1800 (weak team to strong team)
    offense_norm = np.clip((team_offense - 900) / 900, 0, 1)
    
    # Defensive: 900-1500
    defense_norm = np.clip((team_defense - 1000) / 600, 0, 1)
    
    # Speed: Already in [0, 1] percentile
    speed_norm = team_speed
    
    # Weighted combination (offense = defense in importance, speed = mobility)
    strength = 0.40 * offense_norm + 0.40 * defense_norm + 0.20 * speed_norm
    
    return float(strength)


# ============================================================================
# FITNESS COMPONENT 2: Type Coverage
# ============================================================================

def compute_type_coverage(team: pd.DataFrame) -> float:
    """
    Compute offensive type coverage quality.
    
    Measures: % of types where team has ≥1 super-effective move.
    
    Calculation:
    1. For each of 18 types, check if any team member can hit it super-effectively
    2. Count coverage types / 18
    
    Args:
        team: DataFrame with type1, type2 columns
        
    Returns:
        Coverage score [0, 1]
    """
    coverage = set()
    
    for _, pokemon in team.iterrows():
        # Get this Pokémon's offensive types
        offensive_types = [pokemon['type1']]
        if pd.notna(pokemon['type2']):
            offensive_types.append(pokemon['type2'])
        
        # Check which types this Pokémon can hit super-effectively
        for atk_type in offensive_types:
            if atk_type not in TYPE_INDEX:
                continue
            atk_idx = TYPE_INDEX[atk_type]
            
            for def_idx, def_type in enumerate(TYPE_NAMES):
                if TYPE_CHART[atk_idx, def_idx] >= 2.0:
                    coverage.add(def_type)
    
    return len(coverage) / 18.0


# ============================================================================
# FITNESS COMPONENT 3: Team Synergy (Placeholder)
# ============================================================================

def compute_synergy(team: pd.DataFrame) -> float:
    """
    Compute team synergy score.
    
    Placeholder for advanced metrics:
    - Physical/Special attacker balance
    - Coverage overlap minimization
    - Role diversity (sweeper, wall, pivot)
    
    Current implementation: Simple physical/special balance
    
    Args:
        team: DataFrame with physical_special_bias column
        
    Returns:
        Synergy score [0, 1]
    """
    # Simple metric: Measure physical/special attacker diversity
    biases = team['physical_special_bias'].values
    
    # Ideal: Some physical, some special attackers
    # Penalty for all-physical or all-special teams
    mean_bias = np.mean(biases)
    std_bias = np.std(biases)
    
    # Normalize: High std = diverse, centered around 0 = balanced
    balance_score = 1.0 - np.abs(mean_bias)  # Penalty for extreme skew
    diversity_score = min(std_bias / 0.5, 1.0)  # Reward variance
    
    synergy = 0.5 * balance_score + 0.5 * diversity_score
    return float(synergy)


# ============================================================================
# FITNESS COMPONENT 4: Entropy-Based Diversity Bonus
# ============================================================================

def compute_entropy_bonus(team: pd.DataFrame, config: Dict) -> float:
    """
    Compute Shannon entropy of archetype distribution.
    
    Entropy formula:
        H = -Σ p_i * log₂(p_i)
    
    Normalized by H_max (uniform distribution over 6 archetypes):
        H_max = log₂(6) ≈ 2.585
    
    Args:
        team: DataFrame with 'archetype' column
        config: Config dict with fitness.diversity_weight
        
    Returns:
        Diversity bonus [0, diversity_weight]
    """
    weight = config['fitness']['diversity_weight']
    
    if weight == 0:
        return 0.0
    
    # Count archetype distribution
    archetype_counts = team['archetype'].value_counts()
    proportions = archetype_counts / len(team)
    
    # Shannon entropy
    entropy = -np.sum(proportions * np.log2(proportions))
    
    # Normalize by max entropy
    max_entropy = np.log2(6)  # 6 unique archetypes possible
    normalized_entropy = entropy / max_entropy
    
    return float(weight * normalized_entropy)


# ============================================================================
# FITNESS COMPONENT 5: Quadratic Archetype Balance Penalty
# ============================================================================

def compute_imbalance_penalty(team: pd.DataFrame, config: Dict) -> float:
    """
    Penalize archetype stacking using quadratic penalty.
    
    Formula:
        penalty = -λ × (max_proportion)²
    
    Where:
        max_proportion = (count of most common archetype) / 6
    
    Examples:
        [2,2,1,1,0,0] → max_prop = 2/6 = 0.33 → penalty = -0.33² × λ = -0.11λ
        [4,1,1,0,0,0] → max_prop = 4/6 = 0.67 → penalty = -0.67² × λ = -0.45λ
        [6,0,0,0,0,0] → max_prop = 6/6 = 1.00 → penalty = -1.00² × λ = -1.00λ
    
    Args:
        team: DataFrame with 'archetype' column
        config: Config dict with fitness.imbalance_lambda
        
    Returns:
        Balance penalty (negative value)
    """
    lambda_val = config['fitness']['imbalance_lambda']
    
    if lambda_val == 0:
        return 0.0
    
    # Find most common archetype proportion
    archetype_counts = team['archetype'].value_counts()
    max_count = archetype_counts.max()
    max_proportion = max_count / len(team)
    
    penalty = -lambda_val * (max_proportion ** 2)
    return float(penalty)


# ============================================================================
# FITNESS COMPONENT 6: Quadratic Shared Weakness Penalty
# ============================================================================

def compute_weakness_penalty(team: pd.DataFrame, config: Dict) -> float:
    """
    Penalize shared type weaknesses across team.
    
    Formula:
        weakness_score = Σ_t (n_t / 6)²
        penalty = -μ × weakness_score
    
    Where:
        n_t = number of Pokémon weak to type t
    
    This quadratically penalizes stacking weaknesses:
        2 weak → (2/6)² = 0.11
        4 weak → (4/6)² = 0.44 (4× penalty vs 2)
        6 weak → (6/6)² = 1.00 (full penalty)
    
    Args:
        team: DataFrame with type_defense_* columns
        config: Config dict with fitness.weakness_lambda
        
    Returns:
        Weakness penalty (negative value)
    """
    mu = config['fitness']['weakness_lambda']
    
    if mu == 0:
        return 0.0
    
    # Count weaknesses per type
    weakness_counts = {}
    for type_name in TYPE_NAMES:
        col_name = f'type_defense_{type_name}'
        if col_name in team.columns:
            # Count Pokémon with >1.0 multiplier (weak to this type)
            weak_count = (team[col_name] > 1.0).sum()
            if weak_count > 0:
                weakness_counts[type_name] = weak_count
    
    # Compute quadratic weakness score
    weakness_score = sum((count / 6) ** 2 for count in weakness_counts.values())
    
    penalty = -mu * weakness_score
    return float(penalty)


# ============================================================================
# FITNESS COMPONENT 7: BST Cap Penalty
# ============================================================================

def compute_bst_penalty(team: pd.DataFrame, config: Dict) -> float:
    """
    Penalize teams exceeding BST cap (Base Stat Total).
    
    Soft penalty mimics competitive tier systems (VGC/Smogon):
    - Cap typically around 3300 (avg 550 = OU tier)
    - Allows 1 legendary (680 BST) + 5 regulars (~524 avg)
    - Or all OU-tier Pokemon (550 avg)
    
    Formula (soft penalty):
        if team_bst > cap:
            penalty = -(team_bst - cap) / 100 * penalty_weight
    
    Example with cap=3300:
        team_bst=3300 → penalty=0 (at cap)
        team_bst=3400 → penalty=-1.0 * penalty_weight (100 over)
        team_bst=3600 → penalty=-3.0 * penalty_weight (300 over)
    
    Args:
        team: DataFrame with hp, attack, defense, sp_attack, sp_defense, speed columns
        config: Config dict with fitness.bst_cap and fitness.bst_penalty_weight
        
    Returns:
        BST penalty (0 if under cap, negative if over cap)
    """
    cap = config['fitness'].get('bst_cap', 3300)
    penalty_weight = config['fitness'].get('bst_penalty_weight', 2.0)
    
    if cap == 0 or penalty_weight == 0:
        return 0.0
    
    # Calculate team BST
    stat_cols = ['hp', 'attack', 'defense', 'special-attack', 'special-defense', 'speed']
    team_bst = team[stat_cols].sum().sum()
    
    # Apply soft penalty if over cap
    if team_bst > cap:
        penalty = -(team_bst - cap) / 100 * penalty_weight
        return float(penalty)
    
    return 0.0


def compute_rarity_bonus(team: pd.DataFrame, config: Dict) -> float:
    """
    Reward teams with underrepresented Pokemon based on population usage.
    
    For each Pokemon in team, checks how many times it appears across
    the current population. Less frequent = higher bonus.
    
    Formula:
        For each Pokemon:
            usage_rate = count_in_population / population_size
            rarity_score = 1.0 - usage_rate  # [0, 1], higher = rarer
        
        team_rarity = average(rarity_scores)
        bonus = team_rarity * rarity_weight
    
    Example with population_size=100, rarity_weight=0.10:
        - Pokemon appears 1 time → usage=0.01 → rarity=0.99 → bonus contribution=0.099
        - Pokemon appears 50 times → usage=0.50 → rarity=0.50 → bonus contribution=0.050
        - Pokemon appears 100 times → usage=1.00 → rarity=0.00 → bonus contribution=0.000
    
    Args:
        team: DataFrame with 'name' column
        config: Config dict with optional 'pokemon_usage_counts' and 'fitness.rarity_bonus_weight'
        
    Returns:
        Rarity bonus (0.0 if disabled or no usage data, positive otherwise)
    """
    rarity_weight = config['fitness'].get('rarity_bonus_weight', 0.0)
    
    if rarity_weight == 0:
        return 0.0
    
    usage_counts = config.get('pokemon_usage_counts', {})
    population_size = config.get('population_size_tracker', 1)
    
    if not usage_counts or population_size == 0:
        return 0.0
    
    # Calculate rarity score for each team member
    rarity_scores = []
    for pokemon_name in team['name']:
        count = usage_counts.get(pokemon_name, 0)
        usage_rate = count / population_size
        rarity_score = 1.0 - usage_rate  # Higher = rarer
        rarity_scores.append(rarity_score)
    
    # Average rarity across team
    team_rarity = np.mean(rarity_scores) if rarity_scores else 0.0
    bonus = team_rarity * rarity_weight
    
    return float(bonus)


def compute_composition_bonus(team: pd.DataFrame, config: Dict) -> float:
    """
    Reward teams whose archetype counts match a target composition.

    Expected config keys:
    - fitness.composition_weight: bonus weight (0 disables)
    - fitness.target_archetype_counts: dict[str, int] summing to team size

    Score uses normalized L1 distance between actual and target counts:
        score = 1 - (sum |actual_i - target_i|) / (2 * team_size)

    Returns:
        bonus in [0, composition_weight]
    """
    fitness_cfg = config.get('fitness', {})
    weight = float(fitness_cfg.get('composition_weight', 0.0))
    target_counts = fitness_cfg.get('target_archetype_counts')

    if weight <= 0 or not isinstance(target_counts, dict) or not target_counts:
        return 0.0

    team_size = len(team)
    if team_size <= 0:
        return 0.0

    actual_counts = team['archetype'].value_counts().to_dict()
    all_keys = set(actual_counts.keys()) | set(target_counts.keys())
    l1_distance = float(sum(abs(actual_counts.get(k, 0) - int(target_counts.get(k, 0))) for k in all_keys))

    score = max(0.0, 1.0 - (l1_distance / (2.0 * team_size)))
    return float(weight * score)


def compute_pivot_bonus(team: pd.DataFrame, config: Dict) -> float:
    """Reward teams that hit a target number of pivot candidates."""
    fitness_cfg = config.get('fitness', {})
    weight = float(fitness_cfg.get('pivot_weight', 0.0))
    target_count = int(fitness_cfg.get('target_pivot_count', 0) or 0)
    threshold = float(fitness_cfg.get('pivot_threshold', 0.62))

    if weight <= 0 or target_count <= 0 or 'pivot_score' not in team.columns:
        return 0.0

    pivot_scores = pd.to_numeric(team['pivot_score'], errors='coerce').fillna(0.0).clip(lower=0.0, upper=1.0)
    candidate_count = int((pivot_scores >= threshold).sum())
    team_size = max(len(team), 1)

    count_score = max(0.0, 1.0 - (abs(candidate_count - target_count) / team_size))
    top_n = min(target_count, len(pivot_scores))
    quality_score = float(pivot_scores.nlargest(top_n).mean()) if top_n > 0 else 0.0

    return float(weight * (0.60 * count_score + 0.40 * quality_score))


# ============================================================================
# MAIN FITNESS FUNCTION
# ============================================================================

def evaluate_fitness(team: pd.DataFrame, config: Dict) -> Tuple[float, Dict]:
    """
    Evaluate total fitness for a team of 6 Pokémon.
    
    Components:
    1. Base strength (stats normalization)
    2. Type coverage (offensive matchup quality)
    3. Team synergy (physical/special balance)
    4. Entropy bonus (archetype diversity reward)
    5. Imbalance penalty (quadratic archetype stacking)
    6. Weakness penalty (quadratic shared weakness)
    
    Args:
        team: DataFrame with 6 Pokémon
        config: Configuration dict with fitness weights
        
    Returns:
        total_fitness: Scalar fitness value
        breakdown: Dict of individual component scores
    """
    # Base fitness components
    base_strength = compute_base_strength(team)
    type_coverage = compute_type_coverage(team)
    synergy = compute_synergy(team)
    
    # Diversity components
    entropy_bonus = compute_entropy_bonus(team, config)
    imbalance_penalty = compute_imbalance_penalty(team, config)
    weakness_penalty = compute_weakness_penalty(team, config)
    bst_penalty = compute_bst_penalty(team, config)
    rarity_bonus = compute_rarity_bonus(team, config)
    composition_bonus = compute_composition_bonus(team, config)
    pivot_bonus = compute_pivot_bonus(team, config)
    
    # Weighted combination
    base_fitness = (
        config['fitness']['base_stats_weight'] * base_strength +
        config['fitness']['type_coverage_weight'] * type_coverage +
        config['fitness']['synergy_weight'] * synergy
    )
    
    total_fitness = (
        base_fitness + 
        entropy_bonus + 
        imbalance_penalty + 
        weakness_penalty +
        bst_penalty +
        rarity_bonus +
        composition_bonus +
        pivot_bonus
    )
    
    # Detailed breakdown for analysis
    breakdown = {
        'total': total_fitness,
        'base_strength': base_strength,
        'type_coverage': type_coverage,
        'synergy': synergy,
        'entropy_bonus': entropy_bonus,
        'imbalance_penalty': imbalance_penalty,
        'weakness_penalty': weakness_penalty,
        'bst_penalty': bst_penalty,
        'rarity_bonus': rarity_bonus,
        'composition_bonus': composition_bonus,
        'pivot_bonus': pivot_bonus,
        'base_fitness': base_fitness
    }
    
    return float(total_fitness), breakdown


# ============================================================================
# HELPER FUNCTIONS
# ============================================================================

def get_archetype_distribution(team: pd.DataFrame) -> Dict[str, int]:
    """Get count of each archetype in team."""
    return team['archetype'].value_counts().to_dict()


def compute_archetype_entropy(team: pd.DataFrame) -> float:
    """Compute raw Shannon entropy (not normalized)."""
    archetype_counts = team['archetype'].value_counts()
    proportions = archetype_counts / len(team)
    entropy = -np.sum(proportions * np.log2(proportions))
    return float(entropy)


def count_shared_weaknesses(team: pd.DataFrame) -> Dict[str, int]:
    """Count how many Pokémon are weak to each type."""
    weaknesses = {}
    for type_name in TYPE_NAMES:
        col_name = f'type_defense_{type_name}'
        if col_name in team.columns:
            weak_count = (team[col_name] > 1.0).sum()
            if weak_count > 1:  # Only report shared weaknesses
                weaknesses[type_name] = weak_count
    return weaknesses


if __name__ == "__main__":
    print("GA Fitness Module - Type Chart Loaded")
    print(f"Type chart shape: {TYPE_CHART.shape}")
    print(f"Available types: {', '.join(TYPE_NAMES)}")
    
    # Test type effectiveness
    print("\nExample type effectiveness:")
    print(f"  Fire vs Grass/Steel: {get_type_effectiveness('fire', ['grass', 'steel']):.1f}x")
    print(f"  Water vs Fire: {get_type_effectiveness('water', ['fire']):.1f}x")
    print(f"  Electric vs Ground: {get_type_effectiveness('electric', ['ground']):.1f}x")
