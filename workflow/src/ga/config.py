"""
GA Configuration Management

Provides configuration templates for ablation study:
- Config A: Uniform initialization, no diversity bonuses
- Config B: Inverse-weighted initialization, no diversity bonuses
- Config C: Sqrt-weighted initialization with diversity bonuses (recommended)

All configurations use the same random seed for fair comparison.
"""

import copy


def get_base_config():
    """Base configuration shared across all experiments."""
    return {
        "population": {
            "size": 150,
            "generations": 250,
            "tournament_k": 3,
            "elitism": 5  # Keep top 5 individuals each generation
        },
        "fitness": {
            # Base fitness component weights (should sum to ~0.70-0.85)
            "base_stats_weight": 0.40,
            "type_coverage_weight": 0.30,
            "synergy_weight": 0.15,
            
            # Diversity/penalty weights (remaining 0.15-0.30)
            "diversity_weight": 0.15,      # Entropy bonus coefficient
            "imbalance_lambda": 0.20,      # Quadratic balance penalty
            "weakness_lambda": 0.10,       # Quadratic weakness penalty
            
            # BST cap (competitive tier constraint)
            "bst_cap": 3300,               # Total team BST limit (~550 avg = OU tier)
            "bst_penalty_weight": 2.0,     # Penalty scaling: -(excess/100) * weight

            # Optional CLI-side pivot pressure bonus
            "pivot_weight": 0.0,
            "target_pivot_count": 0,
            "pivot_threshold": 0.62,
        },
        "initialization": {
            "method": "uniform"  # uniform | inverse | sqrt_weighted
        },
        "mutation": {
            "rate": 0.15,
            "weighted": False  # Use weighted archetype selection during mutation
        },
        "crossover": {
            "rate": 0.80,
            "type": "two_point"
        },
        "random_seed": 42
    }


def get_config_a():
    """
    Config A: Baseline
    - Uniform random initialization
    - No diversity bonuses or penalties
    - Pure fitness optimization
    """
    config = get_base_config()
    config["name"] = "ConfigA_Baseline"
    config["initialization"]["method"] = "uniform"
    config["mutation"]["weighted"] = False
    
    # Disable diversity components
    config["fitness"]["diversity_weight"] = 0.0
    config["fitness"]["imbalance_lambda"] = 0.0
    config["fitness"]["weakness_lambda"] = 0.0
    
    return config


def get_config_b():
    """
    Config B: Weighted initialization only
    - Inverse-weighted archetype initialization
    - No diversity bonuses in fitness
    - Tests initialization impact alone
    """
    config = get_base_config()
    config["name"] = "ConfigB_InverseWeighted"
    config["initialization"]["method"] = "inverse"
    config["mutation"]["weighted"] = True
    
    # Disable diversity bonuses but keep weakness penalty
    config["fitness"]["diversity_weight"] = 0.0
    config["fitness"]["imbalance_lambda"] = 0.0
    # Keep weakness penalty (team composition quality)
    config["fitness"]["weakness_lambda"] = 0.10
    
    return config


def get_config_c():
    """
    Config C: Full optimization (RECOMMENDED)
    - Sqrt-weighted initialization + mutation
    - Full diversity bonuses and penalties
    - Entropy-based diversity reward
    - Quadratic balance penalty
    - Quadratic weakness penalty
    """
    config = get_base_config()
    config["name"] = "ConfigC_Full"
    config["initialization"]["method"] = "sqrt_weighted"
    config["mutation"]["weighted"] = True
    
    # All diversity components enabled (default values from base_config)
    return config


def get_config_random():
    """
    Config Random: Fun, inclusive team generation
    - Loosened fitness constraints to include underrepresented Pokemon
    - Disables BST penalty (allows weak Pokemon like Magikarp)
    - Reduces base_stats emphasis (0.05 instead of 0.40)
    - Reduces archetype imbalance penalty
    - Keeps type coverage and synergy for team quality
    - Keeps entropy diversity bonus
    - **Enables rarity bonus** to reward underused Pokemon
    
    Purpose: Generate creative, diverse teams without strict optimization
    """
    config = get_base_config()
    config["name"] = "ConfigRandom_Inclusive"
    config["initialization"]["method"] = "sqrt_weighted"
    config["mutation"]["weighted"] = True
    
    # Loosen fitness constraints
    config["fitness"]["base_stats_weight"] = 0.05      # Way down from 0.40
    config["fitness"]["bst_penalty_weight"] = 0.0      # Disable BST cap entirely
    
    # Keep type coverage and synergy for team coherence
    config["fitness"]["type_coverage_weight"] = 0.30   # Keep for quality
    config["fitness"]["synergy_weight"] = 0.15         # Keep for quality
    
    # Soften penalties
    config["fitness"]["imbalance_lambda"] = 0.10       # Down from 0.20
    config["fitness"]["weakness_lambda"] = 0.05        # Down from 0.10
    
    # Enable rarity bonus to promote underused Pokemon
    config["fitness"]["rarity_bonus_weight"] = 0.15    # New! Rewards rare Pokemon
    
    return config


def get_all_configs():
    """Return all three ablation configurations."""
    return {
        "A": get_config_a(),
        "B": get_config_b(),
        "C": get_config_c(),
        "Random": get_config_random()
    }


def modify_config(config, **kwargs):
    """
    Modify a configuration with custom parameters.
    
    Example:
        config = get_config_c()
        config = modify_config(config, 
                              diversity_weight=0.20,
                              random_seed=123)
    """
    config_copy = copy.deepcopy(config)
    
    # Handle nested updates
    for key, value in kwargs.items():
        if "." in key:
            # Handle nested keys like "fitness.diversity_weight"
            parts = key.split(".")
            target = config_copy
            for part in parts[:-1]:
                target = target[part]
            target[parts[-1]] = value
        else:
            # Handle top-level or search all nested dicts
            updated = False
            for section_key, section in config_copy.items():
                if isinstance(section, dict) and key in section:
                    section[key] = value
                    updated = True
                    break
            if not updated:
                config_copy[key] = value
    
    return config_copy


def validate_config(config):
    """
    Validate configuration parameters.
    
    Checks:
    - Fitness weights are reasonable
    - Population parameters are positive
    - Mutation/crossover rates in [0, 1]
    """
    errors = []
    
    # Check population parameters
    if config["population"]["size"] < 10:
        errors.append("Population size too small (min 10)")
    if config["population"]["generations"] < 10:
        errors.append("Generations too small (min 10)")
    
    # Check rates
    if not 0 <= config["mutation"]["rate"] <= 1:
        errors.append("Mutation rate must be in [0, 1]")
    if not 0 <= config["crossover"]["rate"] <= 1:
        errors.append("Crossover rate must be in [0, 1]")
    
    # Check fitness weights
    fitness_sum = (
        config["fitness"]["base_stats_weight"] +
        config["fitness"]["type_coverage_weight"] +
        config["fitness"]["synergy_weight"] +
        config["fitness"]["diversity_weight"]
    )
    if fitness_sum > 1.2 or fitness_sum < 0.5:
        errors.append(f"Fitness weights sum to {fitness_sum:.2f} (expected ~0.7-1.0)")
    
    if errors:
        raise ValueError("Config validation errors:\n" + "\n".join(f"  - {e}" for e in errors))
    
    return True


if __name__ == "__main__":
    # Test configuration generation
    print("=" * 80)
    print("GA CONFIGURATION TEMPLATES")
    print("=" * 80)
    
    for name, config in get_all_configs().items():
        print(f"\n{config['name']}:")
        print(f"  Initialization: {config['initialization']['method']}")
        print(f"  Mutation weighted: {config['mutation']['weighted']}")
        print(f"  Diversity weight: {config['fitness']['diversity_weight']}")
        print(f"  Imbalance lambda: {config['fitness']['imbalance_lambda']}")
        print(f"  Weakness lambda: {config['fitness']['weakness_lambda']}")
        
        try:
            validate_config(config)
            print(f"  ✓ Validation passed")
        except ValueError as e:
            print(f"  ✗ Validation failed: {e}")
