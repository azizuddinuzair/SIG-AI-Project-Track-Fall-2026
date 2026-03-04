"""
Ablation Study Runner

Runs all three GA configurations (A, B, C) for comparative analysis:
- Config A: Baseline (uniform, no diversity bonuses)
- Config B: Inverse-weighted initialization
- Config C: Full optimization (sqrt-weighted + diversity bonuses)

Saves results for each configuration and generates comparison summary.
"""

import sys
from pathlib import Path
import pandas as pd
import time
from datetime import datetime

# Add src/models to path
current_file = Path(__file__).resolve()
models_dir = current_file.parents[1] / "src" / "models"
sys.path.insert(0, str(models_dir))

from ga_optimization import PokemonGA, load_pokemon_data
from ga_config import get_all_configs


def run_single_config(config: dict, pokemon_df: pd.DataFrame, output_dir: Path) -> dict:
    """
    Run GA with a single configuration.
    
    Args:
        config: Configuration dictionary
        pokemon_df: Pokémon dataset
        output_dir: Directory to save results
        
    Returns:
        Summary statistics dict
    """
    config_name = config['name']
    print("\n" + "=" * 80)
    print(f"RUNNING: {config_name}")
    print("=" * 80)
    
    start_time = time.time()
    
    # Initialize and run GA
    ga = PokemonGA(pokemon_df, config)
    history = ga.run()
    
    # Export results
    ga.export_results(output_dir)
    
    elapsed = time.time() - start_time
    
    # Extract summary statistics
    final_gen = history.iloc[-1]
    best_teams = ga.get_best_teams(10)
    
    summary = {
        'config': config_name,
        'initialization': config['initialization']['method'],
        'diversity_weight': config['fitness']['diversity_weight'],
        'imbalance_lambda': config['fitness']['imbalance_lambda'],
        'weakness_lambda': config['fitness']['weakness_lambda'],
        'final_mean_fitness': final_gen['mean_fitness'],
        'final_max_fitness': final_gen['max_fitness'],
        'final_std_fitness': final_gen['std_fitness'],
        'final_mean_entropy': final_gen['mean_entropy'],
        'final_rare_archetype_percent': final_gen['rare_archetype_percent'],
        'best_team_fitness': best_teams[0][1],
        'convergence_generation': _find_convergence(history),
        'elapsed_time_seconds': elapsed
    }
    
    print(f"\n✅ {config_name} complete in {elapsed:.1f}s")
    print(f"   Final fitness: {summary['final_mean_fitness']:.4f} (max: {summary['final_max_fitness']:.4f})")
    print(f"   Convergence: Generation {summary['convergence_generation']}")
    
    return summary


def _find_convergence(history: pd.DataFrame, window: int = 20, threshold: float = 0.001) -> int:
    """
    Find generation where fitness converged (stabilized).
    
    Convergence defined as: std(fitness) < threshold for `window` generations.
    
    Args:
        history: Fitness history DataFrame
        window: Lookback window
        threshold: Std dev threshold for convergence
        
    Returns:
        Generation number where convergence occurred (or last generation)
    """
    if len(history) < window:
        return len(history) - 1
    
    for i in range(window, len(history)):
        recent_fitness = history.iloc[i-window:i]['mean_fitness']
        if recent_fitness.std() < threshold:
            return i
    
    return len(history) - 1  # Never converged


def run_ablation_study(output_dir: Path = None):
    """
    Run complete ablation study with all three configurations.
    
    Args:
        output_dir: Directory to save results (default: Proj1/reports/ga_results/)
    """
    # Setup output directory
    if output_dir is None:
        proj_root = Path(__file__).resolve().parents[1]
        output_dir = proj_root / "reports" / "ga_results"
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    run_dir = output_dir / f"ablation_study_{timestamp}"
    run_dir.mkdir(exist_ok=True)
    
    print("=" * 80)
    print("GENETIC ALGORITHM ABLATION STUDY")
    print("=" * 80)
    print(f"Output directory: {run_dir}")
    print(f"Timestamp: {timestamp}")
    
    # Load Pokémon data
    print("\n📂 Loading data...")
    pokemon_df = load_pokemon_data()
    
    # Get all configurations
    configs = get_all_configs()
    
    # Run each configuration
    summaries = []
    for config_key in ['A', 'B', 'C']:
        config = configs[config_key]
        summary = run_single_config(config, pokemon_df, run_dir)
        summaries.append(summary)
    
    # Create comparison summary
    print("\n" + "=" * 80)
    print("ABLATION STUDY SUMMARY")
    print("=" * 80)
    
    summary_df = pd.DataFrame(summaries)
    
    # Save summary
    summary_path = run_dir / "ablation_summary.csv"
    summary_df.to_csv(summary_path, index=False)
    print(f"\n✓ Saved summary: {summary_path}")
    
    # Print comparison table
    print("\n" + "-" * 80)
    print("CONFIGURATION COMPARISON")
    print("-" * 80)
    print(summary_df[['config', 'initialization', 'final_mean_fitness', 'final_max_fitness', 
                       'final_mean_entropy', 'convergence_generation']].to_string(index=False))
    
    # Compute relative improvements
    baseline_fitness = summary_df.loc[summary_df['config'] == 'ConfigA_Baseline', 'final_mean_fitness'].values[0]
    
    print("\n" + "-" * 80)
    print("RELATIVE IMPROVEMENTS OVER BASELINE")
    print("-" * 80)
    for _, row in summary_df.iterrows():
        if row['config'] == 'ConfigA_Baseline':
            continue
        improvement = 100 * (row['final_mean_fitness'] - baseline_fitness) / baseline_fitness
        print(f"{row['config']:30s}: {improvement:+6.2f}% fitness improvement")
    
    print("\n" + "=" * 80)
    print("✅ ABLATION STUDY COMPLETE")
    print("=" * 80)
    print(f"All results saved to: {run_dir}")
    
    return summary_df, run_dir


def run_quick_test():
    """
    Quick test run with reduced population and generations.
    """
    print("=" * 80)
    print("QUICK TEST RUN (Reduced parameters)")
    print("=" * 80)
    
    # Load data
    pokemon_df = load_pokemon_data()
    
    # Get Config C (full optimization)
    from ga_config import get_config_c, modify_config
    
    config = get_config_c()
    config = modify_config(config, size=30, generations=20)
    
    # Run
    ga = PokemonGA(pokemon_df, config)
    history = ga.run()
    
    # Show best team
    print("\n" + "=" * 80)
    print("BEST TEAM FROM TEST RUN")
    print("=" * 80)
    best_team, fitness, breakdown = ga.get_best_teams(1)[0]
    print(best_team[['name', 'archetype', 'type1', 'type2', 'offensive_index', 'defensive_index']])
    print(f"\nFitness: {fitness:.4f}")
    print("\nBreakdown:")
    for key, value in breakdown.items():
        if key != 'total':
            print(f"  {key:25s}: {value:7.4f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Run GA ablation study")
    parser.add_argument('--quick', action='store_true', help='Run quick test (30 pop, 20 gen)')
    parser.add_argument('--output', type=str, default=None, help='Output directory')
    
    args = parser.parse_args()
    
    if args.quick:
        run_quick_test()
    else:
        output_dir = Path(args.output) if args.output else None
        run_ablation_study(output_dir)
