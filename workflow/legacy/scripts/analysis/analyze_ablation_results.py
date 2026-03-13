"""
Ablation Study Results Analysis

Analyzes and visualizes GA ablation study results:
- Convergence plots (fitness over generations)
- Archetype distribution comparison
- Rare archetype representation
- Pareto front visualization (multi-objective trade-offs)
- Statistical significance testing
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Dict, List
from scipy import stats


sns.set_style('whitegrid')
sns.set_palette('husl')


def load_ablation_results(run_dir: Path) -> Dict[str, pd.DataFrame]:
    """
    Load all results from an ablation study run.
    
    Args:
        run_dir: Directory containing ablation study results
        
    Returns:
        Dict mapping config name to fitness history DataFrame
    """
    results = {}
    
    for config_name in ['ConfigA_Baseline', 'ConfigB_InverseWeighted', 'ConfigC_Full']:
        history_file = run_dir / f"{config_name}_fitness_history.csv"
        if history_file.exists():
            results[config_name] = pd.read_csv(history_file)
    
    return results


def plot_convergence(results: Dict[str, pd.DataFrame], output_dir: Path):
    """
    Plot fitness convergence over generations for all configurations.
    
    Args:
        results: Dict of config name → fitness history
        output_dir: Directory to save plots
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    # Plot 1: Mean fitness over generations
    ax = axes[0, 0]
    for config_name, history in results.items():
        label = config_name.replace('Config', '').replace('_', ' ')
        ax.plot(history['generation'], history['mean_fitness'], 
                label=label, linewidth=2, alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Fitness')
    ax.set_title('Mean Fitness Convergence')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 2: Max fitness over generations
    ax = axes[0, 1]
    for config_name, history in results.items():
        label = config_name.replace('Config', '').replace('_', ' ')
        ax.plot(history['generation'], history['max_fitness'], 
                label=label, linewidth=2, alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Max Fitness')
    ax.set_title('Best Team Fitness')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 3: Fitness std dev (diversity measure)
    ax = axes[1, 0]
    for config_name, history in results.items():
        label = config_name.replace('Config', '').replace('_', ' ')
        ax.plot(history['generation'], history['std_fitness'], 
                label=label, linewidth=2, alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Fitness Std Dev')
    ax.set_title('Population Diversity')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Plot 4: Mean entropy (archetype diversity)
    ax = axes[1, 1]
    for config_name, history in results.items():
        label = config_name.replace('Config', '').replace('_', ' ')
        ax.plot(history['generation'], history['mean_entropy'], 
                label=label, linewidth=2, alpha=0.8)
    ax.set_xlabel('Generation')
    ax.set_ylabel('Mean Archetype Entropy')
    ax.set_title('Archetype Diversity Over Time')
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plot_path = output_dir / 'convergence_comparison.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved convergence plot: {plot_path}")
    plt.close()


def plot_archetype_distribution(results: Dict[str, pd.DataFrame], output_dir: Path):
    """
    Plot final archetype distribution for each configuration.
    
    Args:
        results: Dict of config name → fitness history
        output_dir: Directory to save plots
    """
    # Extract final generation archetype counts
    archetype_data = []
    
    for config_name, history in results.items():
        final_gen = history.iloc[-1]
        
        # Find archetype columns
        arch_cols = [col for col in history.columns if col.startswith('archetype_')]
        
        for col in arch_cols:
            archetype_name = col.replace('archetype_', '').replace('_', ' ').title()
            count = final_gen[col]
            archetype_data.append({
                'Config': config_name.replace('Config', '').replace('_', ' '),
                'Archetype': archetype_name,
                'Count': count
            })
    
    df = pd.DataFrame(archetype_data)
    
    # Create grouped bar plot
    fig, ax = plt.subplots(figsize=(12, 6))
    
    configs = df['Config'].unique()
    archetypes = df['Archetype'].unique()
    x = np.arange(len(archetypes))
    width = 0.25
    
    for i, config in enumerate(configs):
        config_data = df[df['Config'] == config]
        counts = [config_data[config_data['Archetype'] == arch]['Count'].values[0] 
                  if len(config_data[config_data['Archetype'] == arch]) > 0 else 0
                  for arch in archetypes]
        ax.bar(x + i * width, counts, width, label=config, alpha=0.8)
    
    ax.set_xlabel('Archetype')
    ax.set_ylabel('Count in Population (out of 900 total Pokémon)')
    ax.set_title('Archetype Distribution by Configuration (Final Generation)')
    ax.set_xticks(x + width)
    ax.set_xticklabels(archetypes, rotation=45, ha='right')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plot_path = output_dir / 'archetype_distribution.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved archetype distribution: {plot_path}")
    plt.close()


def plot_rare_archetype_trends(results: Dict[str, pd.DataFrame], output_dir: Path):
    """
    Plot rare archetype representation over generations.
    
    Args:
        results: Dict of config name → fitness history
        output_dir: Directory to save plots
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    for config_name, history in results.items():
        label = config_name.replace('Config', '').replace('_', ' ')
        ax.plot(history['generation'], history['rare_archetype_percent'], 
                label=label, linewidth=2, alpha=0.8, marker='o', markersize=3, markevery=25)
    
    ax.set_xlabel('Generation')
    ax.set_ylabel('% of Teams with Rare Archetype')
    ax.set_title('Rare Archetype Representation (Speed Sweeper, Defensive Wall)')
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.axhline(y=50, color='gray', linestyle='--', alpha=0.5, label='50% threshold')
    
    plt.tight_layout()
    plot_path = output_dir / 'rare_archetype_trends.png'
    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"✓ Saved rare archetype trends: {plot_path}")
    plt.close()


def compute_statistical_tests(results: Dict[str, pd.DataFrame], output_dir: Path):
    """
    Perform statistical significance tests on final fitness values.
    
    Tests:
    - Welch's t-test (unequal variances)
    - Effect size (Cohen's d)
    
    Args:
        results: Dict of config name → fitness history
        output_dir: Directory to save results
    """
    print("\n" + "=" * 80)
    print("STATISTICAL SIGNIFICANCE TESTS")
    print("=" * 80)
    
    configs = list(results.keys())
    final_fitness = {name: history.iloc[-20:]['mean_fitness'].values 
                     for name, history in results.items()}
    
    comparisons = []
    
    # Pairwise comparisons
    for i in range(len(configs)):
        for j in range(i + 1, len(configs)):
            config1 = configs[i]
            config2 = configs[j]
            
            fitness1 = final_fitness[config1]
            fitness2 = final_fitness[config2]
            
            # Welch's t-test
            t_stat, p_value = stats.ttest_ind(fitness1, fitness2, equal_var=False)
            
            # Cohen's d (effect size)
            mean_diff = np.mean(fitness1) - np.mean(fitness2)
            pooled_std = np.sqrt((np.std(fitness1)**2 + np.std(fitness2)**2) / 2)
            cohens_d = mean_diff / pooled_std if pooled_std > 0 else 0
            
            comparisons.append({
                'Config 1': config1.replace('Config', '').replace('_', ' '),
                'Config 2': config2.replace('Config', '').replace('_', ' '),
                'Mean Diff': mean_diff,
                't-statistic': t_stat,
                'p-value': p_value,
                "Cohen's d": cohens_d,
                'Significant (α=0.05)': 'Yes' if p_value < 0.05 else 'No'
            })
            
            print(f"\n{config1} vs {config2}:")
            print(f"  Mean difference: {mean_diff:+.6f}")
            print(f"  t-statistic: {t_stat:.4f}")
            print(f"  p-value: {p_value:.6f} {'***' if p_value < 0.001 else '**' if p_value < 0.01 else '*' if p_value < 0.05 else ''}")
            print(f"  Cohen's d: {cohens_d:.4f} ({'large' if abs(cohens_d) > 0.8 else 'medium' if abs(cohens_d) > 0.5 else 'small'})")
    
    # Save to CSV
    comparison_df = pd.DataFrame(comparisons)
    stats_path = output_dir / 'statistical_tests.csv'
    comparison_df.to_csv(stats_path, index=False)
    print(f"\n✓ Saved statistical tests: {stats_path}")


def generate_summary_report(run_dir: Path):
    """
    Generate comprehensive summary report with all analyses.
    
    Args:
        run_dir: Directory containing ablation study results
    """
    print("\n" + "=" * 80)
    print("GENERATING ANALYSIS REPORT")
    print("=" * 80)
    
    # Load results
    print("\n📂 Loading results...")
    results = load_ablation_results(run_dir)
    
    if not results:
        print("❌ No results found in directory!")
        return
    
    print(f"   Found {len(results)} configurations")
    
    # Create plots
    print("\n📊 Generating visualizations...")
    plot_convergence(results, run_dir)
    plot_archetype_distribution(results, run_dir)
    plot_rare_archetype_trends(results, run_dir)
    
    # Statistical tests
    compute_statistical_tests(results, run_dir)
    
    # Generate text summary
    print("\n📝 Writing summary report...")
    report_path = run_dir / 'analysis_report.txt'
    
    with open(report_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("GA ABLATION STUDY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        for config_name, history in results.items():
            f.write(f"\n{config_name}:\n")
            f.write("-" * 40 + "\n")
            
            final = history.iloc[-1]
            initial = history.iloc[0]
            
            f.write(f"  Initial fitness: {initial['mean_fitness']:.4f}\n")
            f.write(f"  Final fitness: {final['mean_fitness']:.4f}\n")
            f.write(f"  Improvement: {final['mean_fitness'] - initial['mean_fitness']:+.4f}\n")
            f.write(f"  Final max fitness: {final['max_fitness']:.4f}\n")
            f.write(f"  Final entropy: {final['mean_entropy']:.4f}\n")
            f.write(f"  Rare archetypes: {final['rare_archetype_percent']:.1f}%\n")
    
    print(f"✓ Saved analysis report: {report_path}")
    
    print("\n" + "=" * 80)
    print("✅ ANALYSIS COMPLETE")
    print("=" * 80)
    # Print relative path from project root
    proj_root = Path(__file__).resolve().parents[2]
    try:
        rel_path = run_dir.relative_to(proj_root)
        print(f"All outputs saved to: {rel_path}")
    except ValueError:
        print(f"All outputs saved to: {run_dir}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Analyze GA ablation study results")
    parser.add_argument('run_dir', type=str, help='Path to ablation study results directory')
    
    args = parser.parse_args()
    
    run_dir = Path(args.run_dir)
    
    if not run_dir.exists():
        print(f"❌ Directory not found: {run_dir}")
        exit(1)
    
    generate_summary_report(run_dir)
