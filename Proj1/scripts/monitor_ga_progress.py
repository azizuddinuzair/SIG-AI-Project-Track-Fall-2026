"""
Monitor diversity-enhanced GA progress in real-time.
"""

import json
from pathlib import Path
import time

def monitor_ga_progress(results_dir: str, poll_interval: int = 30):
    """Monitor GA generation progress."""
    
    results_path = Path(results_dir)
    
    if not results_path.exists():
        print(f"Waiting for GA results directory: {results_dir}")
        return
    
    print(f"\n{'='*80}")
    print("MONITORING DIVERSITY-ENHANCED GA PROGRESS")
    print(f"{'='*80}")
    print(f"Results directory: {results_dir}\n")
    
    while True:
        # Find all generation elite files
        elite_files = list(results_path.glob("generation_elite_*.json"))
        
        if elite_files:
            # Get latest generation
            generations = sorted([int(f.stem.split('_')[-1]) for f in elite_files])
            latest_gen = generations[-1]
            
            # Count unique Pokemon in this generation
            with open(results_path / f"generation_elite_{latest_gen}.json", 'r') as f:
                elite_teams = json.load(f)
            
            all_pokemon = []
            for team in elite_teams:
                all_pokemon.extend([p['name'] for p in team['pokemon']])
            
            unique_count = len(set(all_pokemon))
            timestamp = time.strftime("%H:%M:%S")
            
            print(f"[{timestamp}] Gen {latest_gen:3d}: {unique_count:3d} unique Pokemon in elite teams")
            
            # Check if completed
            if latest_gen >= 300:
                print("\n✅ GA COMPLETE!")
                
                # Load final results
                with open(results_path / "top_10_teams.json", 'r') as f:
                    top_teams = json.load(f)
                
                # Count total unique Pokemon across all teams in final population
                # (would need to check fitness_history.csv or regenerate)
                print(f"\nFinal fitness history:")
                
                import pandas as pd
                if (results_path / "fitness_history.csv").exists():
                    hist_df = pd.read_csv(results_path / "fitness_history.csv")
                    print(f"  Final generation: {hist_df.iloc[-1]['generation']}")
                    print(f"  Final mean fitness: {hist_df.iloc[-1]['mean_fitness']:.4f}")
                    print(f"  Final max fitness: {hist_df.iloc[-1]['max_fitness']:.4f}")
                
                break
        
        time.sleep(poll_interval)

if __name__ == "__main__":
    import sys
    results_dir = sys.argv[1] if len(sys.argv) > 1 else "reports/ga_results/run_diversity_enhanced_20260305_142640"
    monitor_ga_progress(results_dir, poll_interval=10)
