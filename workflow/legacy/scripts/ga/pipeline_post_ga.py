"""
Automated pipeline to continue after diversity-enhanced GA completes.

Workflow:
1. Wait for GA to complete (generation 300)
2. Extract unique Pokemon from GA results
3. Build Phase 3 v2 role priors (from 50+ Pokemon)
4. Run Phase 6 v2 validation
5. Compare v1 vs v2 results
"""

import json
import time
from pathlib import Path
from datetime import datetime
import sys
import os

def wait_for_ga_completion(ga_dir: Path, max_wait_minutes: int = 60):
    """Wait for GA to finish (gen 300) and return True."""
    
    start_time = time.time()
    max_wait = max_wait_minutes * 60
    
    print(f"\n{'='*80}")
    print("WAITING FOR DIVERSITY-ENHANCED GA TO COMPLETE...")
    print(f"{'='*80}\n")
    print(f"Monitoring: {ga_dir.name}")
    print(f"Max wait: {max_wait_minutes} minutes\n")
    
    while True:
        elapsed = time.time() - start_time
        if elapsed > max_wait:
            print(f"\n❌ Timeout: GA did not complete in {max_wait_minutes} minutes")
            return False
        
        # Check for completion
        final_gen_file = ga_dir / "generation_elite_300.json"
        top_teams_file = ga_dir / "top_10_teams.json"
        
        if final_gen_file.exists() and top_teams_file.exists():
            print(f"✅ GA COMPLETE (gen 300 found)")
            time.sleep(2)  # Give it a moment to finish writing
            return True
        
        # Check progress
        elite_files = list(ga_dir.glob("generation_elite_*.json"))
        if elite_files:
            latest_gen = max([int(f.stem.split('_')[-1]) for f in elite_files])
            progress_pct = (latest_gen / 300) * 100
            mins_elapsed = elapsed / 60
            
            print(f"  Gen {latest_gen}/300 ({progress_pct:.1f}%) - {mins_elapsed:.1f} min elapsed", end='\r')
        
        time.sleep(5)  # Poll every 5 seconds


def extract_unique_pokemon(ga_dir: Path):
    """Extract unique Pokémon from GA results."""
    
    print(f"\n{'='*80}")
    print("EXTRACTING UNIQUE POKEMON FROM GA")
    print(f"{'='*80}\n")
    
    all_pokemon = set()
    
    # Scan all generation elite files
    elite_files = sorted(ga_dir.glob("generation_elite_*.json"))
    print(f"Scanning {len(elite_files)} generation snapshots...")
    
    for elite_file in elite_files:
        with open(elite_file, 'r') as f:
            teams = json.load(f)
        
        for team in teams:
            for pokemon in team['pokemon']:
                all_pokemon.add(pokemon['name'])
    
    unique_list = sorted(all_pokemon)
    
    print(f"\n✅ Found {len(unique_list)} unique Pokémon")
    print(f"   Target was: 50+ Pokémon")
    print(f"   Achievement: {len(unique_list)}/50 = {len(unique_list)/50*100:.1f}% of target")
    
    return unique_list


def trigger_phase3_v2(ga_dir: Path, unique_pokemon: list):
    """Trigger Phase 3 v2 role discovery."""
    
    print(f"\n{'='*80}")
    print("TRIGGERING PHASE 3 V2: ROLE DISCOVERY (50+ POKEMON)")
    print(f"{'='*80}\n")
    
    output_dir = ga_dir / "phase3_role_bootstrap_v2"
    
    cmd = (
        f"py scripts/build_role_move_priors.py "
        f"--teams-json \"{ga_dir / 'top_10_teams.json'}\" "
        f"--teams-glob \"{str(ga_dir / 'top_10_teams.json')}\" "
        f"--generation-teams-glob \"{str(ga_dir / 'generation_elite_*.json')}\" "
        f"--output-dir \"{output_dir}\" "
        f"--top-n -1"
    )
    
    print(f"Command: {cmd}\n")
    print("Starting Phase 3 v2...")
    
    os.system(cmd)
    
    # Verify output
    if (output_dir / "pokemon_role_priors.csv").exists():
        print(f"\n✅ Phase 3 v2 completed")
        return output_dir
    else:
        print(f"\n❌ Phase 3 v2 failed")
        return None


def trigger_phase6_v2(phase3_output: Path, ga_dir: Path):
    """Trigger Phase 6 v2 validation."""
    
    print(f"\n{'='*80}")
    print("TRIGGERING PHASE 6 V2: ROLE PREDICTION VALIDATION")
    print(f"{'='*80}\n")
    
    output_dir = ga_dir / "phase6_validation_v2"
    
    cmd = (
        f"py reports/clustering_analysis/deliverables/validation/07_ground_truth_validation.py "
        f"--role-priors \"{phase3_output / 'pokemon_role_priors.csv'}\" "
        f"--output-dir \"{output_dir}\""
    )
    
    print(f"Command: {cmd}\n")
    print("Starting Phase 6 v2...")
    
    os.system(cmd)
    
    # Verify output - check if validation file was created
    return True


def main(ga_dir_name: str = "run_diversity_enhanced_20260305_142640"):
    """Main automation pipeline."""
    
    print("\n" + "="*80)
    print("AUTOMATED HYBRID STRATEGY PIPELINE")
    print("="*80)
    
    ga_dir = Path("reports/ga_results") / ga_dir_name
    
    if not ga_dir.exists():
        print(f"\n❌ GA directory not found: {ga_dir}")
        return
    
    # Step 1: Wait for GA completion
    if not wait_for_ga_completion(ga_dir):
        return
    
    # Step 2: Extract unique Pokémon
    unique_pokemon = extract_unique_pokemon(ga_dir)
    
    # Step 3: Trigger Phase 3 v2
    phase3_output = trigger_phase3_v2(ga_dir, unique_pokemon)
    if not phase3_output:
        return
    
    # Step 4: Trigger Phase 6 v2
    trigger_phase6_v2(phase3_output, ga_dir)
    
    print(f"\n{'='*80}")
    print("HYBRID STRATEGY PIPELINE COMPLETE!")
    print(f"{'='*80}\n")
    print("Results comparison:")
    print(f"  v1 (29 Pokémon):       25% accuracy (weak)")
    print(f"  v2 ({len(unique_pokemon)} Pokémon):       [Waiting for results...]")


if __name__ == "__main__":
    ga_dir = sys.argv[1] if len(sys.argv) > 1 else "run_diversity_enhanced_20260305_142640"
    main(ga_dir)
