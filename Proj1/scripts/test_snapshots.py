#!/usr/bin/env python3
"""Quick test of generation snapshot functionality"""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent / "src" / "models"))

from ga_optimization import PokemonGA, load_pokemon_data
from ga_config import get_config_c
from datetime import datetime

def main():
    print("=" * 80)
    print("TESTING GENERATION SNAPSHOTS")
    print("=" * 80)
    
    # Load data
    print("\n[DATA] Loading Pokemon...")
    pokemon_df = load_pokemon_data()
    print(f"   Loaded {len(pokemon_df)} Pokemon")
    
    # Configure for small test (25 gens)
    print("\n[CONFIG] Setting up small test GA...")
    config = get_config_c()
    config['population']['size'] = 30
    config['population']['generations'] = 25
    config['name'] = "Snapshot Test"
    
    # Create output directory
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_dir = Path(__file__).parent.parent / "reports" / "ga_results" / f"test_snapshots_{timestamp}"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"   Output: {output_dir}")
    print(f"   Gens: {config['population']['generations']}, Pop size: {config['population']['size']}")
    
    # Run GA with snapshots
    print("\n[GA] Running evolution with snapshot export...")
    ga = PokemonGA(pokemon_df, config, output_dir=output_dir)
    ga.run()
    
    # Check for snapshot files
    print("\n[CHECK] Verifying snapshot files...")
    snapshots = sorted(output_dir.glob("generation_elite_*.json"))
    print(f"   Found {len(snapshots)} snapshot files:")
    for snap in snapshots:
        size = snap.stat().st_size
        print(f"     - {snap.name} ({size} bytes)")
    
    if snapshots:
        print("\n[SUCCESS] Generation snapshots working correctly!")
        return 0
    else:
        print("\n[ERROR] No snapshot files created!")
        return 1

if __name__ == "__main__":
    sys.exit(main())
