# Next Validation Experiments

## Three Critical Tests to Validate (or Invalidate) ConfigC

Your skepticism identified a major gap in the ablation study: **we only ran ONE seed per configuration**. Without multiple seeds, one run could be luck. Without entropy ablation, we can't separate cause from effect.

---

## Experiment 1: Multiple Seeds (Robustness Test)

### Why This Matters
If ConfigC always finds the same team across different random initializations → it's a real attractor
If different seeds find different teams → it's luck, and fitness landscape has multiple optima

### Command
```bash
python run_ablation_study.py \
  --configs C \
  --population 150 \
  --generations 250 \
  --runs 5 \
  --seeds 42 123 456 789 999
```

### What to Check
```
For each run:
  1. What's the best team?
  2. Is it always: baxcalibur, groudon, flutter-mane, stakataka, miraidon, mewtwo?
  3. What's the final fitness?
  4. Do they converge at the same generation?
  
Expected outcomes:

SCENARIO A: All identical
  ✓ Run 1: fitness=0.7324, team=X
  ✓ Run 2: fitness=0.7324, team=X
  ✓ ...Run 5: fitness=0.7324, team=X
  
  → ConfigC found a strong attractor
  → Conclusion: "Diversity IS valuable; GA converges here repeatedly"

SCENARIO B: High variance
  ⚠ Run 1: fitness=0.7324, team=X
  ⚠ Run 2: fitness=0.7200, team=Y
  ⚠ Run 3: fitness=0.7150, team=Z
  ⚠ ...Run 5: fitness=0.6900, team=W
  
  → Landscape has multiple peaks
  → Conclusion: "One lucky run found the best. Other seeds are stuck in local optima"
  
SCENARIO C: Same team, different fitness ⚠️ (WORST CASE)
  ⚠ Run 1: fitness=0.7324, team=X
  ⚠ Run 2: fitness=0.7100, team=X
  ⚠ Run 3: fitness=0.6800, team=X
  
  → Random initialization affects final fitness even for same team
  → Conclusion: "Seed matters. Maybe just got lucky in run 1"
```

### Timeline
- Execution time: ~3-5 runs × 250 gen × 150 pop = 5-10 minutes per run
- **Total**: 30-50 minutes runtime

---

## Experiment 2: Entropy Ablation (Causality Test)

### Why This Matters
Currently, entropy bonus is **20.5% of ConfigC's fitness**. But without it, does diversity collapse?

This tells you if diversity is:
- A) **Intrinsically valuable** (team stays diverse even without entropy bonus)
- B) **Artificially forced** (team collapses to all-legendaries without entropy)

### Create ConfigC_NoEntropy

Create a new config in `src/models/ga_config.py`:

```python
def get_config_c_no_entropy():
    """Config C but with ZERO entropy bonus (causality test)"""
    return {
        'name': 'ConfigC_NoEntropy',
        'initialization': 'sqrt_weighted',
        'diversity_weight': 0.0,      # ← ZERO (was 0.15)
        'imbalance_lambda': 0.20,     # Keep same
        'weakness_lambda': 0.10,      # Keep same
        'selection_type': 'tournament',
        'tournament_size': 3,
        'crossover_rate': 0.8,
        'mutation_rate': 0.1,
        'elitism_rate': 0.1,
    }
```

### Command
```bash
python run_ablation_study.py \
  --configs C_NoEntropy \
  --population 150 \
  --generations 250 \
  --run 1
```

### What to Check
```
Compare ConfigC vs ConfigC_NoEntropy:

Fitness:
  ConfigC:           0.7324
  ConfigC_NoEntropy: ???
  
  If NoEntropy > 0.70: Diversity has intrinsic value ✓
  If NoEntropy < 0.60: Diversity bonus was masking weak solution ❌
  
Team composition:
  ConfigC:           baxcalibur, groudon, flutter-mane, stakataka, miraidon, mewtwo
  ConfigC_NoEntropy: ???
  
  If same team: Diversity bonus didn't force anything
  If all legendaries: Entropy bonus was doing the heavy lifting ❌
  
Convergence:
  ConfigC:           Gen 25
  ConfigC_NoEntropy: ???
  
  If earlier convergence: Penalties alone are strong
  If later: Entropy bonus was needed for fast convergence
```

### Expected Outcomes

```
SCENARIO A (Diversity is intrinsically good):
  ConfigC_NoEntropy fitness: ~0.70 or higher
  ConfigC_NoEntropy team: Still has 4-5 archetypes
  → Conclusion: "Diversity is valuable; entropy wasn't needed"

SCENARIO B (Entropy bonus was the secret sauce):
  ConfigC_NoEntropy fitness: ~0.58 (big drop)
  ConfigC_NoEntropy team: All 6 legendaries
  → Conclusion: "Goodhart's Law confirmed; entropy bonus forced diversity"
  
SCENARIO C (Penalties alone drive diversity):
  ConfigC_NoEntropy fitness: ~0.70
  ConfigC_NoEntropy team: Mix of legendary + non-legendary
  → Conclusion: "Imbalance/weakness penalties, not entropy, drive diversity"
```

### Timeline
- Execution time: 250 gen × 150 pop ≈ 10 minutes

---

## Experiment 3: Competitive Battle Simulation (Ground Truth)

### Why This Matters
All of this is moot if **actual battle results tell a different story**.

If ConfigC team loses to ConfigA team in simulated battles → all the analysis is academic.
If ConfigC team wins consistently → the diversity strategy works.

### Command
```bash
python scripts/battle_simulator.py \
  --team_a "baxcalibur, groudon, flutter-mane, stakataka, miraidon, mewtwo" \
  --team_b "koraidon, groudon, yveltal, miraidon, kyogre, dialga" \
  --simulations 1000 \
  --output battle_results.csv
```

### What to Check
```
Win rate:
  ConfigC vs ConfigA: ???%
  
Expected outcomes:

ConfigC wins >55%:
  → Diversity strategy works in practice
  → Entropy bonus was justified
  
ConfigA wins >55%:
  → Stat-maxing beats diversity
  → ConfigC's solution is worse in combat
  
50-50 split:
  → They're equally matched
  → Different tradeoffs for different situations
```

### Timeline
- Execution time: 1000 sims × 2 teams = 2-5 minutes

---

## Summary: What To Run

```
Priority 1 (CRITICAL - do first):
  ✓ Multiple seeds (5 runs of ConfigC)
  ✓ Shows if the solution is robust or lucky
  
Priority 2 (IMPORTANT - do second):
  ✓ ConfigC_NoEntropy (1 run, same length)
  ✓ Shows if entropy bonus is causally driving the result
  
Priority 3 (VALIDATION - if you have time):
  ✓ Battle simulation (1000 sims)
  ✓ Ground truth: does ConfigC actually win?
```

**Total runtime**: 60-90 minutes of computation
**Value**: Definitive answers to all seven skepticism points

---

## Decision Tree

```
                    Start
                      |
        ┌─────────────┴─────────────┐
        └─ Run Multiple Seeds ──────┘
           (5 runs of ConfigC)
           
           ✓ All same team?       │
           │ All same fitness?     │
           └─ YES ────────────────┐
                                  │
                        ┌─────────┴────────┐
                        └─ Run Entropy ────┘
                           Ablation
                           
                           Is fitness
                           >0.70?       │
                           │            │
                           SAME TEAM?  │
                           │            │
                        ┌──YES──┐   ┌──NO──┐
                        │       │   │      │
                    Diversity Goodhart's
                    is REAL  Law Found
                    ✓       ❌
                    
                        ↓
                    
            ┌──────────────────────────┐
            └─ Battle Simulation ───────┘
              which team wins?
              
              ConfigC wins?
              │
              FINAL VERDICT
              
    If YES + diversity REAL + ConfigC wins
    → ConfigC is genuinely superior ✅
    
    If NO + Goodhart's Law + ConfigA wins  
    → You're optimizing the wrong thing ❌
```

---

## What To Do With Results

### If Experiments Validate ConfigC
```
Write: "GA Ablation Study - Validation Report"
  - Multiple seeds prove robustness
  - Entropy ablation proves causality
  - Battle sim proves practical advantage
  
Conclusion: ConfigC is recommended
Update: Use ConfigC for team recommendations
Store: Results in learning materials
```

### If Experiments Invalidate ConfigC
```
Write: "GA Ablation Study - Limitations Report"
  - Identified Goodhart's Law in action
  - Found local optima trap
  - Revealed why diversity bonus was necessary
  
Recommendation: Use ConfigA (stat-maxing)
OR: Redesign fitness function (multi-objective)
OR: Add battle simulation objective
Store: Post-mortem in learning materials
```

---

## Files to Create/Modify

```
NEW:
  src/models/ga_config.py → add get_config_c_no_entropy()
  scripts/multiple_seeds_runner.py → loop through seeds
  scripts/entropy_ablation_runner.py → run once with new config

EXISTING:
  Proj1/scripts/run_ablation_study.py → might need --seeds arg
  
RESULTS:
  reports/ga_results/ablation_study_20260304_171808/
    ├── seeds_5run/ (new folder)
    │  ├── Run_Seed_42/
    │  ├── Run_Seed_123/
    │  ├── Run_Seed_456/
    │  ├── Run_Seed_789/
    │  └── Run_Seed_999/
    │
    ├── entropy_ablation/ (new folder)
    │  └── ConfigC_NoEntropy_results/
    │
    └── battle_simulation/ (new folder)
       └── battle_results.csv
```

---

## Expected Deliverables

After running these three experiments:

```
Final Analysis Report:
  - Does ConfigC solution generalize? (seeds test)
  - Is entropy bonus causal? (ablation test)
  - Does ConfigC team win? (simulation test)
  
Recommendation:
  - Use ConfigC? YES/NO
  - If NO, what's the better approach?
  - What changes would improve the GA?
  
Updated Learning Materials:
  - Document the Goodhart's Law lesson
  - Show the experiments and results
  - Explain the limitations found
```

**This is rigorous science. You're validating (or invalidating) your own work—exactly what good research requires.**

---

Generated: 2026-03-04  
Part of: GA Ablation Study Skepticism Analysis
