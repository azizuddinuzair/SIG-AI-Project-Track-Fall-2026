# Phase 2: Ablation & Sensitivity Analysis Strategy

**Status**: Ready to execute  
**Target Runtime**: < 30 minutes  
**Random Seed**: Fixed at 42 (reproducibility)  
**Date**: March 4, 2026

---

## Overview

Phase 2 answers the question: **Why does ConfigC work?** 

After Phase 1 proved reproducibility (all 5 seeds converge to identical fitness), Phase 2 identifies which components drive that convergence:

1. **Entropy Sweep** — Varies diversity bonus magnitude
2. **Ablation Tests** — Disables individual components
3. **Init Sensitivity** — Tests weighting strategies  
4. **Neighbor Sampling** — Probes fitness landscape sharpness

---

## Execution Strategy (You Recommended)

### 1. Reduced GA Parameters (Signal Preserved, Runtime Halved)

| Parameter | Phase 1 | Phase 2 | Rationale |
|-----------|---------|---------|-----------|
| Population | 150 | 100 | 33% reduction; still enough diversity for selection |
| Generations | 250 | 100 | 60% reduction; signal emerges by gen 50-80 |
| Elite %  | ~3% | 3% | Preserved; maintains selection pressure |
| Tournament k | 3 | 3 | Preserved; selection dynamics unchanged |

**Effect**: ~50-60% faster per-run while preserving relative fitness differences.

### 2. Fixed Seed Across All Experiments

All tests use `random_seed=42` (same as Phase 1 best run).

**Benefit**: Eliminates initialization variance → differences purely from config changes.

**Example**: 
- ConfigC vs. ConfigB won't differ due to "lucky random init"
- Instead, you directly observe which components matter

### 3. Pokémon Subset Sampling

Neighbor sampling uses:
- **Archetype-constrained swaps**: Only swap within same archetype (reduces search space by ~90%)
- **Sampled neighbors**: 50 random swaps per team member instead of exhaustive 3,210
- **Top 100 by stats**: (Optional) For fitness landscape evaluation, sample from high-quality Pokémon

**Benefit**: 3-5× speedup on landscape probing without losing peak detection.

### 4. Minimal Output Logging

Capture only:
- Best fitness & team per run
- Convergence generation  
- Final population std dev
- Landscape sharpness metrics (for neighbor sampling)

**Skipped**: Full generation-by-generation logs, archetype distributions (reduce I/O overhead).

---

## Experiment Breakdown

### Experiment 1: Entropy Sweep (5 runs, ~2 min)

**Question**: How sensitive is convergence to diversity weight?

**Test Cases**:

| diversity_weight | Expected Behavior | Hypothesis |
|---|---|---|
| 0.50 | Heavy diversity push; slower convergence | May prevent specialization |
| 0.35 | Moderate diversity; faster than 0.50 | Balanced approach |
| 0.25 | Mild diversity bonus | Still favors balance |
| 0.15 | ConfigC baseline | Optimal (from Phase 1) |
| 0.10 | Lower diversity pressure | May converge faster but lower variety |

**Interpretation**:
- If all weights → same fitness: Diversity component ineffective
- If 0.15 is best: Optimal balance identified
- If gradient clear: Tune for problem-specific needs

**Runtime**: ~25 sec/run × 5 = ~2 min

---

### Experiment 2: Ablation Tests (5 runs, ~2.5 min)

**Question**: What's the individual contribution of each component?

**Test Cases**:

| Config | Changes | Tests |
|--------|---------|-------|
| baseline_full | All components enabled (ConfigC) | Reference |
| no_entropy | `diversity_weight=0` | Entropy impact |
| no_balance_penalty | `imbalance_lambda=0` | Balance penalty impact |
| no_weakness_penalty | `weakness_lambda=0` | Weakness penalty impact |
| uniform_init_no_diversity | Uniform init + all penalties disabled | Pure base stats only |

**Interpretation**:
- **Fitness drop** from removing component = its importance
- Example: If removing entropy drops fitness 2%, it's minor; if 10%, it's critical

**Runtime**: ~25 sec/run × 5 = ~2.5 min

---

### Experiment 3: Initialization Sensitivity (3 runs, ~1.5 min)

**Question**: Does archetype weighting during init/mutation improve convergence?

**Test Cases**:

| Init Method | Mutation Weighted | Expected Behavior |
|---|---|---|
| uniform | False | Baseline diversity |
| inverse | True | Rare archetype bias (aggressive) |
| sqrt_weighted | True | Balanced rare archetype boost |

**Interpretation**:
- If sqrt_weighted == best: Smooth weighting favored
- If uniform competitive: Weighting doesn't help
- If inverse best: Aggressive rare specialization works

**Runtime**: ~25 sec/run × 3 = ~1.5 min

---

### Experiment 4: Neighbor Sampling (1 run, ~3-5 min)

**Question**: How sharp is the fitness peak? (Local landscape geometry)

**Test Cases**:

For each of 6 team members:
1. Identify archetype (e.g., "Miraidon" → "Special Attacker")
2. Sample 50 random Pokémon from same archetype
3. Swap member with each neighbor
4. Evaluate fitness

**Metrics per position**:
- `max_neighbor_fitness` — Best alternative in archetype
- `max_delta` — How much worse neighbors are
- `peak_sharpness` — "high" (delta < 1%), "medium" (< 5%), "low" (> 5%)

**Interpretation**:
- **High sharpness**: Peak is specific; hard to improve locally
- **Low sharpness**: Many alternatives equally good; team is average
- **Archetype variance**: If rare archetypes have lower neighbors → they're critical

**Runtime**: 6 positions × 50 swaps + fitness evals = ~3-5 min

---

## Expected Outputs

### results/03_ablation_sensitivity_results.json

```json
{
  "phase": "Phase 2 - Ablation & Sensitivity",
  "strategy": "Runtime-optimized (100 pop × 100 gen, fixed seed=42)",
  "elapsed_seconds": 1234.5,
  "experiments": {
    "entropy_sweep": {
      "runs": [
        {
          "diversity_weight": 0.50,
          "best_fitness": 0.7250,
          "convergence_generation": 38,
          "elapsed_seconds": 22.4
        },
        ...
      ]
    },
    "ablation_tests": { ... },
    "init_sensitivity": { ... },
    "neighbor_sampling": {
      "positions": [
        {
          "position": 0,
          "pokemon": "baxcalibur",
          "max_neighbor_fitness": 0.7100,
          "max_delta": 0.0224,
          "peak_sharpness": "high"
        },
        ...
      ]
    }
  }
}
```

---

## Running Phase 2

### Quick Start

```powershell
# From phase2 directory (or ablation_study root)
cd ../
python scripts/03_ablation_sensitivity.py
```

### With Custom Output Directory

```powershell
# From phase2 directory
python scripts/03_ablation_sensitivity.py --output ../results
```

### Progress Indicators

The script prints progress for each experiment:

```
======================================================================
ENTROPY SWEEP: Diversity Weight Sensitivity
======================================================================

  Testing diversity_weight=0.5... fitness=0.7205, conv_gen=42
  Testing diversity_weight=0.35... fitness=0.7289, conv_gen=28
  ...
```

**When complete**, you'll see:

```
======================================================================
PHASE 2 COMPLETE
======================================================================
Total elapsed time: 1234.5 seconds (20.6 minutes)
Results saved to: c:\...\phase2\results\03_ablation_sensitivity_results.json

Next: Analyze results to identify which components matter most.
```

---

## Interpretation Guide

### Entropy Sweep Results

| Pattern | Interpretation |
|---------|---|
| Flat fitness across weights | Entropy component neutral; can disable |
| Peak at 0.15 | ConfigC optimal; fine-tune ±0.05 nearby |
| Monotonic decrease (high→low) | Diversity critical; keep weight high |

### Ablation Results  

| Component Removed | Fitness Drop | Implication |
|---|---|---|
| entropy (diversity_weight) | < 0.5% | Minor;  optional optimization |
| balance_penalty | 1-3% | Important; removes teams with 4× same type |
| weakness_penalty | 1-2% | Moderate; prevents critical gaps |
| All penalties + uniform | > 5% | Diversity essential; ConfigC justified |

### Init Sensitivity Results

| Winner | Interpretation |
|---|---|
| sqrt_weighted | Balanced weighting best (sweet spot) |
| inverse | Aggressive specialization pays off |
| uniform | Weighting irrelevant; GA handles it |

### Landscape Sharpness Results

| Sharpness | Meaning |
|---|---|
| "high" (delta < 1%) | Member critical to team; hard to improve locally |
| "medium" (1-5%) | Some alternatives exist; member somewhat flexible |
| "low" (> 5%) | Member generic; many equivalents in archetype |

---

## Decision Framework: Is Phase 2 Needed?

✅ **Run Phase 2 if**:
- Publishing research (need component impact quantification)
- Tuning for production (identify which components to keep)
- Curiosity about mechanism (understand why ConfigC works)

⏭️ **Skip Phase 2 if**:
- MVP/deadline pressure (Phase 1 proof sufficient)
- ConfigC already deployed successfully
- Need to move to next project phase

---

## What's Next After Phase 2?

**If results show**:
1. **One component dominates** → Simplify; remove others
2. **All components matter** → ConfigC is minimal viable set
3. **Possible improvements identified** → Implement Phase 2b (targeted tuning)

---

## Script Architecture

**Files**:
- `scripts/03_ablation_sensitivity.py` — Main Phase 2 orchestrator
- `results/03_ablation_sensitivity_results.json` — Output with all metrics

**Key Functions**:
- `create_quick_config()` — GA wrapper with reduced params
- `find_convergence_generation()` — Detects gen where fitness plateaus
- `run_entropy_sweep()` — Diversity weight sensitivity test
- `run_ablation_tests()` — Component impact quantification
- `run_init_method_sensitivity()` — Weighting strategy comparison
- `run_neighbor_sampling()` — Fitness landscape exploration

**Dependencies**:
- `PokemonGA`, `get_config_c()` from `ga_optimization.py`, `ga_config.py`
- `evaluate_fitness()` from `ga_fitness.py`
- `load_pokemon_data()` from `clustering_pipeline.py`
- NumPy, Pandas, JSON, time, pathlib

---

## Summary

| Aspect | Details |
|--------|---------|
| **Goal** | Quantify which GA components drive ConfigC's success |
| **Pproach** | 4 focused experiments; ~13 GA runs + landscape sampling |
| **Runtime** | < 30 min total (vs. 5 hrs for Phase 1 with 250 gen) |
| **Output** | JSON with component impact, convergence patterns, landscape geometry |
| **Decision** | Optional if MVP sufficient; essential for research/publication |

You're ready to execute. Let me know when you want to start Phase 2! 🚀
