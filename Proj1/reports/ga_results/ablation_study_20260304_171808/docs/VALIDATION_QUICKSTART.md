# Quick-Start: Validating ConfigC (Priority Order)

## TL;DR - Do These 3 Tests First (90 minutes total)

If you only have 1.5 hours and want answers, run these:

### 1️⃣ Random Baseline (10 minutes)
**Test**:  Does the GA beat random search?

**Command**: 
```bash
python scripts/random_baseline.py --samples 100000 --output results/random_baseline.csv
```

**What it checks**: If the GA's best fitness (0.7324) is in the top 0.5% of 100k random teams, the optimization worked
  
**Expected**:
```
Random mean:        ~0.45
Random std:         ~0.15
ConfigC fitness:    0.7324

If ConfigC > 99th percentile:    GA successfully optimized ✓
If ConfigC < 95th percentile:    Random search is competitive (problem!) ❌
```

---

### 2️⃣ Multi-Seed Runs (45 minutes - can run while you do other work)
**Test**: Is the result repeatable or luck?

**Command**:
```bash
python scripts/run_multiple_seeds.py \
  --config C \
  --seeds 42 123 456 789 999 1111 2222 3333 4444 5555 \
  --num_seeds 10 \
  --output results/seeds_10run
```

**What it checks**: Do independent runs with different random seeds find the same (or similar) solution?

**Expected**:
```
Case 1: Same team every time
  Mean fitness: 0.732 ± 0.001
  → Solution is robust ✓

Case 2: Different teams but high fitness
  Mean fitness: 0.725 ± 0.005
  → Multiple good optima exist ✓

Case 3: High variance
  Mean fitness: 0.720 ± 0.020
  → Each seed finds different thing (concerning)
  
Case 4: Random seed has huge impact
  Mean fitness: 0.730 ± 0.050
  → Luck dominated, not optimization
```

---

### 3️⃣ Independent Fitness Validator (5 minutes)
**Test**: Is the fitness function correctly implemented?

**Command**:
```bash
python scripts/validate_fitness.py \
  --team "baxcalibur,groudon,flutter-mane,stakataka,miraidon,mewtwo" \
  --config C
```

**What it checks**: Recompute fitness in isolation. If it matches GA's 0.7324, no bug exists.

**Expected**:
```
GA reported:       0.7324
Validator found:   0.7324 ± 0.0001
→ Match within floating point error ✓

GA reported:       0.7324
Validator found:   0.6800
→ Bug in fitness function ❌
```

---

## Phase 2: Understand Why (1.5 hours)

**After you confirm tests 1-3 pass**, run these to understand the mechanism:

#### 4️⃣ Entropy Weight Sweep (20 min)
**Why**: How sensitive is the result to entropy weighting?

```bash
python scripts/entropy_sweep.py \
  --weights 0.00 0.05 0.10 0.15 0.20 \
  --output results/entropy_sweep.csv
```

**Result tells you**:
- If smooth curve → Entropy is a reasonable hyperparameter
- If sharp jump at 0.15 → That weight seems suspiciously perfect
- If flat after 0.15 → Diminishing returns (good design)

#### 5️⃣ Fitness Landscape Sampling (10 min)
**Why**: How unique is the best team?

```bash
python scripts/fitness_landscape.py \
  --best_team "baxcalibur,groudon,flutter-mane,stakataka,miraidon,mewtwo" \
  --num_neighbors 50000 \
  --output results/landscape.png
```

**Result tells you**:
- Sharp peak → Solution is unique (good)
- Flat plateau → Many equally-good solutions (also good)
- Multiple peaks → Multiple local optima (typical)

#### 6️⃣ Archetype Shuffle Test (5 min)
**Why**: Do archetypes actually matter, or just entropy numbers?

```bash
python scripts/archetype_shuffle.py \
  --num_runs 5 \
  --output results/shuffle_comparison.csv
```

**Result tells you**:
- Fitness changes >5% → Archetypes are meaningful ✓
- Fitness same → Entropy bonus is arbitrary ❌

---

## Phase 3: Stress Test (30 minutes)

#### 7️⃣ Remove Fitness Terms (3 × 5 min = 15 min)
**Why**: Which components actually drive the result?

```bash
# Test without entropy bonus
python run_ablation.py --remove entropy_bonus --output ablation_no_entropy.csv

# Test without synergy
python run_ablation.py --remove synergy --output ablation_no_synergy.csv

# Test without penalties
python run_ablation.py --remove penalties --output ablation_no_penalties.csv
```

**Result tells you**:
- Fitness drops dramatically → That term is critical
- Fitness similar → That term can be removed

#### 8️⃣ Mutate Winning Team (10 min)
**Why**: How robust is the solution?

```bash
python scripts/perturb_team.py \
  --team "baxcalibur,groudon,flutter-mane,stakataka,miraidon,mewtwo" \
  --output results/perturbations.csv
```

**Result shows**:
```
Replace baxcalibur → dragonite:    0.7324 → 0.7100 (big drop = important)
Replace miraidon → pikachu:        0.7324 → 0.6200 (huge drop = critical)
```

---

## Decision Tree: What Results Mean

```
                Start
                 │
         ┌───────┴───────┐
         ▼               ▼
    Random Test    Multi-Seed Test
    [Passed?]      [Stable?]
      │                │
    YES  NO           YES  NO
     │    │            │    │
     │    │         ┌──┴┘   └──┐
     │    │         │          │
     ▼    ▼         ▼          ▼
     
Case A: Random✓ Seeds✓
→ GA works + solution is robust
Action: ConfigC is validated ✓
Next: Run Phase 2 tests for understanding

Case B: Random✓ Seeds✗
→ GA works but high variance
Action: Try longer runs, larger populations
Next: Use ensemble of multiple runs

Case C: Random✗ Seeds✓
→ GA stuck in bad optimum
Action: Increase population size
Next: Redesign fitness function

Case D: Random✗ Seeds✗  
→ Serious problem
Action: Debug fitness function immediately
Next: Run validator test (Phase 1.3)
```

---

## What Each Test Takes (Runtime Summary)

```
Phase 1 - Validation (Critical, 1 hour total)
├─ Random baseline    10 min    (100k random teams)
├─ Multi-seed        45 min    (can run async)
└─ Validator          5 min    (sanity check)

Phase 2 - Analysis (Important, 1.5 hours)
├─ Entropy sweep     20 min    (5 configs × 250 gen)
├─ Landscape probe   10 min    (sample 50k neighbors)
├─ Archetype shuffle  5 min    (just a test)
└─ Ablation tests    15 min    (3 runs × 5 min)

Phase 3 - Stress (Refinement, 30 minutes)
├─ Mutation test     10 min    (lots of single shifts)
└─ Statistics         5 min    (compute variance)

Total: ~3 hours of computation
Actual waiting: <90 minutes (many can run in parallel)
```

---

## What I Recommend You Do Right Now

### Option A: "I Want Quick Answers" (1 hour)
1. Run random baseline (10 min)
2. Start multi-seed in background (45 min)
3. While that runs, do fitness validator (5 min)
4. Check results while prepping Phase 2

### Option B: "I Want Comprehensive Validation" (3 hours)
1. Run all Phase 1 tests
2. If Phase 1 passes, run Phase 2
3. If Phase 2 shows issues, run Phase 3 to debug

### Option C: "I'm Skeptical, Show Me Everything" (4 hours)
Run all 9 tests. If all pass, ConfigC is scientifically validated.

---

## File References

- Implementation scripts: `Proj1/scripts/`
- Results storage: `Proj1/reports/ga_results/ablation_study_20260304_171808/`
- Full documentation: [VALIDATION_FRAMEWORK.md](VALIDATION_FRAMEWORK.md)
- Analysis results: [SKEPTICISM_ANALYSIS.md](SKEPTICISM_ANALYSIS.md)

---

## Next Steps

1. **Decide your timeline** (Quick/Comprehensive/Everything)
2. **Pick Phase 1 tests to run first**
3. **Let them execute while you review results**
4. **Interpret using the decision tree above**
5. **Proceed to Phase 2 only if Phase 1 shows valid optimization**

The validation is systematic and empirical, not speculative. Each test answers a specific question.

**Ready to run test #1?** It takes 10 minutes and tells you immediately if the GA is actually optimizing.
