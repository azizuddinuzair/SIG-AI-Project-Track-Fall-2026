# Phase 1 Validation Tests

## Overview

These tests determine if ConfigC's GA solution is:
1. **Better than random** (Test 1)
2. **Reproducible** (Test 2)
3. **Correctly computed** (Test 3)

If all three pass: ConfigC is validated and worth investigating further.

---

## Tests

### 1️⃣ Random Baseline Test (`01_random_baseline.py`)
- **What**: Generates 100,000 random 6-Pokémon teams
- **Why**: Proves GA is actually optimizing, not just lucky
- **Time**: ~10 minutes
- **Pass criteria**: ConfigC fitness in top 0.5% of random teams
- **Files**:
  - Input: `../../data/` (pokemon data, ConfigC config)
  - Output: `01_random_baseline_results.json`

**Expected output:**
```
Random team statistics (n=100,000):
  Mean fitness:    0.565342
  Max fitness:     0.683204
  Std deviation:   0.024567

ConfigC comparison:
  ConfigC fitness: 0.732371
  Better than:     99.97% of random teams
  Rank:            3 / 100,000

📊 ✅ PASS: GA optimizes significantly (top 0.5%)
```

---

### 2️⃣ Multi-Seed Validation Test (`02_multiseed.py`)
- **What**: Runs GA 5 times with different seeds (42, 123, 456, 789, 999)
- **Why**: Proves solution is reproducible, not a lucky seed
- **Time**: ~45 minutes
- **Pass criteria**: Standard deviation < 2% between runs
- **Files**:
  - Input: Pokemon data, ConfigC config
  - Output: `02_multiseed_results.json`

**Expected output:**
```
Summary across 5 successful runs:
  Mean fitness:    0.731200
  Max fitness:     0.732371
  Min fitness:     0.729400
  Std deviation:   0.000987 (0.13%)

📊 ✅ EXCELLENT: Highly reproducible (std < 1%)
```

---

### 3️⃣ Fitness Validator Test (`03_fitness_validator.py`)
- **What**: Recomputes ConfigC's best team fitness independently
- **Why**: Ensures no bugs in fitness calculation
- **Time**: ~5 minutes (fast)
- **Pass criteria**: Computed fitness matches reported fitness (within tolerance)
- **Files**:
  - Input: ConfigC best team from ablation study
  - Output: `03_fitness_validator_results.json`

**Expected output:**
```
Fitness computation:
  Reported fitness: 0.7323711134
  Computed fitness: 0.7323711134
  Difference:       0.0000000000

📊 ✅ PASS: Fitness computation is correct
```

---

## How to Run

### Option 1: Run All Tests at Once
```bash
# From phase1 directory
cd ../
python scripts/01_random_baseline.py
python scripts/02_multiseed.py
python scripts/03_fitness_validator.py
```

**Total time**: ~1 hour

### Option 2: Run Individual Tests
```bash
# Test 1 (fast)
python scripts/01_random_baseline.py

# Test 2 (slow - run in background)
python scripts/02_multiseed.py

# Test 3 (fast)
python scripts/03_fitness_validator.py
```

---

## Interpreting Results

### All Tests Pass ✅
```
Configuration is VALID.
→ Proceed to Phase 2 to understand mechanism
```

### Random Baseline Fails ❌
```
GA is not optimizing vs random selection.
→ Check fitness function, selection pressure, or GA parameters
```

### Multi-Seed Fails ❌
```
Results are not reproducible.
→ Increase population size, check mutation rates
```

### Fitness Validator Fails ❌
```
Bug in fitness calculation detected.
→ Review fitness component computation
```

---

## Files Generated

After running Phase 1, you'll have:

```
validation/phase1/
├── scripts/
│   ├── 01_random_baseline.py
│   ├── 02_multiseed.py
│   └── 03_fitness_validator.py
├── results/
│   ├── 01_random_baseline_results.json
│   ├── 02_multiseed_results.json
│   └── 03_fitness_validator_results.json
└── reports/
  └── README.md
```

---

## Next Steps

**If Phase 1 passes:**
- Read: `../../docs/VALIDATION_FRAMEWORK.md` (understanding Phase 2)
- Run Phase 2 tests to understand WHY ConfigC wins:
  - Entropy weight sweep
  - Fitness landscape exploration
  - Component ablation tests

**If Phase 1 fails:**
- Debug the failing test
- Check GA configuration in `../../data/ConfigC_Full_config.txt`
- Review fitness components in `../../docs/SKEPTICISM_ANALYSIS.md`

---

## Timing

| Test | Time | Dependencies |
|------|------|--------------|
| Random Baseline | 10 min | Pokemon data |
| Fitness Validator | 5 min | ConfigC best team |
| Multi-Seed (5 runs) | 45 min | Full GA execution |
| **Total** | **~1 hour** | Sequential or parallel |

---

## Questions to Answer

After Phase 1:

- ❓ "Is the GA actually optimizing?" → Random baseline test
- ❓ "Is the solution stable or lucky?" → Multi-seed test
- ❓ "Were the numbers computed correctly?" → Fitness validator
- ❓ "Which components matter?" → Phase 2 (ablation tests)
- ❓ "Is this solution strategically sound?" → Phase 3 (battle simulation)

