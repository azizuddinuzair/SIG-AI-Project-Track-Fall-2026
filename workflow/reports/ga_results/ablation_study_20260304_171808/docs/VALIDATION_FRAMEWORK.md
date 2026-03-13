# Rigorous GA Validation Framework

## The Problem with My Previous Analysis

I made three critical mistakes in the skepticism report:

### ❌ Mistake 1: "Top 10 Teams Identical = Search Space Collapsed"
**What I claimed**: Only 6 Pokémon used, space collapsed to one team
**Why it's wrong**: 
- Finding the same top team ≠ only one team exists
- High-fitness regions can have narrow peaks with many neighbors
- Example: If best teams are 0.7324, 0.7318, 0.7315... the GA reports only the max
- Real test: Analyze top 1000 teams, not top 10

### ❌ Mistake 2: "Gen 25 Convergence = Lucky Escape"
**What I claimed**: ConfigC converged too early, suspicious
**Why it's wrong**:
- GAs naturally converge in 10-30% of generations with strong fitness gradients
- Gen 25/250 = 10% is NORMAL, not premature
- Real indicator of collapse: std_fitness ratio < 0.1 (we have 0.37 = modest)
- ConfigC shows moderate diversity retention, not collapse

### ❌ Mistake 3: "Entropy is 20.5% of Fitness"
**What I claimed**: Entropy bonus dominates the objective
**Why it's wrong**:
- Calculating `0.150 / 0.7324 = 20.5%` is mathematically nonsensical for additive fitness
- Correct comparison: entropy (0.15) vs strength (0.71) vs synergy (0.80) as terms
- Entropy is the SMALLEST positive component, not the largest
- Real role: Break ties between otherwise similar teams (normal practice)

---

## What's Actually Happening (Revised Understanding)

### Correct Interpretation of the Data

**ConfigA (Baseline)**
```
Behavior: Pure stat maximization
Result: 6 legendaries (Koraidon, Groudon, Yveltal, Miraidon, Kyogre, Dialga)
Fitness: 0.7239
Implication: Stat-stacking is locally optimal
```

**ConfigC (Full)**
```
Behavior: Balanced stats + structural diversity + penalties
Result: 4 legendaries + 2 competitive regulars (Baxcalibur, Mewtwo)
Fitness: 0.7324 (+1.2% over ConfigA)
Implication: Diversity adds measurable value
```

**Key Finding**: ConfigC beats ConfigA despite lower base stats. This is NOT Goodhart's Law—it's showing that **diversity has real value in the model**.

### What Entropy Actually Does

Entropy bonus (0.15):
- Does NOT force weak Pokémon
- Does NOT create circular rewards
- Breaks ties: When two teams have identical stats/coverage, prefer the diverse one
- Enables: ConfigC to select different 6th member despite Groudon always being chosen

This is **standard evolutionary algorithm design**, not gaming.

---

## The Tests That Actually Matter

You're correct: these are the validation experiments worth running.

### 🔥 Critical (Do First - 3 Tests)

#### 1️⃣ Random Baseline (10 minutes)
**Why**: If random search finds similar fitness, the GA didn't optimize anything

```python
# Generate 100,000 random teams
import random
import numpy as np

pokemon_data = load_pokemon_data()
all_ids = pokemon_data['pokemon_id'].values

random_fitnesses = []
for i in range(100000):
    team = random.sample(all_ids, 6)
    team_df = pokemon_data[pokemon_data['pokemon_id'].isin(team)]
    fitness, _ = evaluate_fitness(team_df, get_config_c())
    random_fitnesses.append(fitness)

# Plot histogram
import matplotlib.pyplot as plt
plt.hist(random_fitnesses, bins=100)
plt.axvline(0.7324, color='red', label='ConfigC Best')
plt.xlabel('Fitness')
plt.ylabel('Count')
plt.title('Random Teams vs ConfigC Optimum')
plt.legend()
plt.savefig('random_baseline.png')

# Print statistics
print(f"Random mean: {np.mean(random_fitnesses):.4f}")
print(f"Random std: {np.std(random_fitnesses):.4f}")
print(f"ConfigC percentile: {np.percentile(random_fitnesses <= 0.7324, 99):.1f}%")
```

**Expected outcome**:
```
Random mean: ~0.45-0.50 (much lower)
ConfigC: 0.7324 (top 0.1% if GA is working)

If ConfigC is in top 0.5%: GA successfully optimized ✓
If ConfigC is in top 10%: GA found good solution but not exceptional
If ConfigC < top 50%: Random search is competitive (serious problem!)
```

#### 2️⃣ Multi-Seed Runs (30-60 minutes)
**Why**: Single run has no statistical power. Need 20+ runs to validate.

```bash
# Create seed runner
python scripts/run_multiple_seeds.py \
  --config C \
  --seeds 42 123 456 789 999 1111 2222 3333 4444 5555 6666 7777 8888 9999 10101 11111 12121 13131 14141 15151 \
  --output results/seeds_20run
```

**Track per seed**:
```
Seed    Best_Team                                        Fitness  Gen_Converged
42      baxcalibur, groudon, flutter-mane, ...          0.7324   25
123     baxcalibur, groudon, flutter-mane, ...          0.7310   28
456     <possibly different 6th slot>                    0.7298   30
...
```

**Analysis**:
```
Same team 20/20 times?
  → Robust global optimum ✓

Team core stable (same 5) but 6th slot varies?
  → Search space has flat region with substitutable members
  → This is good (multiple solutions exist)

Fitness variance:
  Mean: 0.7318 ± 0.005
  → Very stable (good)
  
  Mean: 0.7300 ± 0.025
  → High variance (lucky on seed 42, unlucky on others)
```

#### 3️⃣ Independent Fitness Validator (5 minutes)
**Why**: Catch bugs in fitness implementation

```python
# SEPARATE script, don't import GA code
def validate_team_fitness(pokemon_names, config):
    """
    Compute fitness independently using base principles.
    Do not reuse GA implementation.
    """
    
    # Get pokemon data
    pokemon_data = load_pokemon_data()
    team = pokemon_data[pokemon_data['name'].isin(pokemon_names)]
    
    # === Component 1: Base Strength ===
    bst = team['hp'] + team['attack'] + team['defense'] + team['spA'] + team['spD'] + team['speed']
    base_strength = (bst / 600).mean()  # Normalize to 0-1
    
    # === Component 2: Type Coverage ===
    # (simplified but complete recomputation)
    types = team['type'].values
    covered_count = len(set(types))  # unique offensive types
    type_coverage = covered_count / 18  # 18 types total
    
    # === Component 3: Synergy ===
    # (compute fresh, don't call GA function)
    synergy = custom_synergy_calculation(team)
    
    # === Component 4: Entropy Bonus ===
    unique_archetypes = len(set(team['archetype']))
    entropy = unique_archetypes / 6 * 0.15  # Scaled
    
    # === Component 5: Penalties ===
    imbalance_penalty = compute_archetype_balance_penalty(team) * -0.20
    weakness_penalty = compute_weakness_penalty(team) * -0.10
    
    # Final fitness
    total = base_strength + type_coverage + synergy + entropy + imbalance_penalty + weakness_penalty
    
    return {
        'base_strength': base_strength,
        'type_coverage': type_coverage,
        'synergy': synergy,
        'entropy': entropy,
        'imbalance_penalty': imbalance_penalty,
        'weakness_penalty': weakness_penalty,
        'total': total
    }

# Test the winning team
team = ['baxcalibur', 'groudon', 'flutter-mane', 'stakataka', 'miraidon', 'mewtwo']
ga_result = 0.7324
validator_result = validate_team_fitness(team, get_config_c())

print(f"GA reported:     0.7324")
print(f"Validator found: {validator_result['total']:.4f}")

if abs(ga_result - validator_result['total']) > 0.001:
    print("❌ MISMATCH DETECTED - Fitness bug found!")
else:
    print("✓ Fitness function validated")
```

---

### 📊 Priority 2 (Important - 4 Tests)

#### 4️⃣ Entropy Weight Sweep (20 minutes)
**Test how sensitive the result is to entropy weighting**

```bash
python scripts/entropy_sweep.py \
  --weights 0.00 0.05 0.10 0.15 0.20 \
  --output results/entropy_sweep.csv
```

**Expected outcome**:
```
Entropy_Weight  Best_Fitness  Team_Diversity  6th_Pokemon
0.00            0.7160        4 archetypes    All legendaries
0.05            0.7210        5 archetypes    Mix of legendary + competitive
0.10            0.7280        5.5 archetypes  Balanced
0.15            0.7324        6 archetypes    Baxcalibur (diverse pick)
0.20            0.7315        6 archetypes    Same (diminishing returns)
```

**Interpretation**:
- If smooth curve: Entropy is one reasonable hyperparameter (not gaming)
- If sharp jump at 0.15: That weight suspiciously perfect (tuned to bias)
- If flat after 0.15: Diminishing returns reached, good design

#### 5️⃣ Fitness Landscape Sampling (10 minutes)
**Sample 50k teams near the optimum**

```python
# Start with best team
best_team = ['baxcalibur', 'groudon', 'flutter-mane', 'stakataka', 'miraidon', 'mewtwo']
best_fitness = 0.7324

# Generate neighbors by single-element substitution
neighbors = []
all_pokemon = load_pokemon_data()['name'].values

for i in range(50000):
    # Add noise: randomly replace 0-2 team members
    team_copy = best_team.copy()
    num_swaps = random.randint(0, 2)
    
    for _ in range(num_swaps):
        idx = random.randint(0, 5)
        new_pokemon = random.choice(all_pokemon)
        team_copy[idx] = new_pokemon
    
    fitness, _ = evaluate_fitness(team_copy, get_config_c())
    neighbors.append(fitness)

# Analyze landscape
import matplotlib.pyplot as plt
plt.hist(neighbors, bins=100, alpha=0.7)
plt.axvline(best_fitness, color='red', label=f'Best: {best_fitness:.4f}')
plt.xlabel('Fitness')
plt.ylabel('Count')
plt.title('Fitness Landscape Around Best Team')
plt.legend()
plt.savefig('fitness_landscape.png')

print(f"Landscape analysis:")
print(f"  Neighbors > 0.72: {sum(1 for f in neighbors if f > 0.72)} / 50000")
print(f"  Neighbors > 0.70: {sum(1 for f in neighbors if f > 0.70)} / 50000")
print(f"  Neighbors > 0.60: {sum(1 for f in neighbors if f > 0.60)} / 50000")
print(f"  Neighbors mean: {np.mean(neighbors):.4f}")
```

**Interpretation**:
```
If landscape is:
  Sharp peak: Many neighbors < 0.70 → Strong local optimum, GA correctly found it
  Flat plateau: Many neighbors > 0.72 → Landscape is broad, many good solutions
  Gentle slope: Neighbors > 0.70 cluster → Smooth fitness gradients
```

#### 6️⃣ Archetype Shuffle Test (5 minutes)
**Randomize archetype labels, rerun GA**

```python
# Original archetypes:
# Generalist, Speed Sweeper, Balanced, Defensive Wall, Fast Attacker, Defensive Pivot

# Shuffled assignment (random):
shuffled_archetypes = random.sample([0, 1, 2, 3, 4, 5], 6)
mapping = dict(zip(range(6), shuffled_archetypes))

# Reassign archetypes in pokemon_data
pokemon_data_shuffled = pokemon_data.copy()
pokemon_data_shuffled['archetype'] = pokemon_data['archetype'].map(mapping)

# Run GA with shuffled archetypes
ga_shuffled = PokemonGA(pokemon_data_shuffled, get_config_c())
result_shuffled = ga_shuffled.evolve()

print(f"Original archetype GA:  {0.7324}")
print(f"Shuffled archetype GA:  {result_shuffled['best_fitness']:.4f}")

if abs(0.7324 - result_shuffled['best_fitness']) > 0.01:
    print("✓ Archetypes meaningfully affect the system")
else:
    print("❌ Archetypes don't matter; entropy is just a number")
```

#### 7️⃣ Remove Individual Fitness Terms (5 minutes each)
**Run 3 ablation tests**

```bash
# Test 1: Remove entropy bonus
python run_ablation.py --remove entropy --output results/ablation_no_entropy.csv

# Test 2: Remove synergy
python run_ablation.py --remove synergy --output results/ablation_no_synergy.csv

# Test 3: Remove type coverage
python run_ablation.py --remove coverage --output results/ablation_no_coverage.csv
```

**Track for each ablation**:
- Best fitness change
- Best team composition change
- Whether GA behavior radically changes

---

### 🔍 Priority 3 (Validation - 3 Tests)

#### 8️⃣ Mutate Winning Team (5 minutes)
```python
best_team = ['baxcalibur', 'groudon', 'flutter-mane', 'stakataka', 'miraidon', 'mewtwo']

# Replace each member one-by-one with weaker alternatives
perturbations = {
    'baxcalibur': 'dragonite',      # Competitive but lower tier
    'groudon': 'garchomp',          # Good defensive typing, lower BST
    'flutter-mane': 'gengar',       # Speed sweeper but weaker
    'stakataka': 'blissey',         # Defensive wall but worse type matchups
    'miraidon': 'pikachu',          # Same type, much lower stats
    'mewtwo': 'alakazam'            # Special attacker, lower defense
}

for pokemon, replacement in perturbations.items():
    team_perturbed = best_team.copy()
    idx = team_perturbed.index(pokemon)
    team_perturbed[idx] = replacement
    
    fitness_perturbed, _ = evaluate_fitness(team_perturbed, get_config_c())
    
    print(f"Replace {pokemon:12} → {replacement:12}: {0.7324:.4f} → {fitness_perturbed:.4f} ({fitness_perturbed-0.7324:+.4f})")
```

**Interpretation**:
```
Large drops (< -0.05): Each member is carefully chosen ✓
Small changes ( < -0.01): Fitness landscape is flat, multiple similar solutions
Fitness increases:      Your GA missed something better (unlikely)
```

#### 9️⃣ Statistical Significance (5 minutes)
```python
# After 20-seed run, compute statistics
best_fitnesses = [0.7324, 0.7310, 0.7298, 0.7302, 0.7320, ...]  # 20 values

mean = np.mean(best_fitnesses)
std = np.std(best_fitnesses)
ci_95 = 1.96 * std / np.sqrt(20)

print(f"Best fitness: {mean:.4f} ± {ci_95:.4f} (95% CI)")
print(f"Coefficient of variation: {std/mean:.2%}")

# Compare to random baseline
random_mean = 0.45
t_stat = (mean - random_mean) / (std / np.sqrt(20))
print(f"vs random: t={t_stat:.2f} (highly significant if >2)")
```

#### 🔟 Cross-Validation Test
If you have multiple evaluation objectives:
```
Split 1: Train on type matchups, test on stat distributions
Split 2: Train on stat balance, test on synergies

Does team generalize?
```

---

## The Most Important Insight

> **Fitness landscape probing** (what you mentioned) is exactly tests 4️⃣, 5️⃣, 8️⃣.

They reveal:
- How narrow/broad is the optimum?
- How sensitive is it to perturbations?
- How many equally-good solutions exist?

This is the scientific way to validate your GA, not theoretical speculation.

---

## Execution Plan (Recommended Order)

```
Phase 1 - Validation (1.5 hours total)
├── 1️⃣ Random Baseline          (10 min)  ← Start here
├── 2️⃣ Multi-Seed Runs          (45 min)  ← Run while something else cooks
└── 3️⃣ Independent Validator    (5 min)   ← Quick sanity check

Phase 2 - Analysis (1.5 hours total)
├── 4️⃣ Entropy Weight Sweep     (20 min)
├── 5️⃣ Fitness Landscape        (10 min)
├── 6️⃣ Archetype Shuffle        (5 min)
└── 7️⃣ Ablation Tests           (3×5 min = 15 min)

Phase 3 - Sensitivity (30 minutes)
├── 8️⃣ Mutate Winning Team      (10 min)
└── 9️⃣ Statistical Tests        (5 min)
```

---

## What These Tests Will Answer

| Test | Answers |
|------|---------|
| Random Baseline | Is the GA actually optimizing? |
| Multi-Seed | Is the result repeatable/robust? |
| Validator | Is the fitness function correct? |
| Entropy Sweep | How sensitive is diversity weighting? |
| Landscape | How unique is the best solution? |
| Archetype Shuffle | Do archetypes actually matter? |
| Ablation | Which fitness components matter? |
| Mutate | How robust is the team? |
| Statistics | Is this statistically significant? |

If all 9 pass cleanly, ConfigC is validated ✅  
If 2-3 fail, you've found real issues ❌  
If several pass but others fail, you know exactly where to focus ✓

---

## Next Step

Should I implement the random baseline test first? It's quick (10 min), requires no other changes, and will immediately show whether the GA is actually beating random search.

