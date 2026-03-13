#!/usr/bin/env powershell
<#
Phase 2 Execution Guide
Quick reference for running the Phase 2 ablation study.
#>

# ============================================================================
# PHASE 2 ABLATION & SENSITIVITY STUDY
# ============================================================================

# Navigate to project root
cd 'c:\Users\rezas\GitHub\SIG-AI-Project-Track-Fall-2026\Proj1'

# Set Python path
$pythonExe = 'C:\Users\rezas\AppData\Local\Python\bin\python.exe'

# Set Phase 2 script path
$phase2Script = 'reports\ga_results\ablation_study_20260304_171808\validation\phase2\scripts\03_ablation_sensitivity.py'

# ============================================================================
# OPTION 1: SIMPLE EXECUTION (Recommended)
# ============================================================================

Write-Host "Starting Phase 2 Ablation Study..." -ForegroundColor Cyan
Write-Host "This will take approximately 15-25 minutes." -ForegroundColor Yellow
Write-Host ""

& $pythonExe $phase2Script

# Results will be saved to: 
# reports\ga_results\ablation_study_20260304_171808\validation\phase2\results\03_ablation_sensitivity_results.json


# ============================================================================
# OPTION 2: WITH CUSTOM OUTPUT DIRECTORY
# ============================================================================

# $outputDir = 'reports\ga_results\ablation_study_20260304_171808\validation\phase2'
# & $pythonExe $phase2Script --output $outputDir


# ============================================================================
# OPTION 3: RUN WITH TIMING  
# ============================================================================

# $startTime = Get-Date
# & $pythonExe $phase2Script
# $endTime = Get-Date
# Write-Host ""
# Write-Host "Phase 2 Complete!"
# Write-Host "Total Time: $([Math]::Round(($endTime - $startTime).TotalMinutes, 1)) minutes"


# ============================================================================
# WHAT TO EXPECT
# ============================================================================

<#

The script will print progress like this:

======================================================================
PHASE 2: ABLATION & SENSITIVITY ANALYSIS (Optimized for Speed)
======================================================================

Loading Pokémon dataset... Loaded 535 Pokémon

======================================================================
ENTROPY SWEEP: Diversity Weight Sensitivity
======================================================================

  Testing diversity_weight=0.5... fitness=0.7205, conv_gen=42
  Testing diversity_weight=0.35... fitness=0.7289, conv_gen=28
  Testing diversity_weight=0.25... fitness=0.7310, conv_gen=18
  Testing diversity_weight=0.15... fitness=0.7324, conv_gen=14
  Testing diversity_weight=0.10... fitness=0.7290, conv_gen=11

======================================================================
ABLATION TEST: Individual Component Impact
======================================================================

  Testing baseline_full... fitness=0.7324, conv_gen=14
  Testing no_entropy... fitness=0.7310, conv_gen=12
  Testing no_balance_penalty... fitness=0.7200, conv_gen=18
  Testing no_weakness_penalty... fitness=0.7280, conv_gen=15
  Testing uniform_init_no_diversity... fitness=0.6800, conv_gen=45

... (and so on for remaining experiments)

======================================================================
PHASE 2 COMPLETE
======================================================================
Total elapsed time: 1234.5 seconds (20.6 minutes)
Results saved to: c:\Users\rezas\GitHub\SIG-AI-Project-Track-Fall-2026\Proj1\reports\ga_results\ablation_study_20260304_171808\validation\phase2\results\03_ablation_sensitivity_results.json

Next: Analyze results to identify which components matter most.

#>


# ============================================================================
# AFTER COMPLETION: ANALYZE RESULTS
# ============================================================================

<#

1. Read the results JSON:
  cat 'reports\ga_results\ablation_study_20260304_171808\validation\phase2\results\03_ablation_sensitivity_results.json' | ConvertFrom-Json | ConvertTo-Json -Depth 10

2. Key metrics to check:

   a) ENTROPY SWEEP
      - Does fitness vary with diversity_weight?
      - Is 0.15 (ConfigC baseline) optimal?
      - Answer: How much tuning matters

   b) ABLATION TESTS
      - Which removal causes biggest fitness drop?
      - fitness_drop = baseline - ablated
      - Answer: Which components are critical

   c) INIT SENSITIVITY
      - Does sqrt_weighted beat uniform?
      - Answer: Is weighted initialization necessary?

   d) NEIGHBOR SAMPLING
      - Which team members have high peak sharpness?
      - Answer: Are specific team members critical?

3. Decision:
   - If ConfigC components all matter → Keep them (minimal viable set)
   - If one dominates → Simplify (remove others)
   - If room for improvement → Run Phase 2b (targeted tuning)

#>


# ============================================================================
# TROUBLESHOOTING
# ============================================================================

<#

If you see "Python was not found":
  - Verify Python installation: Test-Path 'C:\Users\rezas\AppData\Local\Python\bin\python.exe'
  - Or try: $pythonExe = (Get-Command python3).Source

If script takes too long (> 30 minutes):
  - Further reduce: population_size=80, generations=80 in create_quick_config()
  - Or disable neighbor_sampling: comment out run_neighbor_sampling() call

If memory issues:
  - Reduce: pokemon_df subset, or sample instead of full 50 swaps in neighbor_sampling

If results file not created:
  - Check: Test-Path 'reports\ga_results\ablation_study_20260304_171808\validation\phase2\results\03_ablation_sensitivity_results.json'
  - Or look at terminal output for error messages

#>

# ============================================================================
# REFERENCE: WHAT EACH EXPERIMENT MEASURES
# ============================================================================

<#

ENTROPY SWEEP (5 runs, ~2 min)
  Varies: diversity_weight ∈ [0.5, 0.35, 0.25, 0.15, 0.1]
  Measures: Sensitivity to diversity bonus magnitude
  Key Output: best_fitness values across weights
  Interpretation: Is there an optimal weight? Is ConfigC's 0.15 best?

ABLATION TESTS (5 runs, ~2.5 min)
  Test 1: No entropy bonus (diversity_weight=0)
  Test 2: No balance penalty (imbalance_lambda=0)
  Test 3: No weakness penalty (weakness_penalty=0)
  Test 4: Uniform init + no penalties (pure ConfigA)
  Measures: Individual component contribution
  Key Output: fitness drop per component
  Interpretation: Which removal hurts most? Which components are critical?

INIT SENSITIVITY (3 runs, ~1.5 min)
  Test 1: Uniform init + uniform mutation
  Test 2: Inverse weighted init + weighted mutation
  Test 3: Sqrt-weighted init + weighted mutation (ConfigC standard)
  Measures: Impact of archetype weighting on convergence
  Key Output: fitness & convergence_gen per method
  Interpretation: Does sqrt-weighted beat others? Worth the complexity?

NEIGHBOR SAMPLING (~3-5 min)
  For each of 6 team members:
    - Sample 50 Pokémon from same archetype
    - Swap member, evaluate fitness
    - Track max improvement possible
  Measures: Fitness landscape sharpness (locally)
  Key Output: max_delta & peak_sharpness per member
  Interpretation: Which members are critical (high sharpness)? Which are flexible (low)?

TOTAL: ~13-15 min + 3-5 min = 16-20 min (target < 30 min)

#>
