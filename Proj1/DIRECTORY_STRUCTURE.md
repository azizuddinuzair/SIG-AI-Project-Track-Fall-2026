# Directory Structure Guide

## Consolidated Validation Phases (Option A Implemented)

All phases 1-5 are now consolidated under the ablation study directory for consistency.

### Current Layout

```
Proj1/
├── PHASE3_STRATEGY.md                          (top-level docs - now outdated)
├── validation/                                  (now empty - can be removed)
│
└── reports/
    └── ga_results/
        └── ablation_study_20260304_171808/
            ├── PHASE_1_COMPLETE_RESULTS.md
            ├── PHASE_3_COMPLETE_RESULTS.md
            ├── PHASE_4_COMPLETE_RESULTS.md
            ├── PHASE_5_COMPLETE_RESULTS.md
            ├── READY_TO_RUN.md
            ├── INDEX.md
            └── validation/                     ← All phases consolidated here
                ├── phase1/
                │   ├── scripts/
                │   │   ├── 01_random_baseline.py
                │   │   ├── 02_multiseed.py
                │   │   └── 03_fitness_validator.py
                │   ├── results/
                │   │   ├── 01_random_baseline_results.json
                │   │   ├── 02_multiseed_results.json
                │   │   └── 03_fitness_validator_results.json
                │   └── reports/
                │       └── README.md
                ├── phase2/
                │   ├── scripts/
                │   │   └── 03_ablation_sensitivity.py
                │   ├── results/
                │   │   └── 03_ablation_sensitivity_results.json
                │   └── reports/
                │       ├── README.md
                │       └── PHASE2_STRATEGY.md
                ├── phase3/
                │   ├── scripts/
                │   │   └── 04_system_validation.py
                │   ├── results/
                │   │   └── 04_system_validation_results.json
                │   └── reports/
                │       └── phase3_log.txt
                ├── phase4/
                │   ├── scripts/
                │   │   └── 05_calibration_sweep.py
                │   ├── results/
                │   └── reports/
                └── phase5/
                    ├── scripts/
                    │   └── 06_final_validation.py
                    ├── results/
                    │   └── 06_final_validation_results.json
                    └── reports/
```

## Why This Structure?

All 5 validation phases are part of a single **ablation study campaign** (dated 2026-03-04):
- Phase 1: Baseline confidence checks
- Phase 2: Ablation & sensitivity analysis
- Phase 3: System validation
- Phase 4: Calibration sweep planning
- Phase 5: Final confirmation

Consolidating them keeps the entire study together and makes it easier to navigate and reproduce.

## How to Navigate

### From Project Root (`Proj1/`)

All phases are now under the same validation tree:

```bash
cd reports/ga_results/ablation_study_20260304_171808/validation/
```

Then run any phase:

```bash
# Phase 1
python phase1/scripts/03_fitness_validator.py

# Phase 2
python phase2/scripts/03_ablation_sensitivity.py

# Phase 3
python phase3/scripts/04_system_validation.py

# Phase 4
python phase4/scripts/05_calibration_sweep.py

# Phase 5
python phase5/scripts/06_final_validation.py
```

### All Complete Results

All phase summary markdown files are co-located:
```
Proj1/reports/ga_results/ablation_study_20260304_171808/
├── PHASE_1_COMPLETE_RESULTS.md
├── PHASE_3_COMPLETE_RESULTS.md
├── PHASE_4_COMPLETE_RESULTS.md
└── PHASE_5_COMPLETE_RESULTS.md
```

## Key Points for Cross-Machine Compatibility

✅ **All paths in markdown files are relative** — use `./relative/path` or `../` navigation  
✅ **All Python scripts use `pathlib.Path(__file__)` for self-location** — scripts find themselves  
✅ **No user-specific paths** (like `C:\Users\rezas\...`) in any documentation  

When cloning this repo on another machine:
1. Python scripts will auto-locate using `Path(__file__)`
2. Markdown files reference relative paths (cd to appropriate dir and run)
3. All output directories use relative paths from scripts

## Cleanup Note

The root `Proj1/validation/` directory is now empty and can be removed if desired, as all validation phases have been consolidated under `reports/ga_results/ablation_study_20260304_171808/validation/`.

