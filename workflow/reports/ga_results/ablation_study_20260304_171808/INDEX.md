# Ablation Study 2026-03-04 — Directory Index

## Structure
```
ablation_study_20260304_171808/
├── data/              Raw results from GA runs
├── plots/             Visualizations (PNG)
├── reports/           Analysis reports
├── docs/              Documentation & validation guides
└── validation/        (Will be created) Phase 1-3 test results
```

## Quick Navigation

### 📊 Data (`data/`)
- **ablation_summary.csv** — Summary metrics for all 3 configs
- **Config[A/B/C]_{*}_fitness_history.csv** — Fitness per generation
- **Config[A/B/C]_{*}_best_teams.csv** — Top 10 teams per config
- **statistical_tests.csv** — T-tests and effect sizes

### 📈 Plots (`plots/`)
- **convergence_comparison.png** — Fitness over 250 generations (all configs)
- **archetype_distribution.png** — How many Pokémon from each archetype
- **rare_archetype_trends.png** — Speed Sweeper, Wall, Pivot prevalence

### 📋 Reports (`reports/`)
- **analysis_report.txt** — Initial ablation study findings

### 📚 Docs (`docs/`)
- **README.md** — Choose your validation level (START HERE)
- **VALIDATION_QUICKSTART.md** — Phase 1 tests & decision tree
- **VALIDATION_FRAMEWORK.md** — All 9 tests with code examples
- **SKEPTICISM_ANALYSIS.md** — My corrected assessment
- **NEXT_EXPERIMENTS.md** — Future analysis ideas

## Current Status

**Ablation Study**: ✅ Complete (3 configs)
- ConfigA (Baseline): fitness = 0.7239
- ConfigB (Inverse init): fitness = 0.6488
- ConfigC (Full diversity): fitness = **0.7324** ← Winner

**Validation Status**: 🚀 Starting Phase 1
- [ ] Random baseline test
- [ ] Multi-seed validation
- [ ] Fitness validator
- [ ] Phase 2+ (if Phase 1 passes)

## How to Proceed

1. **Read**: `docs/README.md` (5 minutes)
2. **Choose**: Quick (1h), Comprehensive (2.5h), or Everything (3h)
3. **Run**: Phase 1 tests (automated execution guides provided)
4. **Review**: Results will appear in `validation/phase1/results/` directory

---

**Last updated**: 2026-03-04 17:18:08  
**Next step**: Start Phase 1 validation tests
