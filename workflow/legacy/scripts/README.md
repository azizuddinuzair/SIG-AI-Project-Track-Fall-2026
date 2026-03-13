# Scripts Layout

This directory is organized by workflow area:

- `cli.py`: User-facing menu CLI (team generator, team analyzer, TBD menu slots)
- `ga/`: GA run/orchestration scripts
- `validation/`: validation and quality-check scripts
- `analysis/`: analysis utilities for clustering/ablation outputs
- `utils/`: maintenance helpers
- `creating_csv/`: dataset generation and feature prep scripts
- `test/`: clustering-focused script tests
- `build_role_move_priors.py`: role-prior builder (kept at top-level for compatibility)

## Common Commands

```bash
py scripts/cli.py
py scripts/ga/run_ga_601.py
py scripts/validation/validate_601_clustering.py
py scripts/analysis/cluster_analyzer.py
```
