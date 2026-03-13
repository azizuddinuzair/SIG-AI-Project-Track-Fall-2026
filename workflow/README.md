# Pokemon Team ML CLI

This project now exposes a user-facing command line interface for the core workflows.

## Setup

```bash
py -m pip install -r requirements.txt
```

## CLI Usage

Run all commands from `Proj1/`.

```bash
py scripts/cli.py --help
```

### Menu Mode (No Arguments)

Launch the user-facing menu:

```bash
py scripts/cli.py
```

Menu options currently implemented:
- `1. Team Generator`: generates top teams with bounded, practical ranges, optional locked Pokémon, and lets user pick one to inspect
- `2. Team Analyzer`: analyzes role/stat/type weaknesses for a team of 6 Pokemon
- `3. TBD`

### 1. Clustering Workflow

Run the clustering pipeline:

```bash
py scripts/cli.py cluster
```

Run clustering plus deliverables:

```bash
py scripts/cli.py cluster --with-deliverables
```

Run clustering plus deliverables and validation:

```bash
py scripts/cli.py cluster --with-deliverables --with-validation
```

### 2. GA Workflow

Default GA run (Config C, pop=300, gen=300):

```bash
py scripts/cli.py ga
```

Custom GA run:

```bash
py scripts/cli.py ga --config C --population 150 --generations 100 --seed 123 --top-n 10
```

Write outputs to a specific directory:

```bash
py scripts/cli.py ga --output-dir reports/ga_results/my_run
```

App-friendly mode (no filesystem writes, return JSON payload):

```bash
py scripts/cli.py ga --population 50 --generations 20 --top-n 5 --no-save --json-output
```

### 3. End-to-End Pipeline

Run cluster then GA:

```bash
py scripts/cli.py pipeline --config C --population 150 --generations 100
```

Run full pipeline with deliverables and validation:

```bash
py scripts/cli.py pipeline --with-deliverables --with-validation
```

Pipeline in app-friendly mode:

```bash
py scripts/cli.py pipeline --population 50 --generations 20 --no-save --json-output
```

## Output Files

GA command writes:
- `top_teams.json`
- `fitness_history.csv`
- `metadata.json`

By default, outputs go to `reports/ga_results/run_cli_<timestamp>/`.

## Scripts Organization

See `scripts/README.md` for categorized script paths:
- `scripts/ga/`
- `scripts/validation/`
- `scripts/analysis/`
- `scripts/utils/`

