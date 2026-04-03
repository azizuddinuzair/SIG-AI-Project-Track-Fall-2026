# PokeTeam Optimizer

Pokémon Team Optimization is a research-style project that uses a genetic algorithm to build competitive teams from Pokémon data. The repository is organized to show the full workflow: data preparation, optimization logic, a Streamlit interface, and detailed experiment reporting.

## Project Focus
- **Optimization**: evolve six-Pokémon teams with a custom GA.
- **Data Science**: clustering, feature engineering, and role discovery.
- **Presentation**: a Streamlit app for exploring teams and results.
- **Documentation**: reports and implementation notes that explain the work, not just the setup.

## Repository Map
- [`workflow/`](workflow/README.md): main project code, data, scripts, and experiment outputs.
- [`implementations/`](implementations/README.md): deeper explanations of the main technical pieces.
- [`workflow/reports/`](workflow/reports/README.md): ablation studies, results, and research artifacts.

## Run
- App: `py -m streamlit run workflow/app/streamlit_app.py`

For a more detailed walkthrough of the project structure and experiments, start with [`workflow/README.md`](workflow/README.md).
