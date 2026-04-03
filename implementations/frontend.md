# Frontend (Streamlit) Implementation

This document explains the design and technical details of the Streamlit frontend for the Pokémon Team Optimization project.

## Motivation
The frontend provides an interactive interface for generating, analyzing, saving, and revisiting optimized Pokémon teams without freezing the page during long GA runs.

## Key Components
- **Mode-based UI**: Team Generator, Team Analyzer, Random Team, Pokémon Info, and View Generated Teams.
- **Async job submission**: GA runs are submitted to a bounded worker queue instead of blocking the page.
- **Live job feedback**: The app shows job id, running status, refresh controls, and the latest completed result.
- **Saved-team workflow**: Users can nickname, save, rename, delete, and clear session-scoped teams.
- **Polished presentation**: The app includes guided helper text, a branded sidebar link, and readable result cards.

## Technical Details
- Built with Streamlit and session state for UI persistence.
- Uses the shared GA job queue and job runner to keep CPU-heavy work off the UI thread.
- Uses the shared SQLite `TeamStore` for session-scoped saved teams.
- Renders result panels at the bottom so ongoing job status does not interrupt the active mode.

## Project Value
- Improves usability under long-running GA workloads.
- Makes generated results easier to compare and revisit.
- Creates a cleaner demo story for interviews and presentations.

For code, see [workflow/app/streamlit_app.py](../workflow/app/streamlit_app.py).
