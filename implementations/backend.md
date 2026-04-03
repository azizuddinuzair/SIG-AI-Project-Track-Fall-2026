# Backend Implementation

This document explains the supporting backend architecture for the Pokémon Team Optimization project.

## Motivation
The backend layer keeps CPU-heavy GA work, session-scoped persistence, and result assembly separate from the Streamlit UI so the app stays responsive and maintainable.

## Key Components
- **GA job runner**: Executes optimization work in a worker-friendly function.
- **Bounded job queue**: Limits concurrent and queued GA jobs so the app does not overload itself.
- **Session store**: Persists generated teams in SQLite with session isolation.
- **Shared result payloads**: Standardizes team, fitness, breakdown, and metadata formats for the UI and CLI.
- **Legacy CLI support**: Keeps the interactive CLI aligned with the same storage and payload behavior.

## Technical Details
- The GA worker uses a process pool because the workload is CPU-bound.
- SQLite is accessed through a thread-safe wrapper with per-operation locking.
- Job status is tracked so the UI can show queued, running, completed, and failed states.
- The architecture is intentionally lightweight and project-friendly rather than a full multi-service backend.

## Project Value
- Prevents UI blocking during long optimizations.
- Gives the project a clear separation between computation, persistence, and presentation.
- Supports a believable real-world workflow without requiring production infrastructure.

For code, see [workflow/src/ga/job_runner.py](../workflow/src/ga/job_runner.py), [workflow/src/ga/job_queue.py](../workflow/src/ga/job_queue.py), and [workflow/src/team_store.py](../workflow/src/team_store.py).
