# Ablation Studies & Experimental Design

This document explains the experimental setup and key findings used to compare GA configurations in the Pokémon Team Optimization project.

## Motivation
Ablation studies help isolate the impact of initialization strategy, fitness shaping, and penalties on team quality.

## Key Components
- **Experimental design**: Compares multiple GA configurations instead of relying on one tuned run.
- **Configs**: Baseline, weighted initialization, full optimization, and inclusive/randomized generation.
- **Metrics**: Fitness, entropy, convergence behavior, diversity, and realism constraints.
- **Interpretation**: The goal is not only higher scores, but better team composition and usable results.

## Technical Details
- Runs are reproducible with fixed seeds and documented config templates.
- Results are summarized in the reports directory and the implementation writeups.
- The project now also uses queueing and session persistence, so UX behavior is part of the system design story, not just the algorithm story.

## Results
The experiments show how fitness weights and penalties influence team diversity and realism, and why the project moved toward a more balanced configuration rather than a purely score-maximizing one.

For details, see [workflow/reports/](../workflow/reports/).
