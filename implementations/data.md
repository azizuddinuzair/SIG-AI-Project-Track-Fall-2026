# Data Engineering & Clustering Implementation

This document explains the data engineering, feature creation, and clustering methods used in the project.

## Motivation
High-quality data and engineered features are essential for both team optimization and the interpretability story behind the project.

## Key Components
- **Raw data processing**: Cleans Pokémon stats, typing, archetypes, and supporting metadata.
- **Feature engineering**: Builds offensive, defensive, speed, bias, and type-defense representations.
- **Clustering**: Groups Pokémon into learned archetypes using unsupervised learning.
- **Fitness integration**: Feeds those derived features into GA scoring and team analysis.

## Technical Details
- Uses pandas and NumPy for data manipulation.
- Uses scikit-learn models and preprocessing for clustering pipelines.
- Keeps feature naming and schema consistent across Streamlit, CLI, and GA layers.
- The data pipeline is designed to support both experimentation and user-facing summaries.

## Results
The engineered data improves GA search quality, supports team explanations, and makes the project more interview-ready because the same data powers both optimization and analysis.

For code, see [workflow/src/ga/](../workflow/src/ga/) and [workflow/reports/](../workflow/reports/).
