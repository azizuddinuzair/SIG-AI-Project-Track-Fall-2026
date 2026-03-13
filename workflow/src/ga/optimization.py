"""
Genetic Algorithm for Pokémon Team Optimization

Core GA engine implementing:
- Weighted archetype initialization (uniform/inverse/sqrt)
- Tournament selection
- Two-point crossover with deduplication
- Weighted mutation
- Elitism preservation
- Generation-wise logging

Population evolves teams of 6 Pokémon toward fitness optimization.
"""

import pandas as pd
import numpy as np
import random
import json
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from .fitness import TYPE_NAMES, evaluate_fitness, compute_archetype_entropy, get_type_effectiveness


class PokemonGA:
    """
    Genetic Algorithm for optimizing 6-Pokémon competitive teams.
    
    Attributes:
        pokemon_df: Full Pokémon dataset with cluster assignments
        config: Configuration dictionary
        archetype_weights: Probability weights for archetype sampling
        population: Current generation of teams
        fitness_history: Per-generation statistics
    """
    
    def __init__(
        self,
        pokemon_df: pd.DataFrame,
        config: Dict,
        output_dir: Optional[Path] = None,
        locked_pokemon: Optional[List[str]] = None,
    ):
        """
        Initialize GA with Pokémon dataset and configuration.
        
        Args:
            pokemon_df: DataFrame with all Pokémon (must include 'archetype' column)
            config: Configuration dict from ga_config.py
            output_dir: Optional directory to save generation snapshots
            locked_pokemon: Optional list of Pokemon names that must remain in every team
        """
        self.pokemon_df = pokemon_df
        self.config = config
        self.output_dir = output_dir
        self.rng = np.random.default_rng(config['random_seed'])
        random.seed(config['random_seed'])

        self._pokemon_by_name = {
            row['name']: row for _, row in self.pokemon_df.iterrows()
        }
        self.locked_pokemon = self._validate_locked_pokemon(locked_pokemon or [])
        
        # Compute archetype weights based on initialization method
        self.archetype_weights = self._compute_archetype_weights()
        
        # Initialize population
        self.population = []
        self.fitness_scores = []
        self.fitness_history = []
        
        print(f"[GA] GA Initialized:")
        print(f"   Config: {config['name']}")
        print(f"   Population: {config['population']['size']}")
        print(f"   Generations: {config['population']['generations']}")
        print(f"   Initialization: {config['initialization']['method']}")
        print(f"   Random seed: {config['random_seed']}")
        if self.locked_pokemon:
            print(f"   Locked Pokémon: {', '.join(self.locked_pokemon)}")


    def _validate_locked_pokemon(self, locked_pokemon: List[str]) -> List[str]:
        """Validate locked pokemon names against dataset and duplicate entries."""
        normalized = [str(name).strip() for name in locked_pokemon if str(name).strip()]

        unique = []
        seen = set()
        for name in normalized:
            if name in seen:
                continue
            seen.add(name)
            unique.append(name)

        if len(unique) > 5:
            raise ValueError("At most 5 locked Pokémon are supported.")

        missing = [name for name in unique if name not in self._pokemon_by_name]
        if missing:
            raise ValueError(f"Locked Pokémon not found in dataset: {missing}")

        return unique


    def _build_locked_rows(self) -> List[pd.Series]:
        return [self._pokemon_by_name[name] for name in self.locked_pokemon]


    def _enforce_locked_pokemon(self, team_df: pd.DataFrame) -> pd.DataFrame:
        """Ensure all locked pokemon are included in the team exactly once."""
        if not self.locked_pokemon:
            return team_df.reset_index(drop=True)

        team = team_df.copy().reset_index(drop=True)
        current_names = team['name'].tolist()

        # Remove duplicate rows first while preserving order.
        dedup_rows = []
        seen = set()
        for _, row in team.iterrows():
            if row['name'] in seen:
                continue
            seen.add(row['name'])
            dedup_rows.append(row.to_dict())
        team = pd.DataFrame(dedup_rows).reset_index(drop=True)
        current_names = team['name'].tolist()

        missing_locked = [name for name in self.locked_pokemon if name not in current_names]
        for locked_name in missing_locked:
            replace_idx = None
            for idx, name in enumerate(current_names):
                if name not in self.locked_pokemon:
                    replace_idx = idx
                    break

            locked_row = self._pokemon_by_name[locked_name]
            if replace_idx is not None:
                team.iloc[replace_idx] = locked_row
                current_names[replace_idx] = locked_name
            elif len(team) < 6:
                team = pd.concat([team, pd.DataFrame([locked_row.to_dict()])], ignore_index=True)
                current_names.append(locked_name)
            else:
                # Should only happen if 6 locked pokemon are forced, which is prevented by validation.
                raise ValueError("No available slot to enforce locked Pokémon")

        # Fill missing slots if team dropped below 6 after deduplication.
        while len(team) < 6:
            archetype = self._sample_archetype()
            pokemon = self._sample_pokemon_from_archetype(archetype, exclude=current_names)
            team = pd.concat([team, pd.DataFrame([pokemon.to_dict()])], ignore_index=True)
            current_names.append(pokemon['name'])

        return team.iloc[:6].reset_index(drop=True)
    
    
    def _compute_archetype_weights(self) -> Dict[str, float]:
        """
        Compute archetype sampling weights based on initialization method.
        
        Methods:
        - uniform: Equal probability for all archetypes
        - inverse: 1 / archetype_size (strong bias toward rare)
        - sqrt_weighted: 1 / sqrt(archetype_size) (balanced bias)
        
        Returns:
            Dict mapping archetype name to selection probability
        """
        method = self.config['initialization']['method']
        archetype_counts = self.pokemon_df['archetype'].value_counts()
        
        if method == 'uniform':
            # Equal weights
            weights = {arch: 1.0 for arch in archetype_counts.index}
        
        elif method == 'inverse':
            # Inverse size weighting
            weights = {arch: 1.0 / count for arch, count in archetype_counts.items()}
        
        elif method == 'sqrt_weighted':
            # Square-root dampened inverse weighting
            weights = {arch: 1.0 / np.sqrt(count) for arch, count in archetype_counts.items()}
        
        else:
            raise ValueError(f"Unknown initialization method: {method}")
        
        # Normalize to sum to 1
        total = sum(weights.values())
        weights = {arch: w / total for arch, w in weights.items()}
        
        print(f"\n   Archetype weights ({method}):")
        for arch, weight in sorted(weights.items(), key=lambda x: -x[1]):
            count = archetype_counts[arch]
            print(f"      {arch:25s} ({count:3d} Pokemon) -> {weight:.4f}")
        
        return weights
    
    
    def _sample_archetype(self) -> str:
        """Sample an archetype using weighted probabilities."""
        archetypes = list(self.archetype_weights.keys())
        probs = list(self.archetype_weights.values())
        return self.rng.choice(archetypes, p=probs)
    
    
    def _sample_pokemon_from_archetype(self, archetype: str, exclude: List[str] = None) -> pd.Series:
        """
        Sample a random Pokémon from the specified archetype.
        
        Args:
            archetype: Target archetype name
            exclude: List of Pokémon names to exclude (avoid duplicates)
            
        Returns:
            Random Pokémon row from that archetype
        """
        candidates = self.pokemon_df[self.pokemon_df['archetype'] == archetype]
        
        if exclude:
            candidates = candidates[~candidates['name'].isin(exclude)]
        
        if len(candidates) == 0:
            # Fallback: Sample from entire pool if archetype exhausted
            candidates = self.pokemon_df[~self.pokemon_df['name'].isin(exclude)]
        
        return candidates.sample(n=1, random_state=self.rng.integers(0, 1000000)).iloc[0]
    
    
    def create_random_team(self) -> pd.DataFrame:
        """
        Create a random team of 6 Pokémon using weighted archetype sampling.
        
        Returns:
            DataFrame with 6 unique Pokémon
        """
        team_members = [row.to_dict() for row in self._build_locked_rows()]
        team_names = [row['name'] for row in self._build_locked_rows()]
        
        for _ in range(6 - len(team_names)):
            archetype = self._sample_archetype()
            pokemon = self._sample_pokemon_from_archetype(archetype, exclude=team_names)
            team_members.append(pokemon)
            team_names.append(pokemon['name'])

        return self._enforce_locked_pokemon(pd.DataFrame(team_members))
    
    
    def initialize_population(self):
        """Create initial population of random teams."""
        pop_size = self.config['population']['size']
        print(f"\n[*] Initializing population of {pop_size} teams...")
        
        self.population = []
        for i in range(pop_size):
            team = self.create_random_team()
            self.population.append(team)
            
            if (i + 1) % 50 == 0:
                print(f"   Generated {i + 1}/{pop_size} teams...")
        
        print(f"   [OK] Population initialized")
    
    
    def evaluate_population(self) -> List[Tuple[float, Dict]]:
        """
        Evaluate fitness for all teams in population.
        
        Returns:
            List of (fitness_score, breakdown_dict) tuples
        """
        # Track Pokemon usage for rarity bonus
        if self.config['fitness'].get('rarity_bonus_weight', 0.0) > 0:
            pokemon_usage = {}
            for team in self.population:
                for pokemon_name in team['name']:
                    pokemon_usage[pokemon_name] = pokemon_usage.get(pokemon_name, 0) + 1
            
            # Inject usage data into config for this evaluation
            self.config['pokemon_usage_counts'] = pokemon_usage
            self.config['population_size_tracker'] = len(self.population)
        
        fitness_scores = []
        for team in self.population:
            score, breakdown = evaluate_fitness(team, self.config)
            fitness_scores.append((score, breakdown))
        
        self.fitness_scores = fitness_scores
        return fitness_scores
    
    
    def tournament_selection(self, k: int = 3) -> pd.DataFrame:
        """
        Select one team using tournament selection.
        
        Args:
            k: Tournament size (number of random teams to compare)
            
        Returns:
            Winner team (highest fitness among k random teams)
        """
        tournament_indices = self.rng.choice(len(self.population), size=k, replace=False)
        tournament_fitness = [self.fitness_scores[i][0] for i in tournament_indices]
        winner_idx = tournament_indices[np.argmax(tournament_fitness)]
        return self.population[winner_idx].copy()
    
    
    def crossover(self, parent1: pd.DataFrame, parent2: pd.DataFrame) -> pd.DataFrame:
        """
        Two-point crossover with duplicate removal.
        
        Process:
        1. Select 2 random crossover points
        2. Swap middle segment between parents
        3. Remove duplicates by replacing with mutation-style sampling
        
        Args:
            parent1, parent2: Parent teams (6 Pokémon each)
            
        Returns:
            Child team (6 unique Pokémon)
        """
        # Two-point crossover
        points = sorted(self.rng.choice(range(1, 6), size=2, replace=False))
        pt1, pt2 = points
        
        # Create child by combining segments
        child_list = (
            parent1.iloc[:pt1].to_dict('records') +
            parent2.iloc[pt1:pt2].to_dict('records') +
            parent1.iloc[pt2:].to_dict('records')
        )
        
        # Remove duplicates
        seen_names = set()
        unique_child = []
        duplicates_removed = 0
        
        for pokemon_dict in child_list:
            if pokemon_dict['name'] not in seen_names:
                unique_child.append(pokemon_dict)
                seen_names.add(pokemon_dict['name'])
            else:
                duplicates_removed += 1
        
        # Fill missing slots with mutation-style sampling
        while len(unique_child) < 6:
            archetype = self._sample_archetype()
            pokemon = self._sample_pokemon_from_archetype(archetype, exclude=list(seen_names))
            unique_child.append(pokemon.to_dict())
            seen_names.add(pokemon['name'])
        
        child_df = pd.DataFrame(unique_child).reset_index(drop=True)
        return self._enforce_locked_pokemon(child_df)
    
    
    def mutate(self, team: pd.DataFrame) -> pd.DataFrame:
        """
        Mutate team by replacing one random Pokémon.
        
        Uses weighted archetype sampling if config.mutation.weighted == True.
        
        Args:
            team: Team to mutate
            
        Returns:
            Mutated team
        """
        mutation_rate = self.config['mutation']['rate']
        
        if self.rng.random() >= mutation_rate:
            return team  # No mutation
        
        # Select random non-locked position to mutate
        mutable_indices = [
            idx for idx, name in enumerate(team['name'].tolist())
            if name not in self.locked_pokemon
        ]
        if not mutable_indices:
            return self._enforce_locked_pokemon(team)

        mut_idx = int(self.rng.choice(mutable_indices))
        current_names = team['name'].tolist()
        
        # Select replacement Pokémon
        if self.config['mutation']['weighted']:
            archetype = self._sample_archetype()
            new_pokemon = self._sample_pokemon_from_archetype(archetype, exclude=current_names)
        else:
            # Uniform random selection
            candidates = self.pokemon_df[~self.pokemon_df['name'].isin(current_names)]
            new_pokemon = candidates.sample(n=1, random_state=self.rng.integers(0, 1000000)).iloc[0]
        
        # Replace row using column-aligned reconstruction.
        # pandas row assignment with iloc can raise LossySetitemError on mixed dtypes.
        team_mutated = team.copy().reset_index(drop=True)
        replacement = {
            col: new_pokemon[col] if col in new_pokemon.index else team_mutated.at[mut_idx, col]
            for col in team_mutated.columns
        }

        records = team_mutated.to_dict('records')
        records[mut_idx] = replacement
        team_mutated = pd.DataFrame(records, columns=team_mutated.columns)
        
        return self._enforce_locked_pokemon(team_mutated.reset_index(drop=True))
    
    
    def evolve_one_generation(self, generation: int):
        """
        Perform one generation of evolution.
        
        Steps:
        1. Evaluate current population
        2. Select elite individuals
        3. Generate offspring via tournament selection + crossover + mutation
        4. Replace population (keeping elites)
        5. Log statistics
        
        Args:
            generation: Current generation number
        """
        # Evaluate fitness
        self.evaluate_population()
        
        # Sort by fitness (descending)
        sorted_indices = np.argsort([score for score, _ in self.fitness_scores])[::-1]
        
        # Elitism: Keep top N individuals
        elitism = self.config['population']['elitism']
        elite_teams = [self.population[i].copy() for i in sorted_indices[:elitism]]
        
        # Generate offspring
        offspring = []
        pop_size = self.config['population']['size']
        crossover_rate = self.config['crossover']['rate']
        
        while len(offspring) < pop_size - elitism:
            # Tournament selection
            parent1 = self.tournament_selection(self.config['population']['tournament_k'])
            parent2 = self.tournament_selection(self.config['population']['tournament_k'])
            
            # Crossover
            if self.rng.random() < crossover_rate:
                child = self.crossover(parent1, parent2)
            else:
                child = parent1.copy()
            
            # Mutation
            child = self.mutate(child)
            
            offspring.append(child)
        
        # Replace population
        self.population = elite_teams + offspring
        
        # Re-evaluate for logging
        self.evaluate_population()
        self._log_generation_stats(generation)
    
    
    def _log_generation_stats(self, generation: int):
        """
        Log statistics for current generation.
        
        Metrics tracked:
        - Mean, max, std fitness
        - Mean entropy (archetype diversity)
        - Archetype distribution
        - Rare archetype representation
        """
        fitness_values = [score for score, _ in self.fitness_scores]
        
        mean_fitness = np.mean(fitness_values)
        max_fitness = np.max(fitness_values)
        std_fitness = np.std(fitness_values)
        
        # Compute entropy statistics
        entropies = [compute_archetype_entropy(team) for team in self.population]
        mean_entropy = np.mean(entropies)
        
        # Count archetype representation
        all_archetypes = []
        for team in self.population:
            all_archetypes.extend(team['archetype'].tolist())
        
        archetype_dist = Counter(all_archetypes)
        
        # Rare archetypes (Speed Sweeper, Defensive Wall)
        rare_archetypes = ['Speed Sweeper', 'Defensive Wall']
        teams_with_rare = sum(
            1 for team in self.population
            if any(arch in team['archetype'].values for arch in rare_archetypes)
        )
        rare_percent = 100 * teams_with_rare / len(self.population)
        
        # Store history
        stats = {
            'generation': generation,
            'mean_fitness': mean_fitness,
            'max_fitness': max_fitness,
            'std_fitness': std_fitness,
            'mean_entropy': mean_entropy,
            'rare_archetype_percent': rare_percent
        }
        
        # Add archetype distribution
        for arch, count in archetype_dist.items():
            stats[f'archetype_{arch.lower().replace(" ", "_")}'] = count
        
        self.fitness_history.append(stats)
        
        # Print progress every 10 generations
        if generation % 10 == 0 or generation == 1:
            print(f"   Gen {generation:3d}: "
                  f"Fitness={mean_fitness:.4f} (±{std_fitness:.4f}), "
                  f"Max={max_fitness:.4f}, "
                  f"Entropy={mean_entropy:.3f}, "
                  f"Rare%={rare_percent:.1f}")
    
    
    def _export_generation_snapshot(self, generation: int):
        """
        Export top-5 teams from current generation to JSON file.
        Called every 5 generations to build diverse data for role discovery.
        
        Args:
            generation: Current generation number
        """
        if self.output_dir is None:
            return
        
        best_teams = self.get_best_teams(5)
        teams_data = []
        
        for rank, (team, fitness, breakdown) in enumerate(best_teams, 1):
            team_dict = {
                'rank': rank,
                'fitness': fitness,
                'breakdown': breakdown,
                'pokemon': team[['name', 'archetype', 'type1', 'type2', 'hp', 'attack',
                               'defense', 'special-attack', 'special-defense', 'speed']].to_dict('records')
            }
            teams_data.append(team_dict)
        
        snapshot_path = self.output_dir / f"generation_elite_{generation}.json"
        with open(snapshot_path, 'w') as f:
            json.dump(teams_data, f, indent=2)
    
    
    def run(self) -> pd.DataFrame:
        """
        Run complete GA evolution.
        
        Returns:
            DataFrame of fitness history
        """
        print(f"\n[*] Starting evolution...")
        
        # Initialize population
        self.initialize_population()
        
        # Initial evaluation
        self.evaluate_population()
        self._log_generation_stats(0)
        
        # Evolution loop
        num_generations = self.config['population']['generations']
        for gen in range(1, num_generations + 1):
            self.evolve_one_generation(gen)
            # Export generation snapshot every 5 generations
            if gen % 5 == 0:
                self._export_generation_snapshot(gen)
        
        print(f"\n[OK] Evolution complete!")
        
        # Return fitness history
        return pd.DataFrame(self.fitness_history)
    
    
    def get_best_teams(self, n: int = 10) -> List[Tuple[pd.DataFrame, float, Dict]]:
        """
        Get top N teams from current population.
        
        Args:
            n: Number of teams to return
            
        Returns:
            List of (team, fitness, breakdown) tuples, sorted by fitness
        """
        # Sort by fitness
        sorted_indices = np.argsort([score for score, _ in self.fitness_scores])[::-1]
        
        best_teams = []
        for i in sorted_indices[:n]:
            team = self.population[i]
            fitness, breakdown = self.fitness_scores[i]
            best_teams.append((team, fitness, breakdown))
        
        return best_teams
    
    
    def export_results(self, output_dir: Path):
        """
        Export GA results to files.
        
        Saves:
        - fitness_history.csv: Per-generation statistics
        - best_teams.csv: Top 10 teams with fitness breakdown
        - config.txt: Configuration used
        
        Args:
            output_dir: Directory to save results
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        config_name = self.config['name']
        
        # Save fitness history
        history_df = pd.DataFrame(self.fitness_history)
        history_path = output_dir / f"{config_name}_fitness_history.csv"
        history_df.to_csv(history_path, index=False)
        print(f"   [OK] Saved fitness history: {history_path}")
        
        # Save best teams
        best_teams = self.get_best_teams(10)
        best_teams_data = []
        
        for rank, (team, fitness, breakdown) in enumerate(best_teams, 1):
            team_entry = {
                'rank': rank,
                'fitness': fitness,
                'team': ', '.join(team['name'].tolist()),
                'archetypes': ', '.join(team['archetype'].tolist()),
                **breakdown
            }
            best_teams_data.append(team_entry)
        
        best_teams_df = pd.DataFrame(best_teams_data)
        best_teams_path = output_dir / f"{config_name}_best_teams.csv"
        best_teams_df.to_csv(best_teams_path, index=False)
        print(f"   [OK] Saved best teams: {best_teams_path}")
        
        # Save config
        config_path = output_dir / f"{config_name}_config.txt"
        with open(config_path, 'w') as f:
            f.write(f"Configuration: {config_name}\n")
            f.write("=" * 60 + "\n\n")
            for section, params in self.config.items():
                if isinstance(params, dict):
                    f.write(f"{section}:\n")
                    for key, value in params.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                else:
                    f.write(f"{section}: {params}\n")
        print(f"   [OK] Saved config: {config_path}")


def load_pokemon_data() -> pd.DataFrame:
    """
    Load Pokémon dataset with cluster assignments.
    
    Returns:
        DataFrame with all 535 Pokémon
    """
    # Find project root
    current_file = Path(__file__).resolve()
    proj_root = current_file.parents[2]  # Up to Proj1/
    
    # Load clustered data
    data_path = proj_root / "reports" / "clustering_analysis" / "data" / "pokemon_with_clusters.csv"
    
    if not data_path.exists():
        raise FileNotFoundError(f"Missing clustered data: {data_path}")
    
    df = pd.read_csv(data_path)
    print(f"[DATA] Loaded {len(df)} Pokemon from: {data_path.name}")

    type_defense_cols = [col for col in df.columns if col.startswith('type_defense_')]
    if type_defense_cols:
        defensive_rank = pd.to_numeric(df['defensive_index'], errors='coerce').fillna(0.0).rank(pct=True)
        offensive_index = pd.to_numeric(df['offensive_index'], errors='coerce').fillna(0.0)
        speed = pd.to_numeric(df['speed'], errors='coerce').fillna(0.0)
        physical_special_bias = pd.to_numeric(df.get('physical_special_bias', 0.0), errors='coerce').fillna(0.0)

        resist_count = []
        immune_count = []
        weakness_count = []
        for _, row in df.iterrows():
            defending_types = [row['type1']]
            if pd.notna(row.get('type2')) and row.get('type2'):
                defending_types.append(row['type2'])

            resist = 0
            immune = 0
            weak = 0
            for attacking_type in TYPE_NAMES:
                effectiveness = get_type_effectiveness(attacking_type, defending_types)
                if effectiveness == 0.0:
                    immune += 1
                elif effectiveness <= 0.5:
                    resist += 1
                elif effectiveness > 1.0:
                    weak += 1
            resist_count.append(resist)
            immune_count.append(immune)
            weakness_count.append(weak)

        resist_count = pd.Series(resist_count, index=df.index, dtype=float)
        immune_count = pd.Series(immune_count, index=df.index, dtype=float)
        weakness_count = pd.Series(weakness_count, index=df.index, dtype=float)

        offense_score = (1.0 - ((offensive_index - 175.0).abs() / 95.0)).clip(lower=0.0, upper=1.0)
        fast_speed_score = (1.0 - ((speed - 100.0).abs() / 55.0)).clip(lower=0.0, upper=1.0)
        slow_speed_score = (1.0 - ((speed - 60.0).abs() / 40.0)).clip(lower=0.0, upper=1.0)
        slow_speed_score = slow_speed_score * (0.55 + 0.45 * defensive_rank)
        speed_score = np.maximum(fast_speed_score, slow_speed_score)

        type_utility_score = (
            (resist_count + (1.75 * immune_count) - (1.10 * weakness_count) + 1.0) / 10.0
        ).clip(lower=0.0, upper=1.0)

        if 'offense_to_bulk_ratio' in df.columns:
            offense_to_bulk = pd.to_numeric(df['offense_to_bulk_ratio'], errors='coerce').fillna(0.0)
            profile_score = (1.0 - ((offense_to_bulk - 0.85).abs() / 0.85)).clip(lower=0.0, upper=1.0)
        else:
            profile_score = (1.0 - (physical_special_bias.abs() * 0.6)).clip(lower=0.0, upper=1.0)

        fast_pivot_bonus = (
            (speed >= 88)
            & (speed <= 120)
            & (defensive_rank >= 0.45)
            & (offense_score >= 0.30)
        ).astype(float) * 0.10
        slow_pivot_bonus = (
            (speed <= 80)
            & (defensive_rank >= 0.70)
            & (offense_score >= 0.20)
        ).astype(float) * 0.12

        ability_bonus = pd.Series(0.0, index=df.index)
        abilities_path = proj_root / 'data' / 'pokemon_abilities.csv'
        if abilities_path.exists():
            abilities_df = pd.read_csv(abilities_path)
            ability_name_col = 'name' if 'name' in abilities_df.columns else None
            ability_cols = [col for col in abilities_df.columns if 'ability' in col.lower()]
            if ability_name_col and ability_cols:
                merged = df[['name']].merge(
                    abilities_df[[ability_name_col] + ability_cols],
                    left_on='name',
                    right_on=ability_name_col,
                    how='left',
                )
                ability_text = merged[ability_cols].fillna('').astype(str).agg(' '.join, axis=1).str.lower()
                ability_bonus += ability_text.str.contains('regenerator', regex=False).astype(float) * 0.15
                ability_bonus += ability_text.str.contains('intimidate', regex=False).astype(float) * 0.12
                ability_bonus += ability_text.str.contains('natural cure', regex=False).astype(float) * 0.08
                ability_bonus += ability_text.str.contains('magic guard', regex=False).astype(float) * 0.06
                ability_bonus = ability_bonus.clip(upper=0.18)

        pivot_score = (
            0.32 * defensive_rank +
            0.18 * speed_score +
            0.10 * offense_score +
            0.25 * type_utility_score +
            0.15 * profile_score +
            fast_pivot_bonus +
            slow_pivot_bonus +
            ability_bonus
        ).clip(lower=0.0, upper=1.0)

        df['pivot_bulk_score'] = defensive_rank
        df['pivot_speed_score'] = speed_score
        df['pivot_offense_score'] = offense_score
        df['pivot_type_utility_score'] = type_utility_score
        df['pivot_profile_score'] = profile_score
        df['pivot_ability_bonus'] = ability_bonus
        df['pivot_score'] = pivot_score
        df['pivot_style_hint'] = np.where(fast_speed_score >= slow_speed_score, 'fast', 'slow')
    
    # Verify required columns
    required_cols = ['name', 'archetype', 'cluster', 'offensive_index', 'defensive_index']
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    
    return df


if __name__ == "__main__":
    # Test GA with Config C
    from ga_config import get_config_c
    
    print("=" * 80)
    print("GENETIC ALGORITHM TEST RUN")
    print("=" * 80)
    
    # Load data
    pokemon_df = load_pokemon_data()
    
    # Get config
    config = get_config_c()
    config['population']['size'] = 20  # Small for testing
    config['population']['generations'] = 10
    
    # Run GA
    ga = PokemonGA(pokemon_df, config)
    history = ga.run()
    
    # Show best team
    print("\n" + "=" * 80)
    print("BEST TEAM:")
    print("=" * 80)
    best_team, fitness, breakdown = ga.get_best_teams(1)[0]
    print(best_team[['name', 'archetype', 'type1', 'type2']])
    print(f"\nFitness: {fitness:.4f}")
    print("Breakdown:")
    for key, value in breakdown.items():
        print(f"  {key:20s}: {value:7.4f}")
