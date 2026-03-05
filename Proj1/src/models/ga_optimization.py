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
from pathlib import Path
from typing import List, Dict, Tuple, Optional
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

from ga_fitness import evaluate_fitness, compute_archetype_entropy


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
    
    def __init__(self, pokemon_df: pd.DataFrame, config: Dict):
        """
        Initialize GA with Pokémon dataset and configuration.
        
        Args:
            pokemon_df: DataFrame with all Pokémon (must include 'archetype' column)
            config: Configuration dict from ga_config.py
        """
        self.pokemon_df = pokemon_df
        self.config = config
        self.rng = np.random.default_rng(config['random_seed'])
        random.seed(config['random_seed'])
        
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
            print(f"      {arch:25s} ({count:3d} Pokémon) → {weight:.4f}")
        
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
        team_members = []
        team_names = []
        
        for _ in range(6):
            archetype = self._sample_archetype()
            pokemon = self._sample_pokemon_from_archetype(archetype, exclude=team_names)
            team_members.append(pokemon)
            team_names.append(pokemon['name'])
        
        return pd.DataFrame(team_members).reset_index(drop=True)
    
    
    def initialize_population(self):
        """Create initial population of random teams."""
        pop_size = self.config['population']['size']
        print(f"\n🌱 Initializing population of {pop_size} teams...")
        
        self.population = []
        for i in range(pop_size):
            team = self.create_random_team()
            self.population.append(team)
            
            if (i + 1) % 50 == 0:
                print(f"   Generated {i + 1}/{pop_size} teams...")
        
        print(f"   ✓ Population initialized")
    
    
    def evaluate_population(self) -> List[Tuple[float, Dict]]:
        """
        Evaluate fitness for all teams in population.
        
        Returns:
            List of (fitness_score, breakdown_dict) tuples
        """
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
        
        return pd.DataFrame(unique_child).reset_index(drop=True)
    
    
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
        
        # Select random position to mutate
        mut_idx = self.rng.integers(0, 6)
        current_names = team['name'].tolist()
        
        # Select replacement Pokémon
        if self.config['mutation']['weighted']:
            archetype = self._sample_archetype()
            new_pokemon = self._sample_pokemon_from_archetype(archetype, exclude=current_names)
        else:
            # Uniform random selection
            candidates = self.pokemon_df[~self.pokemon_df['name'].isin(current_names)]
            new_pokemon = candidates.sample(n=1, random_state=self.rng.integers(0, 1000000)).iloc[0]
        
        # Replace
        team_mutated = team.copy()
        team_mutated.iloc[mut_idx] = new_pokemon
        
        return team_mutated.reset_index(drop=True)
    
    
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
    
    
    def run(self) -> pd.DataFrame:
        """
        Run complete GA evolution.
        
        Returns:
            DataFrame of fitness history
        """
        print(f"\n🚀 Starting evolution...")
        
        # Initialize population
        self.initialize_population()
        
        # Initial evaluation
        self.evaluate_population()
        self._log_generation_stats(0)
        
        # Evolution loop
        num_generations = self.config['population']['generations']
        for gen in range(1, num_generations + 1):
            self.evolve_one_generation(gen)
        
        print(f"\n✅ Evolution complete!")
        
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
        print(f"   ✓ Saved fitness history: {history_path}")
        
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
        print(f"   ✓ Saved best teams: {best_teams_path}")
        
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
        print(f"   ✓ Saved config: {config_path}")


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
