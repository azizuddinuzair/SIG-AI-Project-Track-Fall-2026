"""
GA (Genetic Algorithm) module for team generation.
Includes optimization engine, fitness functions, and configuration.
"""

from .optimization import PokemonGA, load_pokemon_data
from .fitness import evaluate_fitness, compute_archetype_entropy
from .config import get_config_c

__all__ = [
    'PokemonGA',
    'load_pokemon_data',
    'evaluate_fitness',
    'compute_archetype_entropy',
    'get_config_c'
]
