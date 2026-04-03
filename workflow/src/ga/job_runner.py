from __future__ import annotations

import contextlib
import copy
import io
import json
from datetime import datetime
from typing import Any
from uuid import uuid4

import pandas as pd

from src.ga import PokemonGA, load_pokemon_data
from src.ga.config import get_config_a, get_config_b, get_config_c, get_config_random


PIVOT_CANDIDATE_THRESHOLD = 0.62


def _config_from_name(name: str) -> dict[str, Any]:
    options = {
        "A": get_config_a,
        "B": get_config_b,
        "C": get_config_c,
        "Random": get_config_random,
    }
    return copy.deepcopy(options[name]())


def _build_composition_presets() -> dict[str, dict[str, int]]:
    return {
        "balanced": {
            "Balanced All-Rounder": 2,
            "Generalist": 2,
            "Fast Attacker": 1,
            "Defensive Tank": 1,
        },
        "hyper_offense": {
            "Speed Sweeper": 2,
            "Fast Attacker": 2,
            "Physical Attacker": 1,
            "Generalist": 1,
        },
        "pivot_pressure": {
            "Generalist": 3,
            "Balanced All-Rounder": 1,
            "Fast Attacker": 1,
            "Defensive Tank": 1,
        },
        "bulky_offense": {
            "Defensive Tank": 2,
            "Balanced All-Rounder": 2,
            "Fast Attacker": 1,
            "Physical Attacker": 1,
        },
    }


def _normalize_composition_target(raw: dict[str, int], data_df: pd.DataFrame) -> dict[str, int]:
    available = set(data_df["archetype"].dropna().astype(str).unique().tolist())
    return {k: int(v) for k, v in raw.items() if int(v) > 0 and k in available}


def _power_mode_config(mode: str) -> dict[str, float]:
    mapping = {
        "standard": {"bst_cap": 3300, "bst_penalty_weight": 2.0},
        "competitive_strict": {"bst_cap": 3200, "bst_penalty_weight": 3.0},
        "open": {"bst_cap": 0, "bst_penalty_weight": 0.0},
    }
    return mapping[mode]


def _serialize_team(team_df: pd.DataFrame) -> list[dict[str, Any]]:
    cols = [
        "name",
        "archetype",
        "type1",
        "type2",
        "hp",
        "attack",
        "defense",
        "special-attack",
        "special-defense",
        "speed",
    ]
    safe_cols = [c for c in cols if c in team_df.columns]
    return team_df[safe_cols].to_dict("records")


def build_ga_job_request(
    *,
    config_name: str,
    population: int,
    generations: int,
    seed: int,
    top_n: int,
    locked_names: list[str],
    composition_name: str = "balanced",
    composition_weight: float = 0.20,
    power_mode: str = "standard",
) -> dict[str, Any]:
    return {
        "job_id": uuid4().hex,
        "config_name": config_name,
        "population": int(population),
        "generations": int(generations),
        "seed": int(seed),
        "top_n": int(top_n),
        "locked_names": list(locked_names),
        "composition_name": composition_name,
        "composition_weight": float(composition_weight),
        "power_mode": power_mode,
    }


def run_ga_job(job_request: dict[str, Any]) -> dict[str, Any]:
    config_name = str(job_request["config_name"])
    population = int(job_request["population"])
    generations = int(job_request["generations"])
    seed = int(job_request["seed"])
    top_n = int(job_request["top_n"])
    locked_names = list(job_request.get("locked_names", []))
    composition_name = str(job_request.get("composition_name", "balanced"))
    composition_weight = float(job_request.get("composition_weight", 0.20))
    power_mode = str(job_request.get("power_mode", "standard"))

    config = _config_from_name(config_name)
    config["name"] = f"Streamlit_{config_name}"
    config["population"]["size"] = population
    config["population"]["generations"] = generations
    config["random_seed"] = seed

    data_df = load_pokemon_data()
    presets = _build_composition_presets()
    target_map = _normalize_composition_target(presets[composition_name], data_df)
    power_cfg = _power_mode_config(power_mode)

    config["fitness"]["composition_weight"] = float(composition_weight)
    config["fitness"]["target_archetype_counts"] = target_map
    config["fitness"]["bst_cap"] = int(power_cfg["bst_cap"])
    config["fitness"]["bst_penalty_weight"] = float(power_cfg["bst_penalty_weight"])
    config["fitness"]["pivot_weight"] = 0.0
    config["fitness"]["target_pivot_count"] = 0
    config["fitness"]["pivot_threshold"] = PIVOT_CANDIDATE_THRESHOLD
    if composition_name == "pivot_pressure":
        config["fitness"]["pivot_weight"] = 0.18
        config["fitness"]["target_pivot_count"] = 3

    stdout_buffer = io.StringIO()
    stderr_buffer = io.StringIO()
    with contextlib.redirect_stdout(stdout_buffer), contextlib.redirect_stderr(stderr_buffer):
        ga = PokemonGA(data_df, config, output_dir=None, locked_pokemon=locked_names)
        history_df = ga.run()
        best_teams = ga.get_best_teams(top_n)

    stderr_output = stderr_buffer.getvalue()
    run_log = stdout_buffer.getvalue()
    if stderr_output:
        run_log += "\n--- warnings ---\n" + stderr_output

    run_context = {
        "run_id": job_request.get("job_id", uuid4().hex),
        "config_name": config_name,
        "population": population,
        "generations": generations,
        "seed": seed,
        "top_n": top_n,
        "locked_names": list(locked_names),
        "composition_name": composition_name,
        "composition_weight": composition_weight,
        "power_mode": power_mode,
        "timestamp": datetime.now().isoformat(),
    }

    top_teams_payload = []
    for rank, (team_df, fitness, breakdown) in enumerate(best_teams, start=1):
        top_teams_payload.append(
            {
                "rank": rank,
                "fitness": float(fitness),
                "breakdown": breakdown,
                "pokemon": _serialize_team(team_df),
            }
        )

    return {
        "run_log": run_log,
        "history": history_df,
        "best_fitness": float(best_teams[0][1]) if best_teams else None,
        "top_teams": top_teams_payload,
        "run_context": run_context,
        "config_used": config,
    }