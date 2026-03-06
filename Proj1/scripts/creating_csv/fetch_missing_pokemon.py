"""
Fetch missing Pokemon varieties using the PokeAPI varieties endpoint.

Behavior:
- Pokemon with SINGLE_FORM_ONLY (cosmetic/temporary forms): Always fetch one form
- All other Pokemon: Fetch based on --mode flag

Usage:
  py fetch_missing_pokemon.py              # Fetch all competitive forms (default)
  py fetch_missing_pokemon.py --mode preferred  # Fetch one preferred form each

Modes:
  all       Fetch every competitive form (respects SINGLE_FORM_ONLY filter)
  preferred Fetch one selected competitive form per base Pokemon
"""

import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
import requests

BASE_URL = "https://pokeapi.co/api/v2"

# Pokemon that should only fetch ONE form (cosmetic variants or temporary battle states)
# For all others in SKIPPED_POKEMON, fetch ALL varieties (they have competitive stat/type differences)
SINGLE_FORM_ONLY = {
    # Temporary battle forms (ability-triggered, revert after battle)
    "aegislash",      # Shield form only (blade is ability-triggered in battle)
    "eiscue",         # Ice Face form (Noice Face is temporary when ability activates)
    "mimikyu",        # Disguised form (Busted is temporary when disguise breaks)
    "minior",         # Meteor form (Core forms are temporary + colors are cosmetic)
    "morpeko",        # Full Belly mode (Hangry is ability-triggered each turn)
    "wishiwashi",     # Solo form (School is temporary when HP > 25%)
    
    # Cosmetic-only variants (identical stats/types)
    "dudunsparce",    # Two-segment vs three-segment (same stats)
    "maushold",       # Family-of-three vs family-of-four (same stats)
    "squawkabilly",   # Color variants (green/blue/yellow/white plumage, same stats)
    "tatsugiri",      # Color variants (curly/droopy/stretchy, same stats)
    "toxtricity",     # Amped vs Low Key (nature-based, identical stats)
    
    # Special cases requiring specific form
    "palafin",        # Fetch hero form only (zero form is pre-transformation)
    "zygarde",        # Fetch 50% form only (Complete is mid-battle transformation)
}

# Pokemon that SHOULD fetch multiple forms (different stats/types):
# - basculegion (male/female different stats)
# - darmanitan (standard vs zen: 140 Atk vs 140 SpA, 55 Def vs 105 Def)
# - deoxys (normal/attack/defense/speed all have drastically different stats)
# - enamorus (incarnate vs therian different stats)
# - giratina (altered vs origin different stats/typing)
# - gourgeist (small/average/large/super have different HP and Speed)
# - indeedee (male/female different stats)
# - keldeo (ordinary vs resolute same stats but both exist)
# - landorus/thundurus/tornadus (incarnate vs therian)
# - lycanroc (midday/midnight/dusk different stats)
# - meloetta (aria vs pirouette different stats/typing)
# - meowstic (male/female different stats)
# - oricorio (4 forms with different secondary types)
# - oinkologne (male/female)
# - pyroar (male/female cosmetic but varieties exist)
# - shaymin (land vs sky different stats/typing)
# - urshifu (single-strike vs rapid-strike different secondary types)
# - wormadam (plant/sandy/trash different types AND stats)
# - zygarde (10%: 216/100/121 | 50%: 108/100/121 | Complete: 216/100/121 - different HP/Def/SpD)

SKIPPED_POKEMON = [
    "aegislash", "basculegion", "darmanitan", "deoxys", "dudunsparce",
    "eiscue", "enamorus", "giratina", "gourgeist", "indeedee",
    "keldeo", "landorus", "lycanroc", "maushold", "meloetta",
    "meowstic", "mimikyu", "minior", "morpeko", "oinkologne",
    "oricorio", "palafin", "pyroar", "shaymin", "squawkabilly",
    "tatsugiri", "thundurus", "tornadus", "toxtricity", "urshifu",
    "wishiwashi", "wormadam", "zygarde",
]

# Used only when mode=preferred OR for single-form Pokemon
FORM_PREFERENCES = {
    "landorus": "landorus-therian",
    "tornadus": "tornadus-therian",
    "thundurus": "thundurus-therian",
    "enamorus": "enamorus-therian",
    "giratina": "giratina-origin",
    "deoxys": "deoxys-defense",
    "shaymin": "shaymin-sky",
    "darmanitan": "darmanitan-standard",  # Can fetch both if mode=all
    "meloetta": "meloetta-aria",
    "urshifu": "urshifu-rapid-strike",
    
    # Single-form Pokemon preferences (always used regardless of mode)
    "minior": "minior-red-meteor",       # Any meteor color works (red chosen arbitrarily)
    "palafin": "palafin-hero",           # Hero is the competitive form
    "aegislash": "aegislash-shield",     # Shield is default form
}

TYPE_EFFECTIVENESS = {
    "normal": {"rock": 0.5, "ghost": 0, "steel": 0.5},
    "fire": {"fire": 0.5, "grass": 0.5, "ice": 0.5, "bug": 0.5, "steel": 2, "water": 0.5, "ground": 1, "rock": 0.5},
    "water": {"fire": 2, "water": 0.5, "grass": 0.5, "ice": 2, "steel": 2, "ground": 2},
    "grass": {"fire": 0.5, "grass": 0.5, "water": 2, "ground": 2, "rock": 2, "ice": 0.5, "flying": 0.5, "bug": 0.5, "poison": 0.5, "steel": 0.5, "dark": 0.5, "fairy": 0.5},
    "electric": {"fire": 0.5, "water": 2, "grass": 0.5, "ice": 2, "flying": 2, "ground": 0, "steel": 2},
    "ice": {"fire": 0.5, "water": 0.5, "grass": 0.5, "ice": 0.5, "ground": 2, "flying": 2, "dragon": 2, "steel": 0.5},
    "fighting": {"normal": 2, "flying": 0.5, "rock": 2, "bug": 0.5, "dark": 2, "steel": 2, "ice": 2, "poison": 0.5, "psychic": 0.5, "fairy": 0.5, "ghost": 0},
    "poison": {"grass": 2, "poison": 0.5, "ground": 0.5, "rock": 0.5, "bug": 2, "ghost": 0.5, "steel": 0, "fairy": 2},
    "ground": {"fire": 2, "electric": 2, "poison": 2, "rock": 2, "grass": 0.5, "water": 1, "ice": 0.5, "flying": 0},
    "flying": {"fighting": 2, "bug": 2, "grass": 2, "rock": 0.5, "steel": 0.5, "electric": 0.5},
    "psychic": {"fighting": 2, "poison": 2, "dark": 0, "psychic": 0.5, "steel": 0.5},
    "bug": {"grass": 2, "fighting": 0.5, "poison": 0.5, "flying": 0.5, "psychic": 2, "ghost": 0.5, "dark": 2, "steel": 0.5, "fairy": 0.5, "fire": 0.5},
    "rock": {"normal": 0.5, "flying": 2, "fighting": 0.5, "ground": 0.5, "fire": 2, "ice": 2, "water": 0.5, "grass": 0.5, "steel": 0.5},
    "ghost": {"normal": 0, "flying": 1, "psychic": 2, "bug": 0.5, "ghost": 2, "dark": 0.5, "steel": 0.5},
    "dragon": {"dragon": 2, "steel": 0.5, "fairy": 0},
    "dark": {"fighting": 0.5, "psychic": 2, "dark": 0.5, "ghost": 2, "steel": 0.5, "fairy": 0.5},
    "steel": {"normal": 2, "flying": 2, "rock": 2, "bug": 1, "steel": 0.5, "grass": 0.5, "psychic": 0.5, "ice": 2, "dragon": 0.5, "fairy": 2, "poison": 0, "ground": 1, "fire": 0.5, "water": 0.5},
    "fairy": {"fighting": 2, "bug": 0.5, "dark": 2, "poison": 0.5, "steel": 0.5, "fire": 0.5},
}

ALL_TYPES = [
    "normal", "fire", "water", "grass", "electric", "ice", "fighting", "poison",
    "ground", "flying", "psychic", "bug", "rock", "ghost", "dragon", "dark", "steel", "fairy",
]


def get_type_defense_vector(type1, type2=None):
    vector = {}
    for attack_type in ALL_TYPES:
        multiplier = 1.0
        if type1 in TYPE_EFFECTIVENESS and attack_type in TYPE_EFFECTIVENESS[type1]:
            multiplier *= TYPE_EFFECTIVENESS[type1][attack_type]
        if type2 and type2 in TYPE_EFFECTIVENESS and attack_type in TYPE_EFFECTIVENESS[type2]:
            multiplier *= TYPE_EFFECTIVENESS[type2][attack_type]
        vector[f"type_defense_{attack_type}"] = multiplier
    return vector


def compute_features(row):
    hp = row["hp"]
    attack = row["attack"]
    defense = row["defense"]
    spattack = row["special-attack"]
    spdefense = row["special-defense"]

    offensive_index = attack + spattack
    defensive_index = hp * 0.5 + defense + spdefense
    speed_percentile = 0.0  # Recomputed for full dataset after merge
    bias = (attack - spattack) / (attack + spattack) if (attack + spattack) > 0 else 0.0

    features = {
        "offensive_index": offensive_index,
        "defensive_index": defensive_index,
        "speed_percentile": speed_percentile,
        "physical_special_bias": bias,
    }
    features.update(get_type_defense_vector(row["type1"], row.get("type2", None)))
    return features


def get_pokemon_varieties(base_name):
    try:
        url = f"{BASE_URL}/pokemon-species/{base_name}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()
        varieties = data.get("varieties", [])
        parsed = []
        for item in varieties:
            pokemon_obj = item.get("pokemon", {})
            if pokemon_obj.get("name"):
                parsed.append({
                    "name": pokemon_obj["name"],
                    "is_default": bool(item.get("is_default", False)),
                })
        return parsed
    except requests.exceptions.RequestException as exc:
        print(f"  x {base_name}: species lookup failed: {exc}")
        return []


def select_preferred_form(base_name, varieties):
    if not varieties:
        return None

    preferred = FORM_PREFERENCES.get(base_name)
    if preferred and any(v["name"] == preferred for v in varieties):
        return preferred

    # Competitive leaning fallback
    for suffix in ("-therian", "-defense", "-wash", "-origin", "-attack"):
        for v in varieties:
            if v["name"].endswith(suffix):
                return v["name"]

    non_default = [v["name"] for v in varieties if not v["is_default"]]
    if non_default:
        return non_default[0]

    return varieties[0]["name"]


def fetch_pokemon_stats(form_name):
    try:
        url = f"{BASE_URL}/pokemon/{form_name}"
        response = requests.get(url, timeout=10)
        response.raise_for_status()
        data = response.json()

        stats_map = {s["stat"]["name"]: s["base_stat"] for s in data.get("stats", [])}
        types = [t["type"]["name"] for t in data.get("types", [])]

        row = {
            "name": data.get("name", form_name),
            "type1": types[0] if types else "normal",
            "type2": types[1] if len(types) > 1 else np.nan,
            "hp": stats_map.get("hp", 0),
            "attack": stats_map.get("attack", 0),
            "defense": stats_map.get("defense", 0),
            "special-attack": stats_map.get("special-attack", 0),
            "special-defense": stats_map.get("special-defense", 0),
            "speed": stats_map.get("speed", 0),
        }
        row.update(compute_features(row))
        return row
    except requests.exceptions.RequestException as exc:
        print(f"  x {form_name}: stat fetch failed: {exc}")
        return None


def choose_forms(base_name, varieties, mode):
    # Check if this Pokemon should only have one form (cosmetic/temporary variants)
    if base_name in SINGLE_FORM_ONLY:
        preferred = select_preferred_form(base_name, varieties)
        return [preferred] if preferred else []
    
    # For Pokemon with competitive form differences, respect mode
    if mode == "preferred":
        preferred = select_preferred_form(base_name, varieties)
        return [preferred] if preferred else []
    return [v["name"] for v in varieties]


def main():
    parser = argparse.ArgumentParser(description="Fetch missing Pokemon forms via species varieties endpoint")
    parser.add_argument("--mode", choices=["all", "preferred"], default="all")
    args = parser.parse_args()

    csv_path = Path(__file__).parent.parent.parent / "reports" / "clustering_analysis" / "data" / "pokemon_with_clusters.csv"
    existing_df = pd.read_csv(csv_path)
    existing_names = set(existing_df["name"].tolist())

    print("=" * 80)
    print(f"Fetching missing Pokemon forms (mode={args.mode})")
    print("=" * 80)

    fetched_rows = []
    selected_forms_log = []

    for base_name in SKIPPED_POKEMON:
        print(f"\n{base_name}:")
        varieties = get_pokemon_varieties(base_name)
        if not varieties:
            print("  - no varieties found")
            continue

        form_names = choose_forms(base_name, varieties, args.mode)
        print(f"  - varieties: {', '.join(v['name'] for v in varieties)}")
        print(f"  - selected: {', '.join(form_names)}")

        for form_name in form_names:
            if form_name in existing_names:
                print(f"    = {form_name} already exists")
                continue
            row = fetch_pokemon_stats(form_name)
            if row is not None:
                fetched_rows.append(row)
                existing_names.add(form_name)
                print(f"    + added {form_name}")
            time.sleep(0.35)

        selected_forms_log.append({
            "base": base_name,
            "available_varieties": [v["name"] for v in varieties],
            "selected_forms": form_names,
        })

    if not fetched_rows:
        print("\nNo new forms fetched. Nothing to update.")
        return

    new_df = pd.DataFrame(fetched_rows)
    combined_df = pd.concat([existing_df, new_df], ignore_index=True)
    combined_df = combined_df.drop_duplicates(subset=["name"], keep="last")

    # Recompute speed percentile globally after adding forms.
    speed_ranks = combined_df["speed"].rank()
    combined_df["speed_percentile"] = speed_ranks / len(combined_df)

    # Clusters/archetypes must be recomputed with your existing clustering scripts.
    combined_df["cluster"] = np.nan
    combined_df["archetype"] = np.nan

    combined_df.to_csv(csv_path, index=False)

    log_path = Path(__file__).parent / "missing_pokemon_selected_forms.json"
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(selected_forms_log, f, indent=2)

    print("\n" + "=" * 80)
    print(f"Added new forms: {len(new_df)}")
    print(f"Updated dataset: {csv_path}")
    print(f"Selection log:   {log_path}")
    print("=" * 80)
    print("Next:")
    print("1. py Proj1/reports/clustering_analysis/scripts/01_generate_phase1_deliverables.py")
    print("2. py Proj1/reports/clustering_analysis/deliverables/validation/07_ground_truth_validation.py")


if __name__ == "__main__":
    main()
