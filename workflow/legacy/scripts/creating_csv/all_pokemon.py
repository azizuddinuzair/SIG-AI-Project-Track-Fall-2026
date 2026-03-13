import requests
import pandas as pd
import pathlib
import itertools

POKEAPI_BASE_URL = "https://pokeapi.co/api/v2"
POKEMON_ENDPOINT = f"{POKEAPI_BASE_URL}/pokemon"

def get_json(session, url):
    response = session.get(url, timeout=20)
    response.raise_for_status()
    return response.json()

def get_all_pokemon_ids():
    """Returns all Pokémon IDs: base (1-1025) and alternate forms (10001-10325)."""
    return itertools.chain(range(1, 1026), range(10001, 10326))

def get_pokemon_row(session, pokemon_id):
    """
    Fetches the stats and types for a given Pokémon ID and returns a dictionary
    with the relevant information.
     - If the Pokémon has only one type, type2 will be set to None.
     - If there is an error fetching the data, it prints an error message and returns None.
    """
    url = f"{POKEMON_ENDPOINT}/{pokemon_id}"

    try:
        data = get_json(session, url)
    except requests.RequestException as error:
        print(f"Skipping ID {pokemon_id}: {error}")
        return None

    stats = {stat["stat"]["name"]: stat["base_stat"] for stat in data["stats"]}
    ordered_types = [t["type"]["name"] for t in sorted(data["types"], key=lambda t: t["slot"])]

    return {
        "id": data["id"],
        "name": data["name"],
        "type1": ordered_types[0] if ordered_types else None,
        "type2": ordered_types[1] if len(ordered_types) > 1 else None,
        "hp": stats.get("hp"),
        "attack": stats.get("attack"),
        "defense": stats.get("defense"),
        "special-attack": stats.get("special-attack"),
        "special-defense": stats.get("special-defense"),
        "speed": stats.get("speed"),
    }

def build_all_pokemon_dataframe():
    """
    Builds a DataFrame of all Pokémon with their stats and types.
    Fetches each Pokémon by numeric ID.
    """
    rows = []
    with requests.Session() as session:
        pokemon_ids = get_all_pokemon_ids()
        print("Fetching Pokémon data...")
        for pokemon_id in pokemon_ids:
            row = get_pokemon_row(session, pokemon_id)
            if row is not None:
                rows.append(row)

    df = pd.DataFrame(rows, columns=[
        "id", "name", "type1", "type2",
        "hp", "attack", "defense",
        "special-attack", "special-defense", "speed"
    ])
    return df

if __name__ == "__main__":
    # Data Folder: ../../../data (scripts moved to creating_csv subfolder)
    data_folder = pathlib.Path(__file__).resolve().parents[2] / "data"
    if not data_folder.exists():
        data_folder.mkdir()

    output_csv = data_folder / "all_pokemon_stats.csv"
    df = build_all_pokemon_dataframe()
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} Pokémon to: {output_csv}")