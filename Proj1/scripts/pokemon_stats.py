import requests
import pandas as pd
import pathlib


POKEAPI_BASE_URL = "https://pokeapi.co/api/v2"
POKEMON_ENDPOINT = f"{POKEAPI_BASE_URL}/pokemon"
EVOLUTION_CHAIN_ENDPOINT = f"{POKEAPI_BASE_URL}/evolution-chain"





def get_json(session, url):
    response = session.get(url, timeout=20) # timeout of 20 seconds to avoid hanging requests
    response.raise_for_status()
    return response.json()


def extract_terminal_species(chain_node):
    """
    Recursive Function:
    - If the current node has no further evolutions, return its species name in a list.
    - If it does have evolutions, recursively call this function on each of the next node and add their results to the terminal_species list.
    """
    if not chain_node["evolves_to"]:
        return [chain_node["species"]["name"]]

    terminal_species = []
    for next_node in chain_node["evolves_to"]:
        terminal_species.extend(extract_terminal_species(next_node))
    return terminal_species


def get_all_fully_evolved_species(session):
    """
    Fetches all evolution chains and extracts the species names of fully evolved Pokémon (to be used to get stats and types later)
    """
    url = f"{EVOLUTION_CHAIN_ENDPOINT}?limit=2000" # set limit of 2000 to make sure we get all evo chains

    evolution_chain_list = get_json(session, url)

    fully_evolved_species = set() 
    for chain in evolution_chain_list["results"]:
        chain_data = get_json(session, chain["url"])
        fully_evolved_species.update(extract_terminal_species(chain_data["chain"]))

    return sorted(fully_evolved_species)


def get_pokemon_row(session, pokemon_name):
    """
    Fetches the stats and types for a given Pokémon name and returns a dictionary with the relevant information.
     - If the Pokémon has only one type, type2 will be set to None.
     - If there is an error fetching the data for a Pokémon, it will print an error message with the pokemon name and URL and return None
    """
    url = f"{POKEMON_ENDPOINT}/{pokemon_name.lower()}"

    try:
        data = get_json(session, url)
    except requests.RequestException as error:
        print(f"Skipping {pokemon_name}: {error}")
        return None

    stats = {stat["stat"]["name"]: stat["base_stat"] for stat in data["stats"]}
    ordered_types = [t["type"]["name"] for t in sorted(data["types"], key=lambda t: t["slot"])]

    return {
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


def build_fully_evolved_dataframe():
    """
    Builds a DataFrame of fully evolved Pokémon with their stats and types.
        - It first gets the list of fully evolved species names, then iterates through them to fetch their stats and types, and finally compiles everything into a DataFrame.
    """
    rows = []
    with requests.Session() as session:
        species_names = get_all_fully_evolved_species(session)
        for species_name in species_names:
            row = get_pokemon_row(session, species_name)
            if row is not None:
                rows.append(row)

    return pd.DataFrame(rows)

if __name__ == "__main__":

    # Data Folder: ../data (relative to this script)
    data_folder = pathlib.Path(__file__).resolve().parents[1] / "data"
    if not data_folder.exists():
        data_folder.mkdir()

    output_csv = data_folder / "fully_evolved_pokemon_stats.csv"
    df = build_fully_evolved_dataframe()
    df.to_csv(output_csv, index=False)

    print(f"Saved {len(df)} fully evolved Pokémon to: {output_csv}")

