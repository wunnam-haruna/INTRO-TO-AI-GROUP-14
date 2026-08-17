from __future__ import annotations

from collections.abc import Iterable

import pandas as pd


FEATURED_PLAYER_NAMES: tuple[str, ...] = (
    # Premier League and major English-club players in the dataset
    "Erling Haaland",
    "Mohamed Salah",
    "Bukayo Saka",
    "Cole Palmer",
    "Phil Foden",
    "Rodri",
    "Bruno Fernandes",
    "Declan Rice",
    "Martin Ødegaard",
    "William Saliba",
    "Virgil van Dijk",
    "Alexis Mac Allister",
    "Florian Wirtz",
    "Viktor Gyökeres",
    "Xavi Simons",
    "Rayan Cherki",
    # La Liga
    "Kylian Mbappé",
    "Vinicius Júnior",
    "Jude Bellingham",
    "Federico Valverde",
    "Trent Alexander-Arnold",
    "Lamine Yamal",
    "Pedri",
    "Raphinha",
    "Robert Lewandowski",
    "Frenkie de Jong",
    "Dani Olmo",
    "Antoine Griezmann",
    "Julián Álvarez",
    "Nico Williams",
    # Bundesliga
    "Harry Kane",
    "Michael Olise",
    "Joshua Kimmich",
    "Luis Díaz",
    "Serhou Guirassy",
    "Jonathan Tah",
    "Dayot Upamecano",
    "Kim Min-jae",
    "Patrik Schick",
    # Serie A
    "Lautaro Martínez",
    "Rafael Leão",
    "Christian Pulisic",
    "Nicolò Barella",
    "Hakan Çalhanoğlu",
    "Alessandro Bastoni",
    "Marcus Thuram",
    "Denzel Dumfries",
    "Federico Dimarco",
    "Dušan Vlahović",
    "Teun Koopmeiners",
    "Scott McTominay",
    "Kevin De Bruyne",
    "Paulo Dybala",
    "Kenan Yıldız",
    "Jonathan David",
    "Adrien Rabiot",
    "Lorenzo Pellegrini",
    # Ligue 1 / PSG and other recognisable players
    "Ousmane Dembélé",
    "Khvicha Kvaratskhelia",
    "Achraf Hakimi",
    "Nuno Mendes",
    "Désiré Doué",
    "João Neves",
    "Bradley Barcola",
    "Marquinhos",
    "Warren Zaïre-Emery",
    "Gonçalo Ramos",
    "Mason Greenwood",
    "Pierre-Emerick Aubameyang",
    "Corentin Tolisso",
    "Benjamin Pavard",
)


FEATURED_CLUBS: tuple[str, ...] = (
    "Arsenal",
    "Chelsea",
    "Liverpool",
    "Manchester City",
    "Manchester Utd",
    "Tottenham Hotspur",
    "Real Madrid",
    "Barcelona",
    "Atlético Madrid",
    "Athletic Club",
    "Bayern Munich",
    "Dortmund",
    "Leverkusen",
    "Inter",
    "Milan",
    "Juventus",
    "Napoli",
    "Roma",
    "Paris Saint-Germain",
    "Marseille",
    "Lyon",
)


LEAGUE_DISPLAY_NAMES: dict[str, str] = {
    "eng Premier League": "Premier League",
    "es La Liga": "La Liga",
    "it Serie A": "Serie A",
    "de Bundesliga": "Bundesliga",
    "fr Ligue 1": "Ligue 1",
}


LEAGUE_SHORT_NAMES: dict[str, str] = {
    "eng Premier League": "ENG",
    "es La Liga": "ESP",
    "it Serie A": "ITA",
    "de Bundesliga": "GER",
    "fr Ligue 1": "FRA",
}


GAME_MODES: tuple[str, ...] = (
    "Star Match",
    "Scout Mode",
    "Blind Scout",
)


def _initials(name: str) -> str:
    parts = [part for part in str(name).replace("-", " ").split() if part]
    if not parts:
        return "?"
    if len(parts) == 1:
        return parts[0][:2].upper()
    return f"{parts[0][0]}{parts[-1][0]}".upper()


def add_game_metadata(data: pd.DataFrame) -> pd.DataFrame:
    """Add display and popularity metadata used by the game UI."""

    prepared = data.copy()

    required = {"Player", "Squad", "Comp", "Position_Group", "Cluster"}
    missing = sorted(required.difference(prepared.columns))
    if missing:
        raise ValueError(f"Game data is missing required columns: {missing}")

    prepared["Player"] = prepared["Player"].astype(str).str.strip()
    prepared["Squad"] = prepared["Squad"].astype(str).str.strip()
    prepared["Comp"] = prepared["Comp"].astype(str).str.strip()

    prepared["Player_Key"] = (
        prepared["Player"]
        + " || "
        + prepared["Squad"]
        + " || "
        + prepared["Comp"]
    )
    prepared["League_Display"] = prepared["Comp"].map(
        LEAGUE_DISPLAY_NAMES
    ).fillna(prepared["Comp"])
    prepared["League_Short"] = prepared["Comp"].map(
        LEAGUE_SHORT_NAMES
    ).fillna("EUR")
    prepared["Initials"] = prepared["Player"].map(_initials)
    prepared["Is_Featured_Player"] = prepared["Player"].isin(
        FEATURED_PLAYER_NAMES
    )
    prepared["Is_Featured_Club"] = prepared["Squad"].isin(FEATURED_CLUBS)

    prepared["Popularity_Tier"] = "Database"
    prepared.loc[prepared["Is_Featured_Club"], "Popularity_Tier"] = (
        "Major Club"
    )
    prepared.loc[prepared["Is_Featured_Player"], "Popularity_Tier"] = (
        "Featured Star"
    )

    return prepared


def available_featured_players(data: pd.DataFrame) -> pd.DataFrame:
    """Return featured names that are genuinely present in the loaded data."""

    prepared = add_game_metadata(data) if "Player_Key" not in data else data.copy()
    return prepared[prepared["Is_Featured_Player"]].copy()


def get_target_pool(data: pd.DataFrame, game_mode: str) -> pd.DataFrame:
    """Return the target-player pool for the selected game mode."""

    prepared = add_game_metadata(data) if "Player_Key" not in data else data.copy()

    if game_mode == "Star Match":
        star_pool = prepared[prepared["Is_Featured_Player"]].copy()
        if len(star_pool) >= 10:
            return star_pool

        club_pool = prepared[prepared["Is_Featured_Club"]].copy()
        if len(club_pool) >= 10:
            return club_pool

    return prepared


def clean_league_name(raw_competition: str) -> str:
    return LEAGUE_DISPLAY_NAMES.get(str(raw_competition), str(raw_competition))


def filter_existing_names(
    data: pd.DataFrame,
    names: Iterable[str],
) -> list[str]:
    existing = set(data["Player"].astype(str))
    return [name for name in names if name in existing]
