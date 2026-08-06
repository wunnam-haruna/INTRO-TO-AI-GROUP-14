from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent

VALIDATED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "validated"
    / "validated_clustered_players.csv"
)

SELECTED_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "selected_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "similarity"


def load_validated_players(
    path: Path = VALIDATED_DATA_PATH,
) -> pd.DataFrame:
    """Load players with validated cluster and role labels."""

    if not path.exists():
        raise FileNotFoundError(
            f"Validated player dataset not found at: {path}\n"
            "Run src/validate_clusters.py first."
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(
            "The validated player dataset contains no rows."
        )

    required_columns = [
        "Player",
        "Squad",
        "Position_Group",
        "Cluster",
        "Final_Role_Name",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The validated dataset is missing required columns: "
            f"{missing_columns}"
        )

    return data


def load_selected_features(
    path: Path = SELECTED_FEATURES_PATH,
) -> list[str]:
    """Load the features used to build the player profiles."""

    if not path.exists():
        raise FileNotFoundError(
            f"Selected feature file not found at: {path}"
        )

    feature_data = pd.read_csv(path)

    if "Selected_Feature" not in feature_data.columns:
        raise ValueError(
            "selected_features.csv must contain a "
            "'Selected_Feature' column."
        )

    features = (
        feature_data["Selected_Feature"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if not features:
        raise ValueError(
            "No selected features were found."
        )

    return features


def get_standardised_columns(
    data: pd.DataFrame,
    selected_features: list[str],
) -> list[str]:
    """Return available z-score columns for similarity calculations."""

    standardised_columns = [
        f"{feature}_z"
        for feature in selected_features
    ]

    missing_columns = [
        column
        for column in standardised_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The validated dataset is missing standardised columns: "
            f"{missing_columns}"
        )

    return standardised_columns


def find_player_matches(
    data: pd.DataFrame,
    player_name: str,
) -> pd.DataFrame:
    """Find all player records whose names match the search text."""

    search_text = player_name.strip().lower()

    if not search_text:
        raise ValueError(
            "A player name must be provided."
        )

    matches = data[
        data["Player"]
        .astype(str)
        .str.lower()
        .str.contains(
            search_text,
            regex=False,
        )
    ].copy()

    return matches.reset_index(drop=True)


def select_player_record(
    matches: pd.DataFrame,
    squad_name: str | None = None,
) -> pd.Series:
    """Select one player record, optionally using the club name."""

    if matches.empty:
        raise ValueError(
            "No matching player was found."
        )

    if squad_name is not None:
        squad_text = squad_name.strip().lower()

        squad_matches = matches[
            matches["Squad"]
            .astype(str)
            .str.lower()
            .str.contains(
                squad_text,
                regex=False,
            )
        ]

        if squad_matches.empty:
            raise ValueError(
                "The player was found, but not for the specified squad."
            )

        return squad_matches.iloc[0]

    if len(matches) > 1:
        print(
            "\nMultiple matching player records were found. "
            "The first result will be used:"
        )

        print(
            matches[
                [
                    "Player",
                    "Squad",
                    "Position_Group",
                    "Final_Role_Name",
                ]
            ].to_string(index=False)
        )

    return matches.iloc[0]


def calculate_similarity_scores(
    data: pd.DataFrame,
    target_player: pd.Series,
    standardised_columns: list[str],
    same_cluster_only: bool = True,
) -> pd.DataFrame:
    """
    Calculate Euclidean distance from one player to comparable players.

    Players are compared only with others in the same broad position.
    By default, they must also belong to the same cluster.
    """

    comparable_players = data[
        data["Position_Group"]
        == target_player["Position_Group"]
    ].copy()

    if same_cluster_only:
        comparable_players = comparable_players[
            comparable_players["Cluster"]
            == target_player["Cluster"]
        ].copy()

    comparable_players = comparable_players[
        ~(
            (comparable_players["Player"] == target_player["Player"])
            & (comparable_players["Squad"] == target_player["Squad"])
        )
    ].copy()

    if comparable_players.empty:
        raise ValueError(
            "No comparable players were available."
        )

    target_vector = target_player[
        standardised_columns
    ].astype(float).to_numpy()

    comparison_matrix = comparable_players[
        standardised_columns
    ].astype(float).to_numpy()

    distances = np.linalg.norm(
        comparison_matrix - target_vector,
        axis=1,
    )

    comparable_players[
        "Similarity_Distance"
    ] = distances

    comparable_players[
        "Similarity_Score"
    ] = 1 / (1 + distances)

    return comparable_players.sort_values(
        by="Similarity_Distance",
        ascending=True,
    ).reset_index(drop=True)


def identify_strengths_and_weaknesses(
    player_record: pd.Series,
    selected_features: list[str],
    top_n: int = 4,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Identify strongest and weakest attributes from position z-scores.

    Positive z-scores are above the player's positional average.
    Negative z-scores are below the positional average.
    """

    rows: list[dict[str, float | str]] = []

    for feature in selected_features:
        standardised_column = f"{feature}_z"

        if standardised_column not in player_record.index:
            continue

        rows.append(
            {
                "Feature": feature,
                "Raw_Value": player_record.get(
                    feature,
                    np.nan,
                ),
                "Position_Z_Score": float(
                    player_record[standardised_column]
                ),
            }
        )

    attribute_data = pd.DataFrame(rows)

    if attribute_data.empty:
        raise ValueError(
            "No player attributes were available."
        )

    strengths = attribute_data.nlargest(
        top_n,
        "Position_Z_Score",
    ).reset_index(drop=True)

    weaknesses = attribute_data.nsmallest(
        top_n,
        "Position_Z_Score",
    ).reset_index(drop=True)

    return strengths, weaknesses


def create_player_profile(
    player_record: pd.Series,
    strengths: pd.DataFrame,
    weaknesses: pd.DataFrame,
) -> dict[str, object]:
    """Create a compact profile for display or API use."""

    profile = {
        "Player": player_record["Player"],
        "Squad": player_record["Squad"],
        "Competition": player_record.get(
            "Comp",
            pd.NA,
        ),
        "Position_Group": player_record[
            "Position_Group"
        ],
        "Original_Position": player_record.get(
            "Pos",
            pd.NA,
        ),
        "Cluster": int(
            player_record["Cluster"]
        ),
        "Role_Profile": player_record[
            "Final_Role_Name"
        ],
        "Minutes": player_record.get(
            "Min",
            pd.NA,
        ),
        "Strongest_Attributes": (
            strengths[
                [
                    "Feature",
                    "Raw_Value",
                    "Position_Z_Score",
                ]
            ].to_dict(
                orient="records"
            )
        ),
        "Weakest_Attributes": (
            weaknesses[
                [
                    "Feature",
                    "Raw_Value",
                    "Position_Z_Score",
                ]
            ].to_dict(
                orient="records"
            )
        ),
    }

    return profile


def save_similarity_results(
    target_player: pd.Series,
    similar_players: pd.DataFrame,
    strengths: pd.DataFrame,
    weaknesses: pd.DataFrame,
) -> None:
    """Save the latest player similarity analysis."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    safe_player_name = (
        str(target_player["Player"])
        .lower()
        .replace(" ", "_")
        .replace("/", "_")
    )

    similar_players.to_csv(
        OUTPUT_DIR
        / f"{safe_player_name}_similar_players.csv",
        index=False,
    )

    strengths.to_csv(
        OUTPUT_DIR
        / f"{safe_player_name}_strengths.csv",
        index=False,
    )

    weaknesses.to_csv(
        OUTPUT_DIR
        / f"{safe_player_name}_weaknesses.csv",
        index=False,
    )


def analyse_player(
    player_name: str,
    squad_name: str | None = None,
    number_of_similar_players: int = 10,
    same_cluster_only: bool = True,
) -> tuple[
    dict[str, object],
    pd.DataFrame,
]:
    """Run a complete player-profile and similarity analysis."""

    player_data = load_validated_players()
    selected_features = load_selected_features()

    standardised_columns = get_standardised_columns(
        player_data,
        selected_features,
    )

    matches = find_player_matches(
        player_data,
        player_name,
    )

    player_record = select_player_record(
        matches,
        squad_name,
    )

    similar_players = calculate_similarity_scores(
        player_data,
        player_record,
        standardised_columns,
        same_cluster_only=same_cluster_only,
    ).head(number_of_similar_players)

    strengths, weaknesses = (
        identify_strengths_and_weaknesses(
            player_record,
            selected_features,
        )
    )

    profile = create_player_profile(
        player_record,
        strengths,
        weaknesses,
    )

    save_similarity_results(
        player_record,
        similar_players,
        strengths,
        weaknesses,
    )

    return profile, similar_players


def print_player_analysis(
    profile: dict[str, object],
    similar_players: pd.DataFrame,
) -> None:
    """Print a readable player analysis in the terminal."""

    print("\nPlayer profile")
    print("-" * 50)

    print(
        f"Player: {profile['Player']}"
    )

    print(
        f"Squad: {profile['Squad']}"
    )

    print(
        f"Competition: {profile['Competition']}"
    )

    print(
        f"Position group: {profile['Position_Group']}"
    )

    print(
        f"Role profile: {profile['Role_Profile']}"
    )

    print(
        f"Cluster: {profile['Cluster']}"
    )

    print(
        f"Minutes: {profile['Minutes']}"
    )

    print("\nStrongest attributes:")

    for attribute in profile[
        "Strongest_Attributes"
    ]:
        print(
            f"- {attribute['Feature']}: "
            f"z = {attribute['Position_Z_Score']:.2f}"
        )

    print("\nWeakest attributes:")

    for attribute in profile[
        "Weakest_Attributes"
    ]:
        print(
            f"- {attribute['Feature']}: "
            f"z = {attribute['Position_Z_Score']:.2f}"
        )

    display_columns = [
        "Player",
        "Squad",
        "Position_Group",
        "Final_Role_Name",
        "Similarity_Distance",
        "Similarity_Score",
    ]

    available_display_columns = [
        column
        for column in display_columns
        if column in similar_players.columns
    ]

    print("\nMost similar players:")
    print(
        similar_players[
            available_display_columns
        ].to_string(index=False)
    )


def run_interactive_similarity_search() -> None:
    """Ask the user for a player and print similar alternatives."""

    print(
        "Football Player Similarity Engine"
    )

    print(
        "Enter part or all of a player's name."
    )

    player_name = input(
        "\nPlayer name: "
    ).strip()

    squad_name_input = input(
        "Squad name (optional): "
    ).strip()

    squad_name = (
        squad_name_input
        if squad_name_input
        else None
    )

    try:
        profile, similar_players = analyse_player(
            player_name=player_name,
            squad_name=squad_name,
            number_of_similar_players=10,
            same_cluster_only=True,
        )

        print_player_analysis(
            profile,
            similar_players,
        )

        print(
            f"\nResults saved to: {OUTPUT_DIR}"
        )

    except ValueError as error:
        print(
            f"\nAnalysis could not be completed: {error}"
        )


if __name__ == "__main__":
    run_interactive_similarity_search()