from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import adjusted_rand_score


PROJECT_ROOT = Path(__file__).resolve().parent.parent

CLUSTERED_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "clustered"
    / "clustered_outfield_players.csv"
)

CENTROID_PATH = (
    PROJECT_ROOT
    / "data"
    / "clustered"
    / "cluster_centroid_profiles.csv"
)

SELECTED_FEATURES_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "selected_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "validated"

RANDOM_STATE = 42

TEST_SEEDS = [
    0,
    1,
    7,
    21,
    42,
    77,
    100,
    2026,
]


def load_clustered_data(
    path: Path = CLUSTERED_DATA_PATH,
) -> pd.DataFrame:
    """Load player records containing K-Means cluster labels."""

    if not path.exists():
        raise FileNotFoundError(
            f"Clustered dataset not found at: {path}\n"
            "Run src/train_kmeans.py first."
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(
            "The clustered player dataset contains no rows."
        )

    required_columns = [
        "Player",
        "Squad",
        "Position_Group",
        "Cluster",
        "Selected_K",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The clustered dataset is missing required columns: "
            f"{missing_columns}"
        )

    return data


def load_selected_features(
    path: Path = SELECTED_FEATURES_PATH,
) -> list[str]:
    """Load the football statistics selected for clustering."""

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


def create_cluster_profiles(
    clustered_data: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    """
    Compare each cluster with the overall median for its position.

    Positive differences show statistics that are stronger than the
    typical player in the same broad position group.
    """

    profile_rows: list[dict[str, object]] = []

    for position in sorted(
        clustered_data["Position_Group"].unique()
    ):
        position_data = clustered_data[
            clustered_data["Position_Group"] == position
        ].copy()

        available_features = [
            feature
            for feature in selected_features
            if feature in position_data.columns
        ]

        position_median = position_data[
            available_features
        ].median()

        for cluster_number in sorted(
            position_data["Cluster"].unique()
        ):
            cluster_data = position_data[
                position_data["Cluster"] == cluster_number
            ].copy()

            cluster_median = cluster_data[
                available_features
            ].median()

            differences = (
                cluster_median - position_median
            )

            strongest_features = differences.sort_values(
                ascending=False
            ).head(4)

            weakest_features = differences.sort_values(
                ascending=True
            ).head(3)

            profile_rows.append(
                {
                    "Position_Group": position,
                    "Cluster": int(cluster_number),
                    "Player_Count": int(len(cluster_data)),
                    "Strongest_Feature_1": (
                        strongest_features.index[0]
                    ),
                    "Strongest_Difference_1": float(
                        strongest_features.iloc[0]
                    ),
                    "Strongest_Feature_2": (
                        strongest_features.index[1]
                    ),
                    "Strongest_Difference_2": float(
                        strongest_features.iloc[1]
                    ),
                    "Strongest_Feature_3": (
                        strongest_features.index[2]
                    ),
                    "Strongest_Difference_3": float(
                        strongest_features.iloc[2]
                    ),
                    "Strongest_Feature_4": (
                        strongest_features.index[3]
                    ),
                    "Strongest_Difference_4": float(
                        strongest_features.iloc[3]
                    ),
                    "Weakest_Feature_1": (
                        weakest_features.index[0]
                    ),
                    "Weakest_Difference_1": float(
                        weakest_features.iloc[0]
                    ),
                    "Weakest_Feature_2": (
                        weakest_features.index[1]
                    ),
                    "Weakest_Difference_2": float(
                        weakest_features.iloc[1]
                    ),
                    "Weakest_Feature_3": (
                        weakest_features.index[2]
                    ),
                    "Weakest_Difference_3": float(
                        weakest_features.iloc[2]
                    ),
                }
            )

    return pd.DataFrame(profile_rows)


def assign_role_name(
    position: str,
    strongest_features: set[str],
) -> str:
    """Assign a cautious football interpretation to each cluster."""

    if position == "Defender":
        defensive_features = {
            "TklW_per90",
            "Int_per90",
            "Fls_per90",
        }

        attacking_features = {
            "Ast_per90",
            "Crs_per90",
            "Fld_per90",
            "Sh_per90",
        }

        if len(
            strongest_features.intersection(
                defensive_features
            )
        ) >= 2:
            return "Defensive-First Defender"

        if len(
            strongest_features.intersection(
                attacking_features
            )
        ) >= 2:
            return "Attacking / High-Involvement Defender"

        return "Balanced Defender"

    if position == "Midfielder":
        defensive_features = {
            "TklW_per90",
            "Int_per90",
            "Fls_per90",
        }

        creative_features = {
            "Ast_per90",
            "Crs_per90",
            "Fld_per90",
        }

        attacking_features = {
            "Gls_per90",
            "G-PK_per90",
            "Sh_per90",
            "SoT_per90",
        }

        if len(
            strongest_features.intersection(
                defensive_features
            )
        ) >= 2:
            return "Ball-Winning Midfielder"

        if len(
            strongest_features.intersection(
                creative_features
            )
        ) >= 2:
            return "Creative Midfielder"

        if len(
            strongest_features.intersection(
                attacking_features
            )
        ) >= 2:
            return "Attack-Minded Midfielder"

        return "Balanced / Box-to-Box Midfielder"

    if position == "Forward":
        goal_features = {
            "Gls_per90",
            "G-PK_per90",
            "Sh_per90",
            "SoT_per90",
            "Off_per90",
        }

        creative_features = {
            "Ast_per90",
            "Crs_per90",
            "Fld_per90",
        }

        defensive_features = {
            "TklW_per90",
            "Int_per90",
            "Fls_per90",
        }

        if len(
            strongest_features.intersection(
                goal_features
            )
        ) >= 2:
            return "Central Goal-Threat Forward"

        if len(
            strongest_features.intersection(
                creative_features
            )
        ) >= 2:
            return "Creative / Wide Forward"

        if len(
            strongest_features.intersection(
                defensive_features
            )
        ) >= 2:
            return "Pressing Forward"

        return "Balanced Forward"

    return "Unclassified Profile"


def add_role_names(
    profile_data: pd.DataFrame,
) -> pd.DataFrame:
    """Add football role labels based on cluster-defining features."""

    profiles = profile_data.copy()

    role_names: list[str] = []

    for _, row in profiles.iterrows():
        strongest_features = {
            row["Strongest_Feature_1"],
            row["Strongest_Feature_2"],
            row["Strongest_Feature_3"],
            row["Strongest_Feature_4"],
        }

        role_names.append(
            assign_role_name(
                row["Position_Group"],
                strongest_features,
            )
        )

    profiles["Final_Role_Name"] = role_names

    return profiles


def identify_representative_players(
    clustered_data: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    """
    Identify players closest to the centre of their cluster.

    These players represent the typical statistical profile of the
    cluster rather than simply being the highest-scoring players.
    """

    representative_rows: list[dict[str, object]] = []

    standardised_columns = [
        f"{feature}_z"
        for feature in selected_features
        if f"{feature}_z" in clustered_data.columns
    ]

    for position in sorted(
        clustered_data["Position_Group"].unique()
    ):
        position_data = clustered_data[
            clustered_data["Position_Group"] == position
        ].copy()

        for cluster_number in sorted(
            position_data["Cluster"].unique()
        ):
            cluster_data = position_data[
                position_data["Cluster"] == cluster_number
            ].copy()

            cluster_centre = cluster_data[
                standardised_columns
            ].mean().to_numpy()

            feature_matrix = cluster_data[
                standardised_columns
            ].to_numpy()

            distances = np.linalg.norm(
                feature_matrix - cluster_centre,
                axis=1,
            )

            cluster_data[
                "Distance_To_Cluster_Centre"
            ] = distances

            representatives = cluster_data.nsmallest(
                8,
                "Distance_To_Cluster_Centre",
            )

            for rank, (_, player) in enumerate(
                representatives.iterrows(),
                start=1,
            ):
                representative_rows.append(
                    {
                        "Position_Group": position,
                        "Cluster": int(cluster_number),
                        "Representative_Rank": rank,
                        "Player": player["Player"],
                        "Squad": player["Squad"],
                        "Competition": player.get(
                            "Comp",
                            pd.NA,
                        ),
                        "Minutes": player.get(
                            "Min",
                            pd.NA,
                        ),
                        "Distance_To_Cluster_Centre": float(
                            player[
                                "Distance_To_Cluster_Centre"
                            ]
                        ),
                    }
                )

    return pd.DataFrame(representative_rows)


def test_cluster_stability(
    clustered_data: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    """Test whether different random seeds produce similar clusters."""

    stability_rows: list[dict[str, float | int | str]] = []

    standardised_columns = [
        f"{feature}_z"
        for feature in selected_features
        if f"{feature}_z" in clustered_data.columns
    ]

    for position in sorted(
        clustered_data["Position_Group"].unique()
    ):
        position_data = clustered_data[
            clustered_data["Position_Group"] == position
        ].copy()

        selected_k = int(
            position_data["Selected_K"].iloc[0]
        )

        feature_matrix = position_data[
            standardised_columns
        ]

        reference_model = KMeans(
            n_clusters=selected_k,
            random_state=RANDOM_STATE,
            n_init=100,
        )

        reference_labels = reference_model.fit_predict(
            feature_matrix
        )

        for test_seed in TEST_SEEDS:
            test_model = KMeans(
                n_clusters=selected_k,
                random_state=test_seed,
                n_init=50,
            )

            test_labels = test_model.fit_predict(
                feature_matrix
            )

            stability_rows.append(
                {
                    "Position_Group": position,
                    "Selected_K": selected_k,
                    "Reference_Seed": RANDOM_STATE,
                    "Test_Seed": test_seed,
                    "Adjusted_Rand_Index": float(
                        adjusted_rand_score(
                            reference_labels,
                            test_labels,
                        )
                    ),
                }
            )

    return pd.DataFrame(stability_rows)


def attach_role_names_to_players(
    clustered_data: pd.DataFrame,
    profile_data: pd.DataFrame,
) -> pd.DataFrame:
    """Attach validated football role names to all player records."""

    role_lookup = profile_data[
        [
            "Position_Group",
            "Cluster",
            "Final_Role_Name",
        ]
    ].copy()

    validated_players = clustered_data.merge(
        role_lookup,
        on=[
            "Position_Group",
            "Cluster",
        ],
        how="left",
    )

    return validated_players


def save_validation_outputs(
    validated_players: pd.DataFrame,
    profile_data: pd.DataFrame,
    representative_players: pd.DataFrame,
    stability_data: pd.DataFrame,
) -> None:
    """Save all cluster-validation outputs."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    validated_players.to_csv(
        OUTPUT_DIR / "validated_clustered_players.csv",
        index=False,
    )

    profile_data.to_csv(
        OUTPUT_DIR / "validated_cluster_profiles.csv",
        index=False,
    )

    representative_players.to_csv(
        OUTPUT_DIR / "representative_players_by_cluster.csv",
        index=False,
    )

    stability_data.to_csv(
        OUTPUT_DIR / "cluster_stability_results.csv",
        index=False,
    )


def run_cluster_validation_pipeline() -> pd.DataFrame:
    """Run cluster interpretation and stability validation."""

    print("Loading clustered player data...")

    clustered_data = load_clustered_data()
    selected_features = load_selected_features()

    print(
        f"Clustered player records: {len(clustered_data):,}"
    )

    profile_data = create_cluster_profiles(
        clustered_data,
        selected_features,
    )

    profile_data = add_role_names(
        profile_data
    )

    representative_players = (
        identify_representative_players(
            clustered_data,
            selected_features,
        )
    )

    stability_data = test_cluster_stability(
        clustered_data,
        selected_features,
    )

    validated_players = attach_role_names_to_players(
        clustered_data,
        profile_data,
    )

    save_validation_outputs(
        validated_players,
        profile_data,
        representative_players,
        stability_data,
    )

    print(
        "\nCluster validation completed successfully."
    )

    print(
        f"Validated player records: "
        f"{len(validated_players):,}"
    )

    print(
        f"Outputs saved to: {OUTPUT_DIR}"
    )

    print("\nValidated cluster profiles:")

    print(
        profile_data[
            [
                "Position_Group",
                "Cluster",
                "Player_Count",
                "Final_Role_Name",
            ]
        ].to_string(index=False)
    )

    print("\nStability summary:")

    stability_summary = (
        stability_data.groupby(
            "Position_Group"
        )["Adjusted_Rand_Index"]
        .agg(
            Mean_ARI="mean",
            Minimum_ARI="min",
            Maximum_ARI="max",
        )
        .round(3)
    )

    print(stability_summary)

    return validated_players


if __name__ == "__main__":
    run_cluster_validation_pipeline()