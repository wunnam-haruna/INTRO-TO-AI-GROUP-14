from __future__ import annotations

from pathlib import Path

import joblib
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import (
    calinski_harabasz_score,
    davies_bouldin_score,
    silhouette_score,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_DATA_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "position_standardised_players.csv"
)

FEATURE_LIST_PATH = (
    PROJECT_ROOT
    / "data"
    / "model_ready"
    / "selected_features.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "clustered"
MODEL_DIR = PROJECT_ROOT / "models"

POSITION_GROUPS = [
    "Defender",
    "Midfielder",
    "Forward",
]

MINIMUM_K = 2
MAXIMUM_K = 8
RANDOM_STATE = 42


def load_model_ready_data(
    path: Path = INPUT_DATA_PATH,
) -> pd.DataFrame:
    """Load the position-standardised outfield-player dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Model-ready dataset not found at: {path}\n"
            "Run src/feature_engineering.py first."
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(
            "The model-ready dataset contains no rows."
        )

    required_columns = [
        "Player",
        "Squad",
        "Position_Group",
    ]

    missing_columns = [
        column
        for column in required_columns
        if column not in data.columns
    ]

    if missing_columns:
        raise ValueError(
            "The model-ready dataset is missing required columns: "
            f"{missing_columns}"
        )

    return data


def load_selected_features(
    path: Path = FEATURE_LIST_PATH,
) -> list[str]:
    """Load the features selected during feature engineering."""

    if not path.exists():
        raise FileNotFoundError(
            f"Selected-feature file not found at: {path}\n"
            "Run src/feature_engineering.py first."
        )

    feature_data = pd.read_csv(path)

    if "Selected_Feature" not in feature_data.columns:
        raise ValueError(
            "selected_features.csv must contain a "
            "'Selected_Feature' column."
        )

    selected_features = (
        feature_data["Selected_Feature"]
        .dropna()
        .astype(str)
        .tolist()
    )

    if not selected_features:
        raise ValueError(
            "No selected modelling features were found."
        )

    return selected_features


def get_standardised_feature_columns(
    data: pd.DataFrame,
    selected_features: list[str],
) -> list[str]:
    """Find the z-score columns used as K-Means inputs."""

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
            "The following standardised feature columns "
            f"are missing: {missing_columns}"
        )

    return standardised_columns


def evaluate_candidate_clusters(
    position_data: pd.DataFrame,
    standardised_columns: list[str],
    position_name: str,
) -> pd.DataFrame:
    """Test candidate values of K for one position group."""

    feature_matrix = position_data[
        standardised_columns
    ].copy()

    if feature_matrix.isna().any().any():
        raise ValueError(
            f"Missing standardised values detected for "
            f"{position_name}."
        )

    maximum_allowed_k = min(
        MAXIMUM_K,
        len(position_data) - 1,
    )

    if maximum_allowed_k < MINIMUM_K:
        raise ValueError(
            f"Not enough {position_name} records for clustering."
        )

    evaluation_rows: list[dict[str, float | int | str]] = []

    for number_of_clusters in range(
        MINIMUM_K,
        maximum_allowed_k + 1,
    ):
        model = KMeans(
            n_clusters=number_of_clusters,
            random_state=RANDOM_STATE,
            n_init=50,
        )

        cluster_labels = model.fit_predict(
            feature_matrix
        )

        cluster_sizes = pd.Series(
            cluster_labels
        ).value_counts()

        evaluation_rows.append(
            {
                "Position_Group": position_name,
                "K": number_of_clusters,
                "Inertia": float(model.inertia_),
                "Silhouette_Score": float(
                    silhouette_score(
                        feature_matrix,
                        cluster_labels,
                    )
                ),
                "Calinski_Harabasz_Score": float(
                    calinski_harabasz_score(
                        feature_matrix,
                        cluster_labels,
                    )
                ),
                "Davies_Bouldin_Score": float(
                    davies_bouldin_score(
                        feature_matrix,
                        cluster_labels,
                    )
                ),
                "Smallest_Cluster": int(
                    cluster_sizes.min()
                ),
                "Largest_Cluster": int(
                    cluster_sizes.max()
                ),
                "Cluster_Balance_Ratio": float(
                    cluster_sizes.min()
                    / cluster_sizes.max()
                ),
            }
        )

    return pd.DataFrame(evaluation_rows)


def choose_best_k(
    evaluation_data: pd.DataFrame,
    player_count: int,
) -> int:
    """
    Select K using multiple evaluation measures.

    Silhouette score is given the highest importance.
    Calinski-Harabasz and Davies-Bouldin scores are
    also considered. Extremely small clusters are
    rejected where possible.
    """

    minimum_cluster_size = max(
        10,
        int(player_count * 0.03),
    )

    acceptable_results = evaluation_data[
        evaluation_data["Smallest_Cluster"]
        >= minimum_cluster_size
    ].copy()

    if acceptable_results.empty:
        acceptable_results = evaluation_data.copy()

    acceptable_results["Silhouette_Rank"] = (
        acceptable_results[
            "Silhouette_Score"
        ].rank(
            ascending=False,
            method="min",
        )
    )

    acceptable_results["Calinski_Rank"] = (
        acceptable_results[
            "Calinski_Harabasz_Score"
        ].rank(
            ascending=False,
            method="min",
        )
    )

    acceptable_results["Davies_Rank"] = (
        acceptable_results[
            "Davies_Bouldin_Score"
        ].rank(
            ascending=True,
            method="min",
        )
    )

    acceptable_results["Composite_Rank"] = (
        acceptable_results["Silhouette_Rank"] * 0.50
        + acceptable_results["Calinski_Rank"] * 0.30
        + acceptable_results["Davies_Rank"] * 0.20
    )

    best_result = acceptable_results.sort_values(
        by=[
            "Composite_Rank",
            "Silhouette_Score",
        ],
        ascending=[
            True,
            False,
        ],
    ).iloc[0]

    return int(best_result["K"])


def fit_final_model(
    position_data: pd.DataFrame,
    standardised_columns: list[str],
    number_of_clusters: int,
) -> tuple[KMeans, pd.DataFrame]:
    """Fit the final K-Means model for one position."""

    feature_matrix = position_data[
        standardised_columns
    ]

    model = KMeans(
        n_clusters=number_of_clusters,
        random_state=RANDOM_STATE,
        n_init=100,
    )

    cluster_labels = model.fit_predict(
        feature_matrix
    )

    clustered_data = position_data.copy()

    clustered_data["Cluster"] = cluster_labels
    clustered_data["Selected_K"] = (
        number_of_clusters
    )

    return model, clustered_data


def create_cluster_centroids(
    clustered_data: pd.DataFrame,
    selected_features: list[str],
) -> pd.DataFrame:
    """Create interpretable median profiles for each cluster."""

    available_features = [
        feature
        for feature in selected_features
        if feature in clustered_data.columns
    ]

    centroid_data = (
        clustered_data.groupby(
            [
                "Position_Group",
                "Cluster",
            ]
        )[available_features]
        .median()
        .reset_index()
    )

    cluster_sizes = (
        clustered_data.groupby(
            [
                "Position_Group",
                "Cluster",
            ]
        )
        .size()
        .reset_index(
            name="Player_Count"
        )
    )

    centroid_data = centroid_data.merge(
        cluster_sizes,
        on=[
            "Position_Group",
            "Cluster",
        ],
        how="left",
    )

    ordered_columns = [
        "Position_Group",
        "Cluster",
        "Player_Count",
    ] + available_features

    return centroid_data[ordered_columns]


def save_kmeans_outputs(
    clustered_data: pd.DataFrame,
    evaluation_data: pd.DataFrame,
    centroid_data: pd.DataFrame,
    models: dict[str, KMeans],
) -> None:
    """Save cluster results, evaluation metrics and models."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    clustered_data.to_csv(
        OUTPUT_DIR / "clustered_outfield_players.csv",
        index=False,
    )

    evaluation_data.to_csv(
        OUTPUT_DIR / "kmeans_evaluation_results.csv",
        index=False,
    )

    centroid_data.to_csv(
        OUTPUT_DIR / "cluster_centroid_profiles.csv",
        index=False,
    )

    for position, model in models.items():
        safe_position = position.lower()

        joblib.dump(
            model,
            MODEL_DIR / f"kmeans_{safe_position}.joblib",
        )


def run_kmeans_training_pipeline() -> pd.DataFrame:
    """Run K-Means evaluation and training for all positions."""

    print("Loading model-ready player data...")

    player_data = load_model_ready_data()

    selected_features = load_selected_features()

    standardised_columns = (
        get_standardised_feature_columns(
            player_data,
            selected_features,
        )
    )

    all_evaluations: list[pd.DataFrame] = []
    all_clustered_groups: list[pd.DataFrame] = []
    trained_models: dict[str, KMeans] = {}

    print(
        f"Total players available: {len(player_data):,}"
    )

    for position in POSITION_GROUPS:
        print(
            f"\nEvaluating clusters for {position}s..."
        )

        position_data = player_data[
            player_data["Position_Group"]
            == position
        ].copy()

        if position_data.empty:
            print(
                f"No records available for {position}."
            )
            continue

        print(
            f"Player records: {len(position_data):,}"
        )

        evaluation_data = (
            evaluate_candidate_clusters(
                position_data,
                standardised_columns,
                position,
            )
        )

        best_k = choose_best_k(
            evaluation_data,
            len(position_data),
        )

        print(
            f"Selected number of clusters: K = {best_k}"
        )

        final_model, clustered_group = (
            fit_final_model(
                position_data,
                standardised_columns,
                best_k,
            )
        )

        trained_models[position] = final_model
        all_evaluations.append(evaluation_data)
        all_clustered_groups.append(
            clustered_group
        )

        cluster_counts = (
            clustered_group["Cluster"]
            .value_counts()
            .sort_index()
        )

        print("Cluster sizes:")

        for cluster, player_count in (
            cluster_counts.items()
        ):
            print(
                f"- Cluster {cluster}: "
                f"{player_count} players"
            )

    if not all_clustered_groups:
        raise ValueError(
            "No position groups were successfully clustered."
        )

    combined_clustered_data = pd.concat(
        all_clustered_groups,
        ignore_index=True,
    )

    combined_evaluation_data = pd.concat(
        all_evaluations,
        ignore_index=True,
    )

    centroid_data = create_cluster_centroids(
        combined_clustered_data,
        selected_features,
    )

    save_kmeans_outputs(
        combined_clustered_data,
        combined_evaluation_data,
        centroid_data,
        trained_models,
    )

    print(
        "\nK-Means training pipeline completed successfully."
    )

    print(
        f"Clustered player records: "
        f"{len(combined_clustered_data):,}"
    )

    print(
        f"Outputs saved to: {OUTPUT_DIR}"
    )

    print(
        f"Models saved to: {MODEL_DIR}"
    )

    print("\nSelected K by position:")

    selected_k_summary = (
        combined_clustered_data.groupby(
            "Position_Group"
        )["Selected_K"]
        .first()
    )

    print(selected_k_summary)

    return combined_clustered_data


if __name__ == "__main__":
    run_kmeans_training_pipeline()