from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler


PROJECT_ROOT = Path(__file__).resolve().parent.parent

INPUT_PATH = (
    PROJECT_ROOT
    / "data"
    / "processed"
    / "outfield_players_eligible_900.csv"
)

OUTPUT_DIR = PROJECT_ROOT / "data" / "model_ready"
MODEL_DIR = PROJECT_ROOT / "models"

CANDIDATE_FEATURES = [
    "Gls_per90",
    "Ast_per90",
    "G-PK_per90",
    "Sh_per90",
    "SoT_per90",
    "Crs_per90",
    "Fld_per90",
    "TklW_per90",
    "Int_per90",
    "Fls_per90",
    "Off_per90",
    "CrdY_per90",
]

POSITION_GROUPS = [
    "Defender",
    "Midfielder",
    "Forward",
]


def load_cleaned_outfield_data(
    path: Path = INPUT_PATH,
) -> pd.DataFrame:
    """Load the cleaned and eligible outfield-player dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Input file not found at: {path}\n"
            "Run src/clean_data.py first."
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError(
            "The cleaned outfield dataset contains no rows."
        )

    return data


def select_available_features(
    data: pd.DataFrame,
) -> list[str]:
    """Keep only candidate features that exist in the dataset."""

    available_features = [
        feature
        for feature in CANDIDATE_FEATURES
        if feature in data.columns
    ]

    missing_features = [
        feature
        for feature in CANDIDATE_FEATURES
        if feature not in data.columns
    ]

    if missing_features:
        print(
            "Warning: these candidate features were unavailable "
            f"and were skipped:\n{missing_features}"
        )

    if not available_features:
        raise ValueError(
            "None of the candidate modelling features were found."
        )

    return available_features


def create_feature_quality_audit(
    data: pd.DataFrame,
    features: list[str],
) -> pd.DataFrame:
    """Evaluate completeness, variation and distribution quality."""

    rows: list[dict[str, float | int | str]] = []

    for feature in features:
        values = pd.to_numeric(
            data[feature],
            errors="coerce",
        )

        rows.append(
            {
                "Feature": feature,
                "Missing_Count": int(values.isna().sum()),
                "Missing_Percentage": round(
                    float(values.isna().mean() * 100),
                    2,
                ),
                "Unique_Values": int(
                    values.nunique(dropna=True)
                ),
                "Mean": float(values.mean()),
                "Median": float(values.median()),
                "Standard_Deviation": float(values.std()),
                "Skewness": float(values.skew()),
                "Minimum": float(values.min()),
                "Maximum": float(values.max()),
            }
        )

    return pd.DataFrame(rows)


def choose_final_features(
    quality_audit: pd.DataFrame,
) -> list[str]:
    """Select complete, varying and interpretable model features."""

    selected_features = quality_audit[
        (quality_audit["Missing_Percentage"] <= 20)
        & (quality_audit["Unique_Values"] > 10)
        & (quality_audit["Standard_Deviation"] > 0)
    ]["Feature"].tolist()

    if not selected_features:
        raise ValueError(
            "No candidate features passed the quality rules."
        )

    return selected_features


def standardise_within_positions(
    data: pd.DataFrame,
    features: list[str],
) -> tuple[
    pd.DataFrame,
    dict[str, StandardScaler],
    dict[str, dict[str, float]],
]:
    """
    Impute and standardise features separately by position group.

    Missing values are replaced with the median for the player's
    own position group. StandardScaler is then fitted separately
    for defenders, midfielders and forwards.
    """

    prepared_groups: list[pd.DataFrame] = []
    scalers: dict[str, StandardScaler] = {}
    medians_by_position: dict[str, dict[str, float]] = {}

    for position in POSITION_GROUPS:
        group = data[
            data["Position_Group"] == position
        ].copy()

        if group.empty:
            print(
                f"Warning: no players found for {position}."
            )
            continue

        feature_data = group[features].apply(
            pd.to_numeric,
            errors="coerce",
        )

        position_medians = feature_data.median()

        if position_medians.isna().any():
            missing_median_features = position_medians[
                position_medians.isna()
            ].index.tolist()

            raise ValueError(
                f"Cannot impute {position}; these features "
                f"have no valid values: {missing_median_features}"
            )

        imputed_data = feature_data.fillna(
            position_medians
        )

        scaler = StandardScaler()

        standardised_values = scaler.fit_transform(
            imputed_data
        )

        standardised_columns = [
            f"{feature}_z"
            for feature in features
        ]

        standardised_frame = pd.DataFrame(
            standardised_values,
            columns=standardised_columns,
            index=group.index,
        )

        group = pd.concat(
            [group, standardised_frame],
            axis=1,
        )

        prepared_groups.append(group)

        scalers[position] = scaler

        medians_by_position[position] = (
            position_medians.to_dict()
        )

    if not prepared_groups:
        raise ValueError(
            "No position groups were available for standardisation."
        )

    prepared_data = pd.concat(
        prepared_groups,
        axis=0,
    ).sort_index()

    prepared_data.replace(
        [np.inf, -np.inf],
        np.nan,
        inplace=True,
    )

    return (
        prepared_data.reset_index(drop=True),
        scalers,
        medians_by_position,
    )


def save_feature_outputs(
    prepared_data: pd.DataFrame,
    quality_audit: pd.DataFrame,
    selected_features: list[str],
    scalers: dict[str, StandardScaler],
    medians_by_position: dict[str, dict[str, float]],
) -> None:
    """Save datasets, feature metadata and fitted scalers."""

    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    MODEL_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    prepared_data.to_csv(
        OUTPUT_DIR / "position_standardised_players.csv",
        index=False,
    )

    quality_audit.to_csv(
        OUTPUT_DIR / "feature_quality_audit.csv",
        index=False,
    )

    selected_features_frame = pd.DataFrame(
        {
            "Selected_Feature": selected_features
        }
    )

    selected_features_frame.to_csv(
        OUTPUT_DIR / "selected_features.csv",
        index=False,
    )

    median_rows: list[dict[str, str | float]] = []

    for position, medians in medians_by_position.items():
        for feature, median_value in medians.items():
            median_rows.append(
                {
                    "Position_Group": position,
                    "Feature": feature,
                    "Imputation_Median": median_value,
                }
            )

    pd.DataFrame(
        median_rows
    ).to_csv(
        OUTPUT_DIR / "position_imputation_values.csv",
        index=False,
    )

    for position, scaler in scalers.items():
        safe_position = position.lower()

        joblib.dump(
            scaler,
            MODEL_DIR / f"scaler_{safe_position}.joblib",
        )


def run_feature_engineering_pipeline() -> pd.DataFrame:
    """Run the complete position-based feature pipeline."""

    print("Loading cleaned outfield dataset...")

    data = load_cleaned_outfield_data()

    print(
        f"Input dataset shape: {data.shape}"
    )

    available_features = select_available_features(
        data
    )

    quality_audit = create_feature_quality_audit(
        data,
        available_features,
    )

    selected_features = choose_final_features(
        quality_audit
    )

    print("\nSelected modelling features:")

    for feature in selected_features:
        print(f"- {feature}")

    (
        prepared_data,
        scalers,
        medians_by_position,
    ) = standardise_within_positions(
        data,
        selected_features,
    )

    save_feature_outputs(
        prepared_data,
        quality_audit,
        selected_features,
        scalers,
        medians_by_position,
    )

    print(
        "\nFeature-engineering pipeline completed successfully."
    )

    print(
        f"Model-ready player records: {len(prepared_data):,}"
    )

    print(
        f"Outputs saved to: {OUTPUT_DIR}"
    )

    print(
        f"Scalers saved to: {MODEL_DIR}"
    )

    print("\nPlayers by position:")

    print(
        prepared_data[
            "Position_Group"
        ].value_counts()
    )

    return prepared_data


if __name__ == "__main__":
    run_feature_engineering_pipeline()