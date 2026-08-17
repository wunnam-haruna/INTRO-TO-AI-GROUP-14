from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DATA_PATH = PROJECT_ROOT / "players_data-2025_2026.csv"
PROCESSED_DIR = PROJECT_ROOT / "data" / "processed"

MINIMUM_MINUTES = 900

POSITION_MAP = {
    "GK": "Goalkeeper",
    "DF": "Defender",
    "MF": "Midfielder",
    "FW": "Forward",
}

TEXT_COLUMNS = [
    "Player",
    "Nation",
    "Pos",
    "Squad",
    "Comp",
]

IDENTIFICATION_COLUMNS = [
    "Player",
    "Nation",
    "Pos",
    "Squad",
    "Comp",
    "Age",
    "Born",
]

PLAYING_TIME_COLUMNS = [
    "MP",
    "Starts",
    "Min",
    "90s",
]

OUTFIELD_TOTAL_COLUMNS = [
    "Gls",
    "Ast",
    "G+A",
    "G-PK",
    "PK",
    "PKatt",
    "CrdY",
    "CrdR",
    "Sh",
    "SoT",
    "Fls",
    "Fld",
    "Off",
    "Crs",
    "Int",
    "TklW",
    "OG",
]

OUTFIELD_EFFICIENCY_COLUMNS = [
    "SoT%",
    "G/Sh",
    "G/SoT",
    "PPM",
    "+/-90",
    "On-Off",
]

GOALKEEPER_COLUMNS = [
    "GA",
    "GA90",
    "SoTA",
    "Saves",
    "Save%",
    "W",
    "D",
    "L",
    "CS",
    "CS%",
    "PKatt_stats_keeper",
    "PKA",
    "PKsv",
    "PKm",
]


def load_raw_data(path: Path = RAW_DATA_PATH) -> pd.DataFrame:
    """Load the original football-player dataset."""

    if not path.exists():
        raise FileNotFoundError(
            f"Dataset not found at: {path}\n"
            "Confirm that players_data-2025_2026.csv is in the project root."
        )

    data = pd.read_csv(path)

    if data.empty:
        raise ValueError("The dataset was loaded, but it contains no rows.")

    return data


def select_relevant_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Retain useful identification, playing-time and performance columns."""

    requested_columns = (
        IDENTIFICATION_COLUMNS
        + PLAYING_TIME_COLUMNS
        + OUTFIELD_TOTAL_COLUMNS
        + OUTFIELD_EFFICIENCY_COLUMNS
        + GOALKEEPER_COLUMNS
    )

    available_columns = [
        column for column in requested_columns if column in data.columns
    ]

    missing_columns = [
        column for column in requested_columns if column not in data.columns
    ]

    if missing_columns:
        print(
            "Warning: the following requested columns were unavailable "
            f"and were skipped:\n{missing_columns}"
        )

    return data[available_columns].copy()


def clean_text_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Trim unnecessary whitespace from textual fields."""

    cleaned = data.copy()

    for column in TEXT_COLUMNS:
        if column in cleaned.columns:
            cleaned[column] = (
                cleaned[column].astype("string").str.strip().replace("", pd.NA)
            )

    return cleaned


def convert_numeric_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Ensure non-text columns use numeric data types where possible."""

    cleaned = data.copy()

    non_numeric_columns = set(TEXT_COLUMNS)

    for column in cleaned.columns:
        if column not in non_numeric_columns:
            cleaned[column] = pd.to_numeric(
                cleaned[column],
                errors="coerce",
            )

    return cleaned


def add_position_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Split raw positions into primary, secondary and broad position groups."""

    cleaned = data.copy()

    position_parts = cleaned["Pos"].str.split(",", n=1, expand=True)

    cleaned["Primary_Position"] = position_parts[0].str.strip()

    if position_parts.shape[1] > 1:
        cleaned["Secondary_Position"] = position_parts[1].str.strip()
    else:
        cleaned["Secondary_Position"] = pd.NA

    cleaned["Is_Multi_Position"] = cleaned["Secondary_Position"].notna()

    cleaned["Position_Group"] = cleaned["Primary_Position"].map(POSITION_MAP)

    return cleaned


def add_eligibility_columns(data: pd.DataFrame) -> pd.DataFrame:
    """Add flags for different playing-time thresholds."""

    cleaned = data.copy()

    cleaned["Eligible_450"] = cleaned["Min"].ge(450)
    cleaned["Eligible_900"] = cleaned["Min"].ge(900)
    cleaned["Eligible_1350"] = cleaned["Min"].ge(1350)

    return cleaned


def calculate_per_90_statistics(data: pd.DataFrame) -> pd.DataFrame:
    """Convert selected raw totals into per-90-minute rates."""

    cleaned = data.copy()

    for column in OUTFIELD_TOTAL_COLUMNS:
        if column not in cleaned.columns:
            continue

        new_column = f"{column}_per90"

        cleaned[new_column] = np.where(
            cleaned["Min"].gt(0),
            cleaned[column] / cleaned["Min"] * 90,
            np.nan,
        )

    cleaned.replace([np.inf, -np.inf], np.nan, inplace=True)

    return cleaned


def remove_invalid_records(data: pd.DataFrame) -> pd.DataFrame:
    """Remove records that cannot support valid player analysis."""

    cleaned = data.copy()

    required_columns = [
        "Player",
        "Pos",
        "Squad",
        "Comp",
        "Min",
        "Position_Group",
    ]

    cleaned = cleaned.dropna(subset=required_columns)

    cleaned = cleaned[cleaned["Min"] >= 0]

    cleaned = cleaned.drop_duplicates(
        subset=["Player", "Squad", "Comp"],
        keep="first",
    )

    return cleaned.reset_index(drop=True)


def create_modelling_populations(
    data: pd.DataFrame,
    minimum_minutes: int = MINIMUM_MINUTES,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Separate eligible outfield players and goalkeepers."""

    eligible = data[data["Min"] >= minimum_minutes].copy()

    outfield = eligible[
        eligible["Position_Group"].isin(["Defender", "Midfielder", "Forward"])
    ].copy()

    goalkeepers = eligible[eligible["Position_Group"] == "Goalkeeper"].copy()

    return (
        outfield.reset_index(drop=True),
        goalkeepers.reset_index(drop=True),
    )


def generate_audit_summary(
    raw_data: pd.DataFrame,
    cleaned_data: pd.DataFrame,
    outfield_data: pd.DataFrame,
    goalkeeper_data: pd.DataFrame,
) -> pd.DataFrame:
    """Create a compact audit summary for reporting."""

    summary = {
        "Raw rows": len(raw_data),
        "Raw columns": raw_data.shape[1],
        "Cleaned rows": len(cleaned_data),
        "Cleaned columns": cleaned_data.shape[1],
        "Exact duplicate raw rows": raw_data.duplicated().sum(),
        "Multi-position players": cleaned_data["Is_Multi_Position"].sum(),
        "Players with at least 450 minutes": cleaned_data["Eligible_450"].sum(),
        "Players with at least 900 minutes": cleaned_data["Eligible_900"].sum(),
        "Players with at least 1350 minutes": cleaned_data["Eligible_1350"].sum(),
        "Eligible outfield players": len(outfield_data),
        "Eligible goalkeepers": len(goalkeeper_data),
    }

    return pd.DataFrame(
        summary.items(),
        columns=["Metric", "Value"],
    )


def save_outputs(
    cleaned_data: pd.DataFrame,
    outfield_data: pd.DataFrame,
    goalkeeper_data: pd.DataFrame,
    audit_summary: pd.DataFrame,
    output_directory: Path = PROCESSED_DIR,
) -> None:
    """Save cleaned datasets and the audit summary."""

    output_directory.mkdir(parents=True, exist_ok=True)

    cleaned_data.to_csv(
        output_directory / "players_cleaned_master.csv",
        index=False,
    )

    outfield_data.to_csv(
        output_directory / "outfield_players_eligible_900.csv",
        index=False,
    )

    goalkeeper_data.to_csv(
        output_directory / "goalkeepers_eligible_900.csv",
        index=False,
    )

    audit_summary.to_csv(
        output_directory / "data_audit_summary.csv",
        index=False,
    )


def run_cleaning_pipeline() -> tuple[
    pd.DataFrame,
    pd.DataFrame,
    pd.DataFrame,
]:
    """Run the complete football-player cleaning pipeline."""

    print("Loading raw dataset...")
    raw_data = load_raw_data()

    print(f"Raw dataset shape: {raw_data.shape}")

    cleaned_data = select_relevant_columns(raw_data)
    cleaned_data = clean_text_columns(cleaned_data)
    cleaned_data = convert_numeric_columns(cleaned_data)
    cleaned_data = add_position_columns(cleaned_data)
    cleaned_data = add_eligibility_columns(cleaned_data)
    cleaned_data = calculate_per_90_statistics(cleaned_data)
    cleaned_data = remove_invalid_records(cleaned_data)

    outfield_data, goalkeeper_data = create_modelling_populations(cleaned_data)

    audit_summary = generate_audit_summary(
        raw_data,
        cleaned_data,
        outfield_data,
        goalkeeper_data,
    )

    save_outputs(
        cleaned_data,
        outfield_data,
        goalkeeper_data,
        audit_summary,
    )

    print("\nCleaning pipeline completed successfully.")
    print(f"Cleaned master records: {len(cleaned_data):,}")
    print(f"Eligible outfield players: {len(outfield_data):,}")
    print(f"Eligible goalkeepers: {len(goalkeeper_data):,}")
    print(f"Outputs saved to: {PROCESSED_DIR}")

    print("\nAudit summary:")
    print(audit_summary.to_string(index=False))

    return cleaned_data, outfield_data, goalkeeper_data


if __name__ == "__main__":
    run_cleaning_pipeline()
