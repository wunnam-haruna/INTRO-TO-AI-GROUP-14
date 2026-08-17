from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import random
import uuid

import numpy as np
import pandas as pd

from src.player_pools import add_game_metadata, get_target_pool


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


FEATURE_LABELS: dict[str, str] = {
    "Gls_per90": "Goals per 90",
    "Ast_per90": "Assists per 90",
    "G-PK_per90": "Non-penalty goals per 90",
    "Sh_per90": "Shots per 90",
    "SoT_per90": "Shots on target per 90",
    "Crs_per90": "Crosses per 90",
    "Fld_per90": "Fouls drawn per 90",
    "TklW_per90": "Tackles won per 90",
    "Int_per90": "Interceptions per 90",
    "Fls_per90": "Fouls committed per 90",
    "Off_per90": "Offsides per 90",
    "CrdY_per90": "Yellow cards per 90",
}


DIFFICULTY_RULES: dict[str, dict[str, object]] = {
    "Rookie": {
        "same_cluster": False,
        "rank_windows": ((0, 5), (15, 40), (40, 80), (80, 150)),
        "minimum_featured_options": 2,
    },
    "Scout": {
        "same_cluster": False,
        "rank_windows": ((0, 4), (5, 18), (18, 40), (40, 80)),
        "minimum_featured_options": 2,
    },
    "Analyst": {
        "same_cluster": True,
        "rank_windows": ((0, 3), (3, 8), (8, 15), (15, 30)),
        "minimum_featured_options": 1,
    },
    "Elite": {
        "same_cluster": True,
        "rank_windows": ((0, 1), (1, 4), (4, 7), (7, 11)),
        "minimum_featured_options": 1,
    },
}


@lru_cache(maxsize=1)
def _cached_game_data() -> pd.DataFrame:
    if not VALIDATED_DATA_PATH.exists():
        raise FileNotFoundError(
            "validated_clustered_players.csv was not found. "
            "Run python src/main.py before starting the game."
        )

    data = pd.read_csv(VALIDATED_DATA_PATH)
    if data.empty:
        raise ValueError("The validated player dataset is empty.")

    prepared = add_game_metadata(data)
    prepared = prepared.drop_duplicates(subset=["Player_Key"]).reset_index(
        drop=True
    )
    return prepared


def load_game_data() -> pd.DataFrame:
    """Return a defensive copy of the validated game dataset."""

    return _cached_game_data().copy()


@lru_cache(maxsize=1)
def _cached_selected_features() -> tuple[str, ...]:
    if not SELECTED_FEATURES_PATH.exists():
        raise FileNotFoundError(
            "selected_features.csv was not found. "
            "Run python src/feature_engineering.py first."
        )

    feature_data = pd.read_csv(SELECTED_FEATURES_PATH)
    if "Selected_Feature" not in feature_data.columns:
        raise ValueError(
            "selected_features.csv must contain a Selected_Feature column."
        )

    features = tuple(
        feature_data["Selected_Feature"].dropna().astype(str).tolist()
    )
    if not features:
        raise ValueError("No selected features were found.")

    return features


def load_selected_features() -> list[str]:
    return list(_cached_selected_features())


def get_standardised_columns(
    data: pd.DataFrame,
    features: list[str],
) -> list[str]:
    columns = [
        f"{feature}_z"
        for feature in features
        if f"{feature}_z" in data.columns
    ]

    if not columns:
        raise ValueError("No standardised feature columns were found.")

    return columns


def similarity_score_from_distance(distance: float) -> float:
    """Convert Euclidean distance into a bounded similarity measure."""

    return 1 / (1 + max(float(distance), 0.0))


def calculate_player_similarities(
    data: pd.DataFrame,
    target_player: pd.Series,
    standardised_columns: list[str],
    *,
    same_cluster_only: bool,
) -> pd.DataFrame:
    """Rank comparable players by distance from the target profile."""

    candidates = data[
        data["Position_Group"] == target_player["Position_Group"]
    ].copy()

    if same_cluster_only:
        candidates = candidates[
            candidates["Cluster"] == target_player["Cluster"]
        ].copy()

    candidates = candidates[
        candidates["Player_Key"] != target_player["Player_Key"]
    ].copy()

    if candidates.empty:
        raise ValueError("No comparable players were available.")

    target_vector = (
        target_player[standardised_columns].astype(float).to_numpy()
    )
    candidate_matrix = (
        candidates[standardised_columns].astype(float).to_numpy()
    )

    distances = np.linalg.norm(candidate_matrix - target_vector, axis=1)
    candidates["Similarity_Distance"] = distances
    candidates["Similarity_Score"] = [
        similarity_score_from_distance(distance) for distance in distances
    ]
    candidates["Similarity_Rank"] = (
        candidates["Similarity_Score"]
        .rank(method="first", ascending=False)
        .astype(int)
    )

    return candidates.sort_values(
        ["Similarity_Score", "Player"], ascending=[False, True]
    ).reset_index(drop=True)


def _weighted_pick(
    pool: pd.DataFrame,
    *,
    rng: random.Random,
    prefer_featured: bool,
) -> pd.Series:
    if pool.empty:
        raise ValueError("Cannot select a player from an empty pool.")

    if not prefer_featured:
        return pool.iloc[rng.randrange(len(pool))]

    weights = np.ones(len(pool), dtype=float)
    weights += pool["Is_Featured_Club"].astype(float).to_numpy() * 2.5
    weights += pool["Is_Featured_Player"].astype(float).to_numpy() * 6.0
    probability = weights / weights.sum()

    selected_position = int(
        np.random.default_rng(rng.randrange(1, 2**32 - 1)).choice(
            len(pool), p=probability
        )
    )
    return pool.iloc[selected_position]


def _target_has_enough_candidates(
    data: pd.DataFrame,
    target: pd.Series,
    *,
    same_cluster_only: bool,
) -> bool:
    candidates = data[
        data["Position_Group"] == target["Position_Group"]
    ]
    if same_cluster_only:
        candidates = candidates[candidates["Cluster"] == target["Cluster"]]
    return len(candidates) >= 8


def choose_target_player(
    data: pd.DataFrame,
    *,
    game_mode: str,
    difficulty: str,
    excluded_target_keys: set[str] | None = None,
    rng: random.Random | None = None,
) -> pd.Series:
    """Choose a balanced, valid target player for the requested mode."""

    rng = rng or random.Random()
    excluded_target_keys = excluded_target_keys or set()
    rules = DIFFICULTY_RULES.get(difficulty, DIFFICULTY_RULES["Scout"])
    same_cluster_only = bool(rules["same_cluster"])

    pool = get_target_pool(data, game_mode).copy()
    unused_pool = pool[~pool["Player_Key"].isin(excluded_target_keys)].copy()
    if unused_pool.empty:
        unused_pool = pool

    valid_rows = [
        index
        for index, target in unused_pool.iterrows()
        if _target_has_enough_candidates(
            data, target, same_cluster_only=same_cluster_only
        )
    ]
    if not valid_rows:
        raise ValueError("No valid target players were available for this mode.")

    valid_pool = unused_pool.loc[valid_rows]
    positions = valid_pool["Position_Group"].dropna().unique().tolist()
    selected_position = rng.choice(positions)
    position_pool = valid_pool[
        valid_pool["Position_Group"] == selected_position
    ]

    return _weighted_pick(
        position_pool,
        rng=rng,
        prefer_featured=(game_mode == "Star Match"),
    )


def _pick_from_rank_window(
    ranked_players: pd.DataFrame,
    start: int,
    end: int,
    selected_keys: set[str],
    *,
    rng: random.Random,
    prefer_featured: bool,
) -> pd.Series | None:
    window = ranked_players.iloc[start:end].copy()
    window = window[~window["Player_Key"].isin(selected_keys)]
    if window.empty:
        return None
    return _weighted_pick(
        window,
        rng=rng,
        prefer_featured=prefer_featured,
    )


def _ensure_featured_options(
    selected: list[pd.Series],
    ranked_players: pd.DataFrame,
    minimum_featured: int,
) -> list[pd.Series]:
    if minimum_featured <= 0:
        return selected

    current_featured = sum(
        bool(player["Is_Featured_Player"]) for player in selected
    )
    if current_featured >= minimum_featured:
        return selected

    selected_keys = {str(player["Player_Key"]) for player in selected}
    featured_candidates = ranked_players[
        ranked_players["Is_Featured_Player"]
        & ~ranked_players["Player_Key"].isin(selected_keys)
    ].copy()

    while current_featured < minimum_featured and not featured_candidates.empty:
        replacement = featured_candidates.iloc[0]
        replaceable_indexes = [
            index
            for index, player in enumerate(selected)
            if not bool(player["Is_Featured_Player"])
        ]
        if not replaceable_indexes:
            break

        replace_index = min(
            replaceable_indexes,
            key=lambda index: float(selected[index]["Similarity_Score"]),
        )
        selected[replace_index] = replacement
        current_featured += 1
        featured_candidates = featured_candidates.iloc[1:]

    return selected


def select_question_candidates(
    ranked_players: pd.DataFrame,
    *,
    difficulty: str,
    game_mode: str,
    rng: random.Random | None = None,
) -> pd.DataFrame:
    """Select four credible options using difficulty-specific rank bands."""

    rng = rng or random.Random()
    if len(ranked_players) < 4:
        raise ValueError("Not enough comparable players to create a question.")

    rules = DIFFICULTY_RULES.get(difficulty, DIFFICULTY_RULES["Scout"])
    rank_windows = rules["rank_windows"]
    prefer_featured = game_mode == "Star Match"

    selected: list[pd.Series] = []
    selected_keys: set[str] = set()

    for start, end in rank_windows:
        player = _pick_from_rank_window(
            ranked_players,
            int(start),
            int(end),
            selected_keys,
            rng=rng,
            prefer_featured=prefer_featured,
        )
        if player is None:
            continue
        selected.append(player)
        selected_keys.add(str(player["Player_Key"]))

    if len(selected) < 4:
        remaining = ranked_players[
            ~ranked_players["Player_Key"].isin(selected_keys)
        ]
        for _, player in remaining.iterrows():
            selected.append(player)
            selected_keys.add(str(player["Player_Key"]))
            if len(selected) == 4:
                break

    if len(selected) < 4:
        raise ValueError("Unable to create four unique candidate options.")

    if prefer_featured:
        selected = _ensure_featured_options(
            selected,
            ranked_players,
            int(rules["minimum_featured_options"]),
        )

    options = pd.DataFrame(selected).drop_duplicates(subset=["Player_Key"])
    if len(options) < 4:
        remaining = ranked_players[
            ~ranked_players["Player_Key"].isin(options["Player_Key"])
        ]
        options = pd.concat(
            [options, remaining.head(4 - len(options))], ignore_index=True
        )

    options = options.head(4).copy()
    options = options.sample(
        frac=1,
        random_state=rng.randrange(1, 2**32 - 1),
    ).reset_index(drop=True)
    options["Candidate_Letter"] = ["A", "B", "C", "D"]

    return options


def create_question(
    difficulty: str = "Scout",
    game_mode: str = "Star Match",
    excluded_target_keys: set[str] | None = None,
    random_seed: int | None = None,
) -> dict[str, object]:
    """Generate one complete, model-backed StyleMatch question."""

    if difficulty not in DIFFICULTY_RULES:
        raise ValueError(f"Unknown difficulty: {difficulty}")

    rng = random.Random(random_seed)
    data = load_game_data()
    features = load_selected_features()
    standardised_columns = get_standardised_columns(data, features)
    target_player = choose_target_player(
        data,
        game_mode=game_mode,
        difficulty=difficulty,
        excluded_target_keys=excluded_target_keys,
        rng=rng,
    )

    same_cluster_only = bool(DIFFICULTY_RULES[difficulty]["same_cluster"])
    ranked_players = calculate_player_similarities(
        data,
        target_player,
        standardised_columns,
        same_cluster_only=same_cluster_only,
    )
    options = select_question_candidates(
        ranked_players,
        difficulty=difficulty,
        game_mode=game_mode,
        rng=rng,
    )
    correct_player = options.loc[options["Similarity_Score"].idxmax()]

    return {
        "question_id": uuid.uuid4().hex,
        "target_player": target_player,
        "options": options,
        "correct_player": correct_player,
        "difficulty": difficulty,
        "game_mode": game_mode,
        "selected_features": features,
    }


def calculate_round_points(
    chosen_similarity: float,
    best_similarity: float,
) -> int:
    """Award 0–100 base points relative to the best displayed option."""

    if best_similarity <= 0:
        return 0
    ratio = max(0.0, min(1.0, chosen_similarity / best_similarity))
    return int(round(ratio * 100))


def _pair_profile_analysis(
    target: pd.Series,
    candidate: pd.Series,
    selected_features: list[str],
) -> dict[str, list[str]]:
    comparisons: list[tuple[str, float]] = []

    for feature in selected_features:
        z_column = f"{feature}_z"
        if z_column not in target.index or z_column not in candidate.index:
            continue
        difference = abs(float(target[z_column]) - float(candidate[z_column]))
        comparisons.append((FEATURE_LABELS.get(feature, feature), difference))

    comparisons.sort(key=lambda item: item[1])
    aligned = [name for name, _ in comparisons[:3]]
    different = [name for name, _ in comparisons[-2:]][::-1]
    return {"aligned_features": aligned, "different_features": different}


def evaluate_choice(
    question: dict[str, object],
    chosen_player_key: str,
) -> dict[str, object]:
    """Evaluate a selected candidate and prepare an explainable result."""

    options = question["options"]
    if not isinstance(options, pd.DataFrame):
        raise TypeError("Question options are invalid.")

    selected_rows = options[options["Player_Key"] == chosen_player_key]
    if selected_rows.empty:
        raise ValueError("The selected candidate was not found.")

    selected_player = selected_rows.iloc[0]
    correct_player = question["correct_player"]
    target_player = question["target_player"]
    if not isinstance(correct_player, pd.Series) or not isinstance(
        target_player, pd.Series
    ):
        raise TypeError("Question player data is invalid.")

    chosen_similarity = float(selected_player["Similarity_Score"])
    best_similarity = float(correct_player["Similarity_Score"])
    points = calculate_round_points(chosen_similarity, best_similarity)
    exact_match = str(selected_player["Player_Key"]) == str(
        correct_player["Player_Key"]
    )

    match_index = points
    if exact_match:
        verdict = "Model Match"
    elif match_index >= 95:
        verdict = "Elite Read"
    elif match_index >= 88:
        verdict = "Strong Read"
    elif match_index >= 75:
        verdict = "Competitive Choice"
    else:
        verdict = "Risky Choice"

    profile_analysis = _pair_profile_analysis(
        target_player,
        correct_player,
        list(question.get("selected_features", [])),
    )

    return {
        "selected_player": selected_player,
        "correct_player": correct_player,
        "target_player": target_player,
        "chosen_similarity": chosen_similarity,
        "best_similarity": best_similarity,
        "base_points": points,
        "match_index": match_index,
        "exact_match": exact_match,
        "verdict": verdict,
        **profile_analysis,
    }
