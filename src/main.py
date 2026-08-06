from __future__ import annotations

from clean_data import run_cleaning_pipeline
from feature_engineering import run_feature_engineering_pipeline
from train_kmeans import run_kmeans_training_pipeline
from validate_clusters import run_cluster_validation_pipeline


def run_full_pipeline() -> None:
    """Run the complete football player analysis pipeline."""

    print("=" * 70)
    print("FOOTBALL PLAYER PROFILING AND RECRUITMENT SUPPORT SYSTEM")
    print("=" * 70)

    print("\nSTEP 1: CLEANING RAW PLAYER DATA")
    print("-" * 70)
    run_cleaning_pipeline()

    print("\nSTEP 2: FEATURE ENGINEERING")
    print("-" * 70)
    run_feature_engineering_pipeline()

    print("\nSTEP 3: K-MEANS TRAINING")
    print("-" * 70)
    run_kmeans_training_pipeline()

    print("\nSTEP 4: CLUSTER VALIDATION")
    print("-" * 70)
    run_cluster_validation_pipeline()

    print("\n" + "=" * 70)
    print("FULL PIPELINE COMPLETED SUCCESSFULLY")
    print("=" * 70)

    print("\nYou can now run the player similarity engine with:")

    print("python src/player_similarity.py")


if __name__ == "__main__":
    run_full_pipeline()
