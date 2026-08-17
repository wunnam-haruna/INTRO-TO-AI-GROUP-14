from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from src.game_engine import create_question


def test_question_has_four_options():
    question = create_question(
        difficulty="Scout",
        game_mode="Scout Mode",
    )

    assert len(question["options"]) == 4


def test_question_has_target():
    question = create_question(
        difficulty="Scout",
        game_mode="Scout Mode",
    )

    assert question["target_player"] is not None


def test_best_option_exists():
    question = create_question(
        difficulty="Scout",
        game_mode="Scout Mode",
    )

    assert question["correct_player"] is not None
