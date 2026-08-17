from __future__ import annotations

from html import escape
from textwrap import dedent

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from src.game_engine import create_question, evaluate_choice


# -----------------------------------------------------------------------------
# Application configuration
# -----------------------------------------------------------------------------

st.set_page_config(
    page_title="StyleMatch AI",
    page_icon="⚽",
    layout="wide",
    initial_sidebar_state="collapsed",
)

TOTAL_ROUNDS = 10
GAME_MODES = ("Star Match", "Scout Mode", "Blind Scout")
DIFFICULTIES = ("Rookie", "Scout", "Analyst", "Elite")
DIFFICULTY_MULTIPLIERS = {
    "Rookie": 1.00,
    "Scout": 1.10,
    "Analyst": 1.22,
    "Elite": 1.35,
}

MODE_COPY = {
    "Star Match": (
        "Featured names from Europe’s major leagues, with model-generated "
        "playstyle comparisons."
    ),
    "Scout Mode": (
        "The complete eligible player database. Expect stars, specialists "
        "and less obvious discoveries."
    ),
    "Blind Scout": (
        "Candidate identities stay hidden until the reveal. Judge the profile, "
        "not the reputation."
    ),
}

DEFAULT_STATE = {
    "screen": "landing",
    "game_mode": "Star Match",
    "difficulty": "Scout",
    "score": 0,
    "round_number": 1,
    "streak": 0,
    "best_streak": 0,
    "exact_matches": 0,
    "question": None,
    "answered": False,
    "result": None,
    "history": [],
    "used_target_keys": set(),
}


# -----------------------------------------------------------------------------
# Styling
# -----------------------------------------------------------------------------

st.markdown(
    dedent(
        """
        <style>
        :root {
            --bg: #050806;
            --panel: #0c120f;
            --panel-2: #101914;
            --line: rgba(203, 224, 210, 0.13);
            --muted: #92a198;
            --text: #f4f7f5;
            --green: #50e38a;
            --green-2: #19b964;
            --amber: #e6b65d;
            --red: #e16d6d;
        }

        html, body, [class*="css"] {
            font-family: Inter, ui-sans-serif, system-ui, -apple-system,
                BlinkMacSystemFont, "Segoe UI", sans-serif;
        }

        .stApp {
            background:
                radial-gradient(circle at 88% 2%, rgba(28, 101, 61, .25), transparent 30%),
                radial-gradient(circle at 8% 92%, rgba(20, 66, 43, .16), transparent 28%),
                var(--bg);
            color: var(--text);
        }

        header[data-testid="stHeader"] {
            background: transparent;
        }

        #MainMenu, footer, div[data-testid="stToolbar"] {
            visibility: hidden;
        }

        .block-container {
            max-width: 1320px;
            padding-top: 1.4rem;
            padding-bottom: 4rem;
        }

        .sm-topline {
            display: flex;
            align-items: center;
            justify-content: space-between;
            gap: 20px;
            padding: 10px 0 28px 0;
        }

        .sm-wordmark {
            font-weight: 900;
            letter-spacing: -1.3px;
            font-size: 1.35rem;
            color: #fff;
        }

        .sm-wordmark span { color: var(--green); }

        .sm-kicker {
            text-transform: uppercase;
            letter-spacing: 2.4px;
            font-weight: 800;
            font-size: .71rem;
            color: var(--green);
            margin-bottom: 14px;
        }

        .sm-hero-title {
            font-size: clamp(3rem, 7vw, 6.6rem);
            line-height: .94;
            letter-spacing: -4px;
            font-weight: 950;
            margin: 0;
            color: #fff;
            max-width: 1050px;
        }

        .sm-hero-subtitle {
            max-width: 760px;
            font-size: 1.18rem;
            line-height: 1.7;
            color: #a7b2ab;
            margin-top: 24px;
            margin-bottom: 34px;
        }

        .sm-stat-grid {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 12px;
            margin: 22px 0 36px;
        }

        .sm-stat {
            border: 1px solid var(--line);
            background: rgba(12, 18, 15, .76);
            border-radius: 16px;
            padding: 18px;
        }

        .sm-stat-value {
            font-size: 1.7rem;
            font-weight: 900;
            color: #fff;
            letter-spacing: -1px;
        }

        .sm-stat-label {
            margin-top: 4px;
            color: var(--muted);
            font-size: .78rem;
            text-transform: uppercase;
            letter-spacing: 1.2px;
        }

        .sm-mode-card {
            min-height: 190px;
            border: 1px solid var(--line);
            background: linear-gradient(145deg, rgba(17, 27, 21, .96), rgba(9, 14, 11, .96));
            border-radius: 20px;
            padding: 24px;
            transition: border-color .2s ease, transform .2s ease;
        }

        .sm-mode-card.selected {
            border-color: rgba(80, 227, 138, .75);
            box-shadow: 0 0 0 1px rgba(80, 227, 138, .17) inset;
        }

        .sm-mode-index {
            color: var(--green);
            font-size: .72rem;
            font-weight: 850;
            letter-spacing: 1.6px;
        }

        .sm-mode-name {
            font-size: 1.35rem;
            font-weight: 850;
            margin-top: 22px;
            color: #fff;
        }

        .sm-mode-copy {
            margin-top: 10px;
            color: #9eaaa2;
            line-height: 1.55;
            font-size: .92rem;
        }

        .sm-hud {
            display: grid;
            grid-template-columns: 1.35fr repeat(4, minmax(0, .7fr));
            gap: 10px;
            margin-bottom: 22px;
        }

        .sm-hud-cell {
            border: 1px solid var(--line);
            background: rgba(10, 16, 12, .88);
            border-radius: 14px;
            padding: 14px 16px;
        }

        .sm-hud-brand {
            font-weight: 900;
            font-size: 1.08rem;
            letter-spacing: -.4px;
        }

        .sm-hud-brand span { color: var(--green); }

        .sm-hud-label {
            color: #77857d;
            font-size: .65rem;
            text-transform: uppercase;
            letter-spacing: 1.3px;
        }

        .sm-hud-value {
            color: #fff;
            margin-top: 5px;
            font-size: 1.02rem;
            font-weight: 850;
        }

        .sm-progress-track {
            height: 3px;
            background: rgba(255,255,255,.08);
            border-radius: 999px;
            overflow: hidden;
            margin: 4px 0 28px;
        }

        .sm-progress-fill {
            height: 100%;
            background: linear-gradient(90deg, var(--green-2), var(--green));
        }

        .sm-target {
            position: relative;
            overflow: hidden;
            border: 1px solid rgba(80, 227, 138, .30);
            background:
                linear-gradient(105deg, rgba(14, 32, 22, .98), rgba(8, 14, 11, .98));
            border-radius: 24px;
            padding: 30px;
            margin-bottom: 24px;
        }

        .sm-target::after {
            content: "";
            position: absolute;
            width: 360px;
            height: 360px;
            border: 1px solid rgba(80, 227, 138, .13);
            border-radius: 50%;
            right: -120px;
            top: -170px;
        }

        .sm-target-grid {
            display: grid;
            grid-template-columns: 100px 1fr;
            gap: 22px;
            align-items: center;
        }

        .sm-avatar {
            width: 88px;
            height: 88px;
            display: flex;
            align-items: center;
            justify-content: center;
            border-radius: 22px;
            background: linear-gradient(135deg, #1f4c33, #0c2117);
            border: 1px solid rgba(80, 227, 138, .38);
            color: #dff9e8;
            font-size: 1.55rem;
            font-weight: 900;
            letter-spacing: -1px;
        }

        .sm-target-overline {
            color: var(--green);
            font-size: .68rem;
            font-weight: 900;
            text-transform: uppercase;
            letter-spacing: 2px;
        }

        .sm-target-name {
            color: #fff;
            font-size: clamp(2rem, 4vw, 3.35rem);
            font-weight: 950;
            letter-spacing: -2px;
            line-height: 1.02;
            margin-top: 5px;
        }

        .sm-target-meta {
            color: #a5b1a9;
            margin-top: 10px;
            font-size: .95rem;
        }

        .sm-question {
            color: #fff;
            font-size: 1.02rem;
            font-weight: 700;
            margin-top: 18px;
        }

        .sm-section-label {
            color: #89978f;
            font-size: .7rem;
            text-transform: uppercase;
            letter-spacing: 1.7px;
            font-weight: 850;
            margin: 12px 0 8px;
        }

        .sm-candidate {
            min-height: 184px;
            border: 1px solid var(--line);
            background: linear-gradient(150deg, rgba(16, 24, 19, .96), rgba(8, 12, 10, .98));
            border-radius: 20px;
            padding: 21px;
            margin-bottom: 8px;
        }

        .sm-candidate.correct {
            border-color: rgba(80, 227, 138, .85);
            box-shadow: 0 0 0 1px rgba(80, 227, 138, .18) inset;
        }

        .sm-candidate.selected-wrong {
            border-color: rgba(230, 182, 93, .85);
            box-shadow: 0 0 0 1px rgba(230, 182, 93, .16) inset;
        }

        .sm-candidate-top {
            display: flex;
            justify-content: space-between;
            gap: 12px;
            align-items: flex-start;
        }

        .sm-candidate-letter {
            width: 34px;
            height: 34px;
            border-radius: 10px;
            display: inline-flex;
            justify-content: center;
            align-items: center;
            background: #17221b;
            color: var(--green);
            border: 1px solid rgba(80, 227, 138, .22);
            font-weight: 900;
        }

        .sm-league-pill {
            border: 1px solid rgba(255,255,255,.12);
            color: #87968d;
            padding: 5px 8px;
            border-radius: 999px;
            font-size: .65rem;
            font-weight: 800;
            letter-spacing: .9px;
        }

        .sm-candidate-name {
            color: #fff;
            font-size: 1.28rem;
            font-weight: 880;
            margin-top: 24px;
            letter-spacing: -.5px;
        }

        .sm-candidate-meta {
            color: #88968e;
            font-size: .84rem;
            margin-top: 6px;
        }

        .sm-style-row {
            display: grid;
            grid-template-columns: 86px 1fr;
            align-items: center;
            gap: 10px;
            margin-top: 9px;
            color: #89978f;
            font-size: .72rem;
        }

        .sm-mini-track {
            height: 5px;
            border-radius: 999px;
            background: rgba(255,255,255,.08);
            overflow: hidden;
        }

        .sm-mini-fill {
            height: 100%;
            background: #4edb85;
        }

        div.stButton > button {
            width: 100%;
            min-height: 46px;
            border-radius: 12px;
            border: 1px solid rgba(80, 227, 138, .32);
            background: #10261a;
            color: #dff9e8;
            font-weight: 850;
            letter-spacing: .4px;
        }

        div.stButton > button:hover {
            border-color: var(--green);
            color: #fff;
            background: #153422;
        }

        div.stButton > button[kind="primary"] {
            background: var(--green);
            color: #041008;
            border-color: var(--green);
        }

        .sm-verdict {
            border: 1px solid var(--line);
            border-radius: 22px;
            background: rgba(11, 18, 14, .94);
            padding: 24px;
            margin: 26px 0 18px;
        }

        .sm-verdict-label {
            color: var(--green);
            font-size: .7rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            font-weight: 900;
        }

        .sm-verdict-title {
            color: #fff;
            font-size: 2rem;
            font-weight: 950;
            margin-top: 6px;
            letter-spacing: -1px;
        }

        .sm-verdict-copy {
            color: #a2aea6;
            line-height: 1.65;
            margin-top: 10px;
        }

        .sm-result-ribbon {
            display: grid;
            grid-template-columns: repeat(4, minmax(0, 1fr));
            gap: 10px;
            margin: 18px 0;
        }

        .sm-result-stat {
            border: 1px solid var(--line);
            background: #0d1511;
            border-radius: 14px;
            padding: 16px;
        }

        .sm-result-stat .value {
            font-size: 1.45rem;
            color: #fff;
            font-weight: 900;
        }

        .sm-result-stat .label {
            color: #7f8e85;
            font-size: .67rem;
            text-transform: uppercase;
            letter-spacing: 1px;
            margin-top: 4px;
        }

        .sm-disclaimer {
            border-left: 2px solid #34523f;
            color: #849087;
            font-size: .8rem;
            line-height: 1.55;
            padding-left: 14px;
            margin: 22px 0;
        }

        @media (max-width: 850px) {
            .sm-stat-grid, .sm-result-ribbon { grid-template-columns: repeat(2, 1fr); }
            .sm-hud { grid-template-columns: repeat(2, 1fr); }
            .sm-hud-cell:first-child { grid-column: 1 / -1; }
            .sm-target-grid { grid-template-columns: 1fr; }
            .sm-hero-title { letter-spacing: -2px; }
        }
        </style>
        """
    ).strip(),
    unsafe_allow_html=True,
)


# -----------------------------------------------------------------------------
# Utilities
# -----------------------------------------------------------------------------


def render_html(markup: str) -> None:
    st.markdown(dedent(markup).strip(), unsafe_allow_html=True)


def initialise_state() -> None:
    for key, value in DEFAULT_STATE.items():
        if key not in st.session_state:
            st.session_state[key] = value.copy() if isinstance(value, (list, set)) else value


def reset_game(*, keep_settings: bool = True) -> None:
    mode = st.session_state.game_mode
    difficulty = st.session_state.difficulty
    for key, value in DEFAULT_STATE.items():
        st.session_state[key] = value.copy() if isinstance(value, (list, set)) else value
    if keep_settings:
        st.session_state.game_mode = mode
        st.session_state.difficulty = difficulty


def start_game() -> None:
    reset_game(keep_settings=True)
    st.session_state.screen = "game"


def scout_rank(score: int) -> str:
    if score >= 1350:
        return "Elite Talent Identifier"
    if score >= 1120:
        return "Sporting Director"
    if score >= 880:
        return "Head Scout"
    if score >= 650:
        return "Senior Scout"
    if score >= 420:
        return "Recruitment Analyst"
    return "Rookie Scout"


def awarded_points(result: dict[str, object], difficulty: str, streak: int) -> int:
    base = int(result["base_points"])
    score = base * DIFFICULTY_MULTIPLIERS[difficulty]
    if bool(result["exact_match"]):
        score += 20
        score += min(max(streak - 1, 0) * 4, 20)
    return int(round(score))


def blind_dimensions(player: pd.Series) -> dict[str, int]:
    groups = {
        "Attack": ["Gls_per90_z", "G-PK_per90_z", "Sh_per90_z", "SoT_per90_z"],
        "Creation": ["Ast_per90_z", "Crs_per90_z", "Fld_per90_z"],
        "Defence": ["TklW_per90_z", "Int_per90_z"],
    }
    values: dict[str, int] = {}
    for label, columns in groups.items():
        available = [column for column in columns if column in player.index]
        z_value = float(np.mean([float(player[column]) for column in available])) if available else 0.0
        values[label] = int(np.clip(round(50 + z_value * 16), 8, 92))
    return values


def candidate_card(
    player: pd.Series,
    *,
    blind: bool,
    answered: bool,
    selected_key: str | None,
    correct_key: str | None,
) -> str:
    player_key = str(player["Player_Key"])
    css_class = ""
    if answered and player_key == correct_key:
        css_class = "correct"
    elif answered and player_key == selected_key:
        css_class = "selected-wrong"

    letter = escape(str(player["Candidate_Letter"]))
    league = escape(str(player["League_Short"]))

    if blind and not answered:
        name = f"Candidate {letter}"
        meta = "Identity concealed • profile-only decision"
        dimensions = blind_dimensions(player)
        style_rows = "".join(
            f'<div class="sm-style-row"><span>{escape(label)}</span>'
            f'<div class="sm-mini-track"><div class="sm-mini-fill" style="width:{value}%"></div></div></div>'
            for label, value in dimensions.items()
        )
    else:
        name = escape(str(player["Player"]))
        meta = (
            f'{escape(str(player["Squad"]))} • '
            f'{escape(str(player["League_Display"]))} • '
            f'{escape(str(player["Position_Group"]))}'
        )
        style_rows = ""
        if answered:
            style_rows = (
                f'<div class="sm-style-row"><span>AI role</span>'
                f'<span>{escape(str(player["Final_Role_Name"]))}</span></div>'
            )

    return dedent(
        f"""
        <div class="sm-candidate {css_class}">
            <div class="sm-candidate-top">
                <div class="sm-candidate-letter">{letter}</div>
                <div class="sm-league-pill">{league}</div>
            </div>
            <div class="sm-candidate-name">{name}</div>
            <div class="sm-candidate-meta">{meta}</div>
            {style_rows}
        </div>
        """
    ).strip()


def similarity_figure(options: pd.DataFrame, result: dict[str, object]) -> go.Figure:
    chart = options.copy()
    best = float(chart["Similarity_Score"].max())
    chart["Match_Index"] = (chart["Similarity_Score"] / best * 100).round(1)
    chart = chart.sort_values("Match_Index", ascending=True)

    selected_key = str(result["selected_player"]["Player_Key"])
    correct_key = str(result["correct_player"]["Player_Key"])
    colours = []
    for key in chart["Player_Key"]:
        if str(key) == correct_key:
            colours.append("#50e38a")
        elif str(key) == selected_key:
            colours.append("#e6b65d")
        else:
            colours.append("#425249")

    figure = go.Figure(
        go.Bar(
            x=chart["Match_Index"],
            y=chart["Player"],
            orientation="h",
            marker_color=colours,
            text=chart["Match_Index"].astype(str) + "%",
            textposition="outside",
            hovertemplate="%{y}<br>Round match index: %{x:.1f}%<extra></extra>",
        )
    )
    figure.update_layout(
        height=330,
        margin=dict(l=10, r=45, t=25, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#dbe5de"),
        xaxis=dict(
            title="Round match index",
            range=[0, 108],
            showgrid=True,
            gridcolor="rgba(255,255,255,.06)",
            zeroline=False,
        ),
        yaxis=dict(title="", showgrid=False),
    )
    return figure


initialise_state()


# -----------------------------------------------------------------------------
# Shared top line
# -----------------------------------------------------------------------------

render_html(
    """
    <div class="sm-topline">
        <div class="sm-wordmark">STYLEMATCH <span>/ AI</span></div>
        <div class="sm-kicker">Position-aware football intelligence</div>
    </div>
    """
)


# -----------------------------------------------------------------------------
# Landing screen
# -----------------------------------------------------------------------------

if st.session_state.screen == "landing":
    render_html(
        """
        <div class="sm-kicker">The playstyle similarity challenge</div>
        <h1 class="sm-hero-title">WHO PLAYS<br>LIKE WHO?</h1>
        <div class="sm-hero-subtitle">
            One target player. Four credible alternatives. Your football intuition
            is scored against similarities discovered from position-standardised
            performance data and K-Means-derived player profiles.
        </div>
        <div class="sm-stat-grid">
            <div class="sm-stat"><div class="sm-stat-value">1,464</div><div class="sm-stat-label">Eligible profiles</div></div>
            <div class="sm-stat"><div class="sm-stat-value">5</div><div class="sm-stat-label">Major leagues</div></div>
            <div class="sm-stat"><div class="sm-stat-value">6</div><div class="sm-stat-label">AI archetypes</div></div>
            <div class="sm-stat"><div class="sm-stat-value">10</div><div class="sm-stat-label">Rounds per game</div></div>
        </div>
        """
    )

    st.markdown("### Select a game mode")
    mode_columns = st.columns(3)
    for index, mode in enumerate(GAME_MODES, start=1):
        with mode_columns[index - 1]:
            selected_class = "selected" if st.session_state.game_mode == mode else ""
            render_html(
                f"""
                <div class="sm-mode-card {selected_class}">
                    <div class="sm-mode-index">MODE 0{index}</div>
                    <div class="sm-mode-name">{escape(mode)}</div>
                    <div class="sm-mode-copy">{escape(MODE_COPY[mode])}</div>
                </div>
                """
            )
            if st.button(
                "Selected" if st.session_state.game_mode == mode else f"Choose {mode}",
                key=f"mode_{mode}",
                use_container_width=True,
            ):
                st.session_state.game_mode = mode
                st.rerun()

    st.markdown("### Set the difficulty")
    difficulty = st.radio(
        "Difficulty",
        DIFFICULTIES,
        index=DIFFICULTIES.index(st.session_state.difficulty),
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state.difficulty = difficulty

    st.caption(
        "Rookie uses wider similarity gaps. Elite selects candidates from the target's nearest statistical neighbours."
    )

    if st.button("START THE CHALLENGE", type="primary", use_container_width=True):
        start_game()
        st.rerun()

    render_html(
        """
        <div class="sm-disclaimer">
            The model is a statistical reference, not absolute football truth.
            Tactical fit, physical qualities, injuries, mentality and live scouting
            remain outside the current dataset.
        </div>
        """
    )
    st.stop()


# -----------------------------------------------------------------------------
# Final results
# -----------------------------------------------------------------------------

if st.session_state.screen == "results":
    history = pd.DataFrame(st.session_state.history)
    average_quality = (
        float(history["Choice quality"].mean()) if not history.empty else 0.0
    )
    rank = scout_rank(st.session_state.score)

    render_html(
        f"""
        <div class="sm-kicker">Final scout report</div>
        <h1 class="sm-hero-title">{escape(rank.upper())}</h1>
        <div class="sm-hero-subtitle">
            Your football intuition aligned with the model at an average choice quality
            of {average_quality:.1f}% across {TOTAL_ROUNDS} rounds.
        </div>
        <div class="sm-stat-grid">
            <div class="sm-stat"><div class="sm-stat-value">{st.session_state.score:,}</div><div class="sm-stat-label">Final score</div></div>
            <div class="sm-stat"><div class="sm-stat-value">{st.session_state.exact_matches}/{TOTAL_ROUNDS}</div><div class="sm-stat-label">Exact matches</div></div>
            <div class="sm-stat"><div class="sm-stat-value">{average_quality:.1f}%</div><div class="sm-stat-label">Average choice quality</div></div>
            <div class="sm-stat"><div class="sm-stat-value">{st.session_state.best_streak}</div><div class="sm-stat-label">Best streak</div></div>
        </div>
        """
    )

    if not history.empty:
        position_summary = (
            history.groupby("Position", as_index=False)
            .agg(
                Rounds=("Round", "count"),
                Average_Choice_Quality=("Choice quality", "mean"),
                Exact_Matches=("Exact", lambda values: int((values == "Yes").sum())),
            )
            .round({"Average_Choice_Quality": 1})
        )
        st.markdown("### Position recognition")
        st.dataframe(position_summary, hide_index=True, use_container_width=True)
        st.markdown("### Decision history")
        st.dataframe(history, hide_index=True, use_container_width=True)

    render_html(
        """
        <div class="sm-disclaimer">
            This report measures agreement with a statistical similarity model, not scouting
            accuracy in absolute terms. Human experts may correctly use information that the
            model does not contain.
        </div>
        """
    )

    left, right = st.columns(2)
    with left:
        if st.button("PLAY AGAIN", type="primary", use_container_width=True):
            start_game()
            st.rerun()
    with right:
        if st.button("CHANGE MODE", use_container_width=True):
            reset_game(keep_settings=False)
            st.rerun()

    st.stop()

# -----------------------------------------------------------------------------
# Generate / load question
# -----------------------------------------------------------------------------

if st.session_state.question is None:
    with st.spinner("Building a model-backed matchup..."):
        try:
            st.session_state.question = create_question(
                difficulty=st.session_state.difficulty,
                game_mode=st.session_state.game_mode,
                excluded_target_keys=set(st.session_state.used_target_keys),
            )
            target_key = str(st.session_state.question["target_player"]["Player_Key"])
            st.session_state.used_target_keys.add(target_key)
        except Exception as error:
            st.error(f"The matchup could not be generated: {error}")
            if st.button("Return to mode selection"):
                reset_game(keep_settings=True)
                st.rerun()
            st.stop()

question = st.session_state.question
target: pd.Series = question["target_player"]
options: pd.DataFrame = question["options"]
result = st.session_state.result


# -----------------------------------------------------------------------------
# Game HUD
# -----------------------------------------------------------------------------

progress = ((st.session_state.round_number - 1) / TOTAL_ROUNDS) * 100
render_html(
    f"""
    <div class="sm-hud">
        <div class="sm-hud-cell"><div class="sm-hud-brand">STYLEMATCH <span>/ AI</span></div><div class="sm-hud-label">{escape(st.session_state.game_mode)}</div></div>
        <div class="sm-hud-cell"><div class="sm-hud-label">Round</div><div class="sm-hud-value">{st.session_state.round_number:02d} / {TOTAL_ROUNDS:02d}</div></div>
        <div class="sm-hud-cell"><div class="sm-hud-label">Score</div><div class="sm-hud-value">{st.session_state.score:,}</div></div>
        <div class="sm-hud-cell"><div class="sm-hud-label">Streak</div><div class="sm-hud-value">{st.session_state.streak}</div></div>
        <div class="sm-hud-cell"><div class="sm-hud-label">Difficulty</div><div class="sm-hud-value">{escape(st.session_state.difficulty)}</div></div>
    </div>
    <div class="sm-progress-track"><div class="sm-progress-fill" style="width:{progress}%"></div></div>
    """
)

exit_col, spacer_col = st.columns([1, 7])
with exit_col:
    if st.button("Exit game", use_container_width=True):
        reset_game(keep_settings=True)
        st.rerun()


# -----------------------------------------------------------------------------
# Target player
# -----------------------------------------------------------------------------

role_reveal = ""
if st.session_state.answered:
    role_reveal = f" • AI profile: {escape(str(target['Final_Role_Name']))}"

render_html(
    f"""
    <div class="sm-target">
        <div class="sm-target-grid">
            <div class="sm-avatar">{escape(str(target['Initials']))}</div>
            <div>
                <div class="sm-target-overline">Target player</div>
                <div class="sm-target-name">{escape(str(target['Player']))}</div>
                <div class="sm-target-meta">{escape(str(target['Squad']))} • {escape(str(target['League_Display']))} • {escape(str(target['Position_Group']))}{role_reveal}</div>
                <div class="sm-question">Which candidate has the closest statistical playstyle among these four?</div>
            </div>
        </div>
    </div>
    """
)

render_html('<div class="sm-section-label">Candidate board</div>')


# -----------------------------------------------------------------------------
# Candidate grid
# -----------------------------------------------------------------------------

selected_key = None
correct_key = None
if result:
    selected_key = str(result["selected_player"]["Player_Key"])
    correct_key = str(result["correct_player"]["Player_Key"])

option_rows = [options.iloc[:2], options.iloc[2:4]]
for option_row in option_rows:
    columns = st.columns(2, gap="medium")
    for column, (_, candidate) in zip(columns, option_row.iterrows()):
        with column:
            render_html(
                candidate_card(
                    candidate,
                    blind=(st.session_state.game_mode == "Blind Scout"),
                    answered=st.session_state.answered,
                    selected_key=selected_key,
                    correct_key=correct_key,
                )
            )
            label = (
                f"SELECT {candidate['Candidate_Letter']}"
                if not st.session_state.answered
                else "ANSWER LOCKED"
            )
            if st.button(
                label,
                key=f"candidate_{question['question_id']}_{candidate['Player_Key']}",
                disabled=st.session_state.answered,
                use_container_width=True,
            ):
                try:
                    evaluated = evaluate_choice(question, str(candidate["Player_Key"]))
                    if bool(evaluated["exact_match"]):
                        st.session_state.exact_matches += 1
                        st.session_state.streak += 1
                        st.session_state.best_streak = max(
                            st.session_state.best_streak, st.session_state.streak
                        )
                    else:
                        st.session_state.streak = 0

                    round_points = awarded_points(
                        evaluated,
                        st.session_state.difficulty,
                        st.session_state.streak,
                    )
                    evaluated["awarded_points"] = round_points
                    st.session_state.score += round_points
                    st.session_state.result = evaluated
                    st.session_state.answered = True
                    st.session_state.history.append(
                        {
                            "Round": st.session_state.round_number,
                            "Position": str(target["Position_Group"]),
                            "Target": str(target["Player"]),
                            "Selection": str(evaluated["selected_player"]["Player"]),
                            "Best match": str(evaluated["correct_player"]["Player"]),
                            "Choice quality": int(evaluated["match_index"]),
                            "Points": round_points,
                            "Exact": "Yes" if evaluated["exact_match"] else "No",
                        }
                    )
                    st.rerun()
                except Exception as error:
                    st.error(f"Your selection could not be evaluated: {error}")


# -----------------------------------------------------------------------------
# Answer reveal
# -----------------------------------------------------------------------------

if st.session_state.answered and result:
    selected = result["selected_player"]
    correct = result["correct_player"]
    aligned = ", ".join(result["aligned_features"]) or "the available model features"
    differences = ", ".join(result["different_features"]) or "several secondary attributes"

    if bool(result["exact_match"]):
        verdict_copy = (
            f"{escape(str(correct['Player']))} was the strongest match in this candidate set. "
            f"The target and match were most closely aligned in {escape(aligned)}."
        )
    else:
        verdict_copy = (
            f"Your selection, {escape(str(selected['Player']))}, achieved a {int(result['match_index'])}% "
            f"round match index. The model's best option was {escape(str(correct['Player']))}. "
            f"The best pair aligned most strongly in {escape(aligned)}, while their larger differences appeared in {escape(differences)}."
        )

    render_html(
        f"""
        <div class="sm-verdict">
            <div class="sm-verdict-label">Model reveal</div>
            <div class="sm-verdict-title">{escape(str(result['verdict']))}</div>
            <div class="sm-verdict-copy">{verdict_copy}</div>
        </div>
        <div class="sm-result-ribbon">
            <div class="sm-result-stat"><div class="value">+{int(result['awarded_points'])}</div><div class="label">Round points</div></div>
            <div class="sm-result-stat"><div class="value">{int(result['match_index'])}%</div><div class="label">Choice quality</div></div>
            <div class="sm-result-stat"><div class="value">{escape(str(correct['Player']))}</div><div class="label">Best match</div></div>
            <div class="sm-result-stat"><div class="value">{st.session_state.streak}</div><div class="label">Current streak</div></div>
        </div>
        """
    )

    st.plotly_chart(similarity_figure(options, result), use_container_width=True)

    ranking = options.copy()
    best_similarity = float(ranking["Similarity_Score"].max())
    ranking["Round Match Index"] = (
        ranking["Similarity_Score"] / best_similarity * 100
    ).round(1)
    ranking = ranking.sort_values("Round Match Index", ascending=False)
    st.dataframe(
        ranking[
            [
                "Player",
                "Squad",
                "League_Display",
                "Final_Role_Name",
                "Round Match Index",
            ]
        ].rename(
            columns={
                "League_Display": "League",
                "Final_Role_Name": "AI Profile",
            }
        ),
        hide_index=True,
        use_container_width=True,
    )

    render_html(
        """
        <div class="sm-disclaimer">
            The round match index is relative to the strongest option shown in this round.
            It is not a probability and does not prove that two footballers are tactically
            interchangeable. The model compares only the statistical variables available.
        </div>
        """
    )

    next_label = (
        "VIEW FINAL SCOUT REPORT"
        if st.session_state.round_number >= TOTAL_ROUNDS
        else "NEXT MATCHUP"
    )
    if st.button(next_label, type="primary", use_container_width=True):
        if st.session_state.round_number >= TOTAL_ROUNDS:
            st.session_state.screen = "results"
        else:
            st.session_state.round_number += 1
            st.session_state.question = None
            st.session_state.answered = False
            st.session_state.result = None
        st.rerun()
