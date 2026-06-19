from collections import defaultdict, deque

import numpy as np
import pandas as pd

from src.config import INITIAL_ELO, ELO_K, HOME_ADVANTAGE
from src.elo import expected_score, actual_score


def build_team_state(
    played_matches: pd.DataFrame,
    initial_elo: int = INITIAL_ELO,
    k: int = ELO_K,
    home_advantage: int = HOME_ADVANTAGE,
) -> dict:
    """
    Build current dynamic team state from historical played matches.

    The state stores:
    - current Elo rating
    - recent goals for
    - recent goals against
    - recent points

    This object will later be updated match by match.
    """
    df = played_matches.copy()
    df = df.sort_values("date").reset_index(drop=True)

    state = defaultdict(lambda: {
        "elo": initial_elo,
        "goals_for": deque(maxlen=10),
        "goals_against": deque(maxlen=10),
        "points": deque(maxlen=10),
    })

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_score = int(row["home_score"])
        away_score = int(row["away_score"])
        neutral = bool(row["neutral"])

        update_team_state_after_match(
            state=state,
            home_team=home_team,
            away_team=away_team,
            home_score=home_score,
            away_score=away_score,
            neutral=neutral,
            k=k,
            home_advantage=home_advantage,
        )

    return dict(state)


def get_team_form_features(
    state: dict,
    team: str,
    window: int,
) -> dict:
    """
    Get rolling form features for one team from the dynamic state.
    """
    if team not in state:
        return {
            f"avg_goals_for_{window}": np.nan,
            f"avg_goals_against_{window}": np.nan,
            f"avg_points_{window}": np.nan,
        }

    goals_for = list(state[team]["goals_for"])[-window:]
    goals_against = list(state[team]["goals_against"])[-window:]
    points = list(state[team]["points"])[-window:]

    return {
        f"avg_goals_for_{window}": np.mean(goals_for) if goals_for else np.nan,
        f"avg_goals_against_{window}": np.mean(goals_against) if goals_against else np.nan,
        f"avg_points_{window}": np.mean(points) if points else np.nan,
    }


def get_team_features(
    state: dict,
    team: str,
) -> dict:
    """
    Get all model-related team form features for one team.
    """
    features = {}

    for window in (5, 10):
        features.update(
            get_team_form_features(
                state=state,
                team=team,
                window=window,
            )
        )

    features["elo"] = state.get(team, {}).get("elo", INITIAL_ELO)

    return features


def update_team_state_after_match(
    state: dict,
    home_team: str,
    away_team: str,
    home_score: int,
    away_score: int,
    neutral: bool,
    k: int = ELO_K,
    home_advantage: int = HOME_ADVANTAGE,
) -> dict:
    """
    Update team state after a played or simulated match.

    Updates:
    - Elo
    - goals for
    - goals against
    - points
    """
    if home_team not in state:
        state[home_team] = {
            "elo": INITIAL_ELO,
            "goals_for": deque(maxlen=10),
            "goals_against": deque(maxlen=10),
            "points": deque(maxlen=10),
        }

    if away_team not in state:
        state[away_team] = {
            "elo": INITIAL_ELO,
            "goals_for": deque(maxlen=10),
            "goals_against": deque(maxlen=10),
            "points": deque(maxlen=10),
        }

    home_elo = state[home_team]["elo"]
    away_elo = state[away_team]["elo"]

    home_elo_for_expected = home_elo
    if not neutral:
        home_elo_for_expected += home_advantage

    expected_home = expected_score(home_elo_for_expected, away_elo)
    expected_away = 1 - expected_home

    actual_home = actual_score(home_score, away_score)
    actual_away = 1 - actual_home

    state[home_team]["elo"] = home_elo + k * (actual_home - expected_home)
    state[away_team]["elo"] = away_elo + k * (actual_away - expected_away)

    home_points, away_points = get_match_points(home_score, away_score)

    state[home_team]["goals_for"].append(home_score)
    state[home_team]["goals_against"].append(away_score)
    state[home_team]["points"].append(home_points)

    state[away_team]["goals_for"].append(away_score)
    state[away_team]["goals_against"].append(home_score)
    state[away_team]["points"].append(away_points)

    return state


def get_match_points(
    home_score: int,
    away_score: int,
) -> tuple[int, int]:
    """
    Return points for home and away teams.
    """
    if home_score > away_score:
        return 3, 0

    if home_score == away_score:
        return 1, 1

    return 0, 3


def team_state_to_dataframe(state: dict) -> pd.DataFrame:
    """
    Convert dynamic team state into a readable dataframe.
    """
    rows = []

    for team, values in state.items():
        row = {
            "team": team,
            "elo": values["elo"],
        }

        for window in (5, 10):
            row.update(
                get_team_form_features(
                    state=state,
                    team=team,
                    window=window,
                )
            )

        rows.append(row)

    return pd.DataFrame(rows).sort_values("elo", ascending=False).reset_index(drop=True)

def build_match_features_from_state(
    state: dict,
    home_team: str,
    away_team: str,
    neutral: bool,
    tournament_features: dict | None = None,
) -> dict:
    """
    Build one match-level feature row from the current dynamic team state.

    This is used for future or next-round predictions, where the real score
    is not known yet.

    The output is designed to be compatible with MODEL_FEATURES.
    """
    home_features = get_team_features(state, home_team)
    away_features = get_team_features(state, away_team)

    home_elo = home_features["elo"]
    away_elo = away_features["elo"]

    row = {
        "home_team": home_team,
        "away_team": away_team,
        "neutral": int(neutral),

        "home_elo_before": home_elo,
        "away_elo_before": away_elo,
        "elo_diff": home_elo - away_elo,
    }

    # Home rolling features
    for window in (5, 10):
        row[f"home_avg_goals_for_{window}"] = home_features[
            f"avg_goals_for_{window}"
        ]
        row[f"home_avg_goals_against_{window}"] = home_features[
            f"avg_goals_against_{window}"
        ]
        row[f"home_avg_points_{window}"] = home_features[
            f"avg_points_{window}"
        ]

    # Away rolling features
    for window in (5, 10):
        row[f"away_avg_goals_for_{window}"] = away_features[
            f"avg_goals_for_{window}"
        ]
        row[f"away_avg_goals_against_{window}"] = away_features[
            f"avg_goals_against_{window}"
        ]
        row[f"away_avg_points_{window}"] = away_features[
            f"avg_points_{window}"
        ]

    # Tournament features
    default_tournament_features = {
        "is_friendly": 0,
        "is_world_cup": 0,
        "is_world_cup_qualifier": 0,
        "is_continental_cup": 0,
        "is_nations_league": 0,
        "is_competitive": 0,
    }

    if tournament_features is not None:
        default_tournament_features.update(tournament_features)

    row.update(default_tournament_features)

    row = add_dynamic_difference_features(row)
    row = add_dynamic_elo_transformations(row)
    row = add_dynamic_parity_features(row)

    return row


def add_dynamic_difference_features(row: dict) -> dict:
    """
    Add home-away difference features to one dynamic match row.
    """
    row["diff_attack_5"] = (
        row["home_avg_goals_for_5"] - row["away_avg_goals_for_5"]
    )
    row["diff_defense_5"] = (
        row["home_avg_goals_against_5"] - row["away_avg_goals_against_5"]
    )
    row["diff_points_5"] = (
        row["home_avg_points_5"] - row["away_avg_points_5"]
    )

    row["diff_attack_10"] = (
        row["home_avg_goals_for_10"] - row["away_avg_goals_for_10"]
    )
    row["diff_defense_10"] = (
        row["home_avg_goals_against_10"] - row["away_avg_goals_against_10"]
    )
    row["diff_points_10"] = (
        row["home_avg_points_10"] - row["away_avg_points_10"]
    )

    return row


def add_dynamic_elo_transformations(row: dict) -> dict:
    """
    Add transformed Elo features to one dynamic match row.
    """
    row["elo_tanh_400"] = np.tanh(row["elo_diff"] / 400)
    row["abs_elo_tanh_400"] = abs(row["elo_tanh_400"])
    row["abs_elo_diff"] = abs(row["elo_diff"])

    return row


def add_dynamic_parity_features(row: dict) -> dict:
    """
    Add absolute-difference features to one dynamic match row.

    These help detect balanced matches, which are more likely to end in draws.
    """
    row["abs_diff_attack_5"] = abs(row["diff_attack_5"])
    row["abs_diff_defense_5"] = abs(row["diff_defense_5"])
    row["abs_diff_points_5"] = abs(row["diff_points_5"])

    row["abs_diff_attack_10"] = abs(row["diff_attack_10"])
    row["abs_diff_defense_10"] = abs(row["diff_defense_10"])
    row["abs_diff_points_10"] = abs(row["diff_points_10"])

    return row


def build_matches_features_from_state(
    state: dict,
    matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build model-ready features for multiple future matches using dynamic state.

    Expected input columns:
    - home_team
    - away_team
    - neutral

    Optional tournament columns:
    - is_friendly
    - is_world_cup
    - is_world_cup_qualifier
    - is_continental_cup
    - is_nations_league
    - is_competitive
    """
    tournament_cols = [
        "is_friendly",
        "is_world_cup",
        "is_world_cup_qualifier",
        "is_continental_cup",
        "is_nations_league",
        "is_competitive",
    ]

    rows = []

    for _, match in matches.iterrows():
        tournament_features = {
            col: int(match[col])
            for col in tournament_cols
            if col in matches.columns
        }

        row = build_match_features_from_state(
            state=state,
            home_team=match["home_team"],
            away_team=match["away_team"],
            neutral=bool(match["neutral"]),
            tournament_features=tournament_features,
        )

        # Preserve useful metadata if available
        for col in ["date", "tournament", "city", "country", "match_id"]:
            if col in matches.columns:
                row[col] = match[col]

        rows.append(row)

    return pd.DataFrame(rows)