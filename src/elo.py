import pandas as pd

from src.config import INITIAL_ELO, ELO_K, HOME_ADVANTAGE


def expected_score(rating_a: float, rating_b: float) -> float:
    """
    Calculate expected score for team A against team B.
    """
    return 1 / (1 + 10 ** ((rating_b - rating_a) / 400))


def actual_score(goals_for: int, goals_against: int) -> float:
    """
    Convert match result into Elo score.
    """
    if goals_for > goals_against:
        return 1.0
    if goals_for == goals_against:
        return 0.5
    return 0.0


def build_elo_features(
    df: pd.DataFrame,
    initial_elo: int = INITIAL_ELO,
    k: int = ELO_K,
    home_advantage: int = HOME_ADVANTAGE,
) -> pd.DataFrame:
    """
    Build Elo features match by match.

    Important:
    - Elo before the match is saved first.
    - Elo is updated only after that.
    - Home advantage is applied only for expectation calculation.
    - If neutral == True, home advantage is not applied.
    """
    required_cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "neutral",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    ratings = {}

    home_elo_before = []
    away_elo_before = []

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_rating = ratings.get(home_team, initial_elo)
        away_rating = ratings.get(away_team, initial_elo)

        home_elo_before.append(home_rating)
        away_elo_before.append(away_rating)

        home_rating_for_expected = home_rating
        if not row["neutral"]:
            home_rating_for_expected += home_advantage

        expected_home = expected_score(home_rating_for_expected, away_rating)
        expected_away = 1 - expected_home

        actual_home = actual_score(row["home_score"], row["away_score"])
        actual_away = 1 - actual_home

        new_home_rating = home_rating + k * (actual_home - expected_home)
        new_away_rating = away_rating + k * (actual_away - expected_away)

        ratings[home_team] = new_home_rating
        ratings[away_team] = new_away_rating

    df["home_elo_before"] = home_elo_before
    df["away_elo_before"] = away_elo_before
    df["elo_diff"] = df["home_elo_before"] - df["away_elo_before"]

    return df

def calculate_current_elo_ratings(
    df: pd.DataFrame,
    initial_elo: int = INITIAL_ELO,
    k: int = ELO_K,
    home_advantage: int = HOME_ADVANTAGE,
) -> dict:
    """
    Calculate current Elo ratings after processing all played matches.

    Returns
    -------
    dict
        Team name -> current Elo rating.
    """
    required_cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "neutral",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()
    df = df.sort_values("date").reset_index(drop=True)

    ratings = {}

    for _, row in df.iterrows():
        home_team = row["home_team"]
        away_team = row["away_team"]

        home_rating = ratings.get(home_team, initial_elo)
        away_rating = ratings.get(away_team, initial_elo)

        home_rating_for_expected = home_rating
        if not row["neutral"]:
            home_rating_for_expected += home_advantage

        expected_home = expected_score(home_rating_for_expected, away_rating)
        expected_away = 1 - expected_home

        actual_home = actual_score(row["home_score"], row["away_score"])
        actual_away = 1 - actual_home

        ratings[home_team] = home_rating + k * (actual_home - expected_home)
        ratings[away_team] = away_rating + k * (actual_away - expected_away)

    return ratings