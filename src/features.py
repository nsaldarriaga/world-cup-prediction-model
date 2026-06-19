import numpy as np
import pandas as pd


# ============================================================
# Long-format dataset
# ============================================================

def create_matches_long(df: pd.DataFrame) -> pd.DataFrame:
    """
    Convert match-level dataset into team-match-level dataset.

    Each match creates two rows:
    - one from the home team perspective
    - one from the away team perspective

    This format is useful for calculating rolling team features.
    """
    required_cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "tournament",
        "city",
        "country",
        "neutral",
    ]

    missing_cols = [col for col in required_cols if col not in df.columns]
    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")

    df = df.copy()

    if "match_id" not in df.columns:
        df["match_id"] = range(len(df))

    home_rows = pd.DataFrame({
        "match_id": df["match_id"],
        "date": df["date"],
        "team": df["home_team"],
        "opponent": df["away_team"],
        "goals_for": df["home_score"],
        "goals_against": df["away_score"],
        "is_home": True,
        "tournament": df["tournament"],
        "neutral": df["neutral"],
    })

    away_rows = pd.DataFrame({
        "match_id": df["match_id"],
        "date": df["date"],
        "team": df["away_team"],
        "opponent": df["home_team"],
        "goals_for": df["away_score"],
        "goals_against": df["home_score"],
        "is_home": False,
        "tournament": df["tournament"],
        "neutral": df["neutral"],
    })

    matches_long = pd.concat([home_rows, away_rows], ignore_index=True)

    matches_long["points"] = np.select(
        [
            matches_long["goals_for"] > matches_long["goals_against"],
            matches_long["goals_for"] == matches_long["goals_against"],
        ],
        [3, 1],
        default=0,
    )

    matches_long = matches_long.sort_values(["team", "date"]).reset_index(drop=True)

    return matches_long

# ============================================================
# Rolling features
# ============================================================

def add_rolling_features(
    matches_long: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """
    Add rolling team-form features using shift(1) to avoid data leakage.

    Features created:
    - avg_goals_for_{window}
    - avg_goals_against_{window}
    - avg_points_{window}
    """
    df = matches_long.copy()
    df = df.sort_values(["team", "date"]).reset_index(drop=True)

    for window in windows:
        df[f"avg_goals_for_{window}"] = (
            df.groupby("team")["goals_for"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

        df[f"avg_goals_against_{window}"] = (
            df.groupby("team")["goals_against"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

        df[f"avg_points_{window}"] = (
            df.groupby("team")["points"]
            .transform(lambda s: s.shift(1).rolling(window, min_periods=1).mean())
        )

    return df


# ============================================================
# Tournament features
# ============================================================

def add_tournament_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add tournament category features.

    These features are intentionally simple and explainable.
    """
    df = df.copy()

    tournament = df["tournament"].fillna("").str.lower()

    df["is_friendly"] = tournament.str.contains("friendly", case=False, regex=False)

    df["is_world_cup"] = (
        tournament.str.contains("fifa world cup", case=False, regex=False)
        & ~tournament.str.contains("qualification", case=False, regex=False)
        & ~tournament.str.contains("qualifying", case=False, regex=False)
    )

    df["is_world_cup_qualifier"] = (
        tournament.str.contains("world cup", case=False, regex=False)
        & (
            tournament.str.contains("qualification", case=False, regex=False)
            | tournament.str.contains("qualifying", case=False, regex=False)
        )
    )

    df["is_continental_cup"] = tournament.str.contains(
        "uefa euro|copa américa|copa america|african cup|asian cup|gold cup|oceania nations",
        case=False,
        regex=True,
    )

    df["is_nations_league"] = tournament.str.contains(
        "nations league",
        case=False,
        regex=False,
    )

    df["is_competitive"] = ~df["is_friendly"]

    bool_cols = [
        "is_friendly",
        "is_world_cup",
        "is_world_cup_qualifier",
        "is_continental_cup",
        "is_nations_league",
        "is_competitive",
    ]

    df[bool_cols] = df[bool_cols].astype(int)

    return df


def build_latest_team_form_features(
    played_matches: pd.DataFrame,
    windows: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """
    Build latest available rolling form features for each team.

    Unlike add_rolling_features(), this is used for future predictions,
    so it includes the most recent played matches.
    """
    matches_long = create_matches_long(played_matches)
    matches_long = matches_long.sort_values(["team", "date"]).reset_index(drop=True)

    team_features = []

    for team, group in matches_long.groupby("team"):
        row = {"team": team}

        for window in windows:
            last_games = group.tail(window)

            row[f"avg_goals_for_{window}"] = last_games["goals_for"].mean()
            row[f"avg_goals_against_{window}"] = last_games["goals_against"].mean()
            row[f"avg_points_{window}"] = last_games["points"].mean()

        team_features.append(row)

    return pd.DataFrame(team_features)