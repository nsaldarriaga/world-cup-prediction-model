import pandas as pd

from src.config import INITIAL_ELO
from src.features import add_tournament_features
from src.model import MODEL_FEATURES, predict_expected_goals
from src.probabilities import (
    add_match_probabilities,
    calibrate_probabilities,
)
from src.simulation import simulate_match, monte_carlo_match_fast
from src.team_state import (
    get_team_features,
    update_team_state_after_match,
)


def build_match_features_from_state(
    match: pd.Series | dict,
    state: dict,
) -> pd.DataFrame:
    """
    Build a one-row model dataframe for a future match using dynamic team_state.

    Parameters
    ----------
    match : pd.Series | dict
        Future match row containing at least:
        date, home_team, away_team, tournament, neutral.
    state : dict
        Current team state with Elo and recent form.

    Returns
    -------
    pd.DataFrame
        One-row dataframe with model features.
    """
    if isinstance(match, dict):
        match = pd.Series(match)

    home_team = match["home_team"]
    away_team = match["away_team"]

    home_features = get_team_features(state, home_team)
    away_features = get_team_features(state, away_team)

    row = {
        "date": match.get("date", None),
        "home_team": home_team,
        "away_team": away_team,
        "tournament": match.get("tournament", ""),
        "city": match.get("city", ""),
        "country": match.get("country", ""),
        "neutral": int(bool(match.get("neutral", False))),
        "home_elo_before": home_features.get("elo", INITIAL_ELO),
        "away_elo_before": away_features.get("elo", INITIAL_ELO),
    }

    row["elo_diff"] = row["home_elo_before"] - row["away_elo_before"]

    for window in (5, 10):
        row[f"home_avg_goals_for_{window}"] = home_features.get(
            f"avg_goals_for_{window}"
        )
        row[f"home_avg_goals_against_{window}"] = home_features.get(
            f"avg_goals_against_{window}"
        )
        row[f"home_avg_points_{window}"] = home_features.get(
            f"avg_points_{window}"
        )

        row[f"away_avg_goals_for_{window}"] = away_features.get(
            f"avg_goals_for_{window}"
        )
        row[f"away_avg_goals_against_{window}"] = away_features.get(
            f"avg_goals_against_{window}"
        )
        row[f"away_avg_points_{window}"] = away_features.get(
            f"avg_points_{window}"
        )

    match_df = pd.DataFrame([row])
    match_df = add_tournament_features(match_df)

    match_df = fill_missing_dynamic_features(match_df)

    return match_df


def fill_missing_dynamic_features(match_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing dynamic features.

    This is a defensive fallback for teams with no state history.
    """
    df = match_df.copy()

    fallback_values = {
        "home_avg_goals_for_5": 1.0,
        "home_avg_goals_against_5": 1.0,
        "home_avg_points_5": 1.0,
        "home_avg_goals_for_10": 1.0,
        "home_avg_goals_against_10": 1.0,
        "home_avg_points_10": 1.0,
        "away_avg_goals_for_5": 1.0,
        "away_avg_goals_against_5": 1.0,
        "away_avg_points_5": 1.0,
        "away_avg_goals_for_10": 1.0,
        "away_avg_goals_against_10": 1.0,
        "away_avg_points_10": 1.0,
        "neutral": 1,
        "elo_diff": 0.0,
        "is_friendly": 0,
        "is_world_cup": 0,
        "is_world_cup_qualifier": 0,
        "is_continental_cup": 0,
        "is_nations_league": 0,
        "is_competitive": 1,
    }

    for col in MODEL_FEATURES:
        if col not in df.columns:
            df[col] = fallback_values.get(col, 0)

        if df[col].isna().any():
            df[col] = df[col].fillna(fallback_values.get(col, 0))

    return df


def predict_match_from_state(
    match: pd.Series | dict,
    state: dict,
    model_bundle: dict,
) -> pd.DataFrame:
    """
    Predict one match from current dynamic team_state.
    """
    match_df = build_match_features_from_state(
        match=match,
        state=state,
    )

    prediction_df = predict_expected_goals(
        model_bundle=model_bundle,
        model_df=match_df,
    )

    prediction_df = add_match_probabilities(prediction_df)
    prediction_df = calibrate_probabilities(
        prediction_df,
        base_probs=model_bundle.get("base_probs"),
    )

    return prediction_df


def simulate_match_from_state(
    match: pd.Series | dict,
    state: dict,
    model_bundle: dict,
    random_state: int | None = None,
    update_state: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """
    Predict and simulate one match from state.

    If update_state=True, the state is updated using the simulated score.
    """
    prediction_df = predict_match_from_state(
        match=match,
        state=state,
        model_bundle=model_bundle,
    )

    row = prediction_df.iloc[0]

    simulated = simulate_match(
        lambda_home=row["lambda_home"],
        lambda_away=row["lambda_away"],
        random_state=random_state,
    )

    prediction_df["simulated_home_goals"] = simulated["home_goals"]
    prediction_df["simulated_away_goals"] = simulated["away_goals"]
    prediction_df["simulated_result"] = simulated["result"]

    if update_state:
        update_team_state_after_match(
            state=state,
            home_team=row["home_team"],
            away_team=row["away_team"],
            home_score=simulated["home_goals"],
            away_score=simulated["away_goals"],
            neutral=bool(row["neutral"]),
        )

    return prediction_df, state


def monte_carlo_match_from_state(
    match: pd.Series | dict,
    state: dict,
    model_bundle: dict,
) -> pd.DataFrame:
    """
    Predict one match from state and add Monte Carlo probabilities.
    """
    prediction_df = predict_match_from_state(
        match=match,
        state=state,
        model_bundle=model_bundle,
    )

    row = prediction_df.iloc[0]

    mc_result = monte_carlo_match_fast(
        lambda_home=row["lambda_home"],
        lambda_away=row["lambda_away"],
    )

    mc_df = pd.DataFrame([mc_result])

    return pd.concat(
        [
            prediction_df.reset_index(drop=True),
            mc_df.reset_index(drop=True),
        ],
        axis=1,
    )


def predict_round_dynamic(
    round_matches: pd.DataFrame,
    state: dict,
    model_bundle: dict,
) -> pd.DataFrame:
    """
    Predict a group of matches using the same initial state.

    This does not update state between matches.
    Useful for predicting all matches in a round before results are known.
    """
    predictions = []

    for _, match in round_matches.iterrows():
        pred = monte_carlo_match_from_state(
            match=match,
            state=state,
            model_bundle=model_bundle,
        )
        predictions.append(pred)

    return pd.concat(predictions, ignore_index=True)


def simulate_round_dynamic(
    round_matches: pd.DataFrame,
    state: dict,
    model_bundle: dict,
    random_state: int = 42,
) -> tuple[pd.DataFrame, dict]:
    """
    Simulate a group of matches and update team_state after each match.

    This is useful for dynamic tournament simulation.
    """
    predictions = []

    for idx, match in round_matches.iterrows():
        pred, state = simulate_match_from_state(
            match=match,
            state=state,
            model_bundle=model_bundle,
            random_state=random_state + idx,
            update_state=True,
        )

        predictions.append(pred)

    predictions_df = pd.concat(predictions, ignore_index=True)

    return predictions_df, state