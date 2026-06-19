import numpy as np
import pandas as pd

from src.config import INITIAL_ELO, OUTPUTS_DIR

from src.features import (
    add_tournament_features,
    build_latest_team_form_features,
)

from src.elo import calculate_current_elo_ratings

from src.model import (
    MODEL_FEATURES,
    CALIBRATOR_CONTEXT_FEATURES,
    predict_expected_goals,
)

from src.probabilities import (
    add_match_probabilities,
    calibrate_probabilities,
)

from src.simulation import monte_carlo_dataframe_fast


def build_calendar_features(
    future_matches: pd.DataFrame,
    played_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build feature dataset for future matches.

    Uses:
    - latest rolling form from played matches
    - latest Elo ratings from played matches
    - tournament features from future matches

    The output must be compatible with MODEL_FEATURES and with the
    multinomial calibrator context features.
    """
    calendar_df = future_matches.copy()
    calendar_df = calendar_df.sort_values("date").reset_index(drop=True)
    calendar_df["match_id"] = range(len(calendar_df))

    # 1. Tournament features
    calendar_df = add_tournament_features(calendar_df)

    # 2. Latest team form
    latest_form = build_latest_team_form_features(played_matches)

    home_form = latest_form.rename(
        columns={
            "team": "home_team",
            "avg_goals_for_5": "home_avg_goals_for_5",
            "avg_goals_against_5": "home_avg_goals_against_5",
            "avg_points_5": "home_avg_points_5",
            "avg_goals_for_10": "home_avg_goals_for_10",
            "avg_goals_against_10": "home_avg_goals_against_10",
            "avg_points_10": "home_avg_points_10",
        }
    )

    away_form = latest_form.rename(
        columns={
            "team": "away_team",
            "avg_goals_for_5": "away_avg_goals_for_5",
            "avg_goals_against_5": "away_avg_goals_against_5",
            "avg_points_5": "away_avg_points_5",
            "avg_goals_for_10": "away_avg_goals_for_10",
            "avg_goals_against_10": "away_avg_goals_against_10",
            "avg_points_10": "away_avg_points_10",
        }
    )

    calendar_df = calendar_df.merge(home_form, on="home_team", how="left")
    calendar_df = calendar_df.merge(away_form, on="away_team", how="left")

    # 3. Current Elo
    ratings = calculate_current_elo_ratings(played_matches)

    calendar_df["home_elo_before"] = (
        calendar_df["home_team"].map(ratings).fillna(INITIAL_ELO)
    )

    calendar_df["away_elo_before"] = (
        calendar_df["away_team"].map(ratings).fillna(INITIAL_ELO)
    )

    calendar_df["elo_diff"] = (
        calendar_df["home_elo_before"] - calendar_df["away_elo_before"]
    )

    # 4. Neutral as int
    calendar_df["neutral"] = calendar_df["neutral"].astype(int)

    # 5. New model features
    calendar_df = add_calendar_difference_features(calendar_df)
    calendar_df = add_calendar_elo_transformations(calendar_df)
    calendar_df = add_calendar_parity_features(calendar_df)

    # 6. Fill missing values and validate final feature set
    calendar_df = fill_missing_calendar_features(calendar_df, played_matches)

    return calendar_df


def add_calendar_difference_features(
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add home-away difference features required by the Poisson model.
    """
    df = calendar_df.copy()

    df["diff_attack_5"] = (
        df["home_avg_goals_for_5"] - df["away_avg_goals_for_5"]
    )

    df["diff_defense_5"] = (
        df["home_avg_goals_against_5"] - df["away_avg_goals_against_5"]
    )

    df["diff_points_5"] = (
        df["home_avg_points_5"] - df["away_avg_points_5"]
    )

    df["diff_attack_10"] = (
        df["home_avg_goals_for_10"] - df["away_avg_goals_for_10"]
    )

    df["diff_defense_10"] = (
        df["home_avg_goals_against_10"] - df["away_avg_goals_against_10"]
    )

    df["diff_points_10"] = (
        df["home_avg_points_10"] - df["away_avg_points_10"]
    )

    return df


def add_calendar_elo_transformations(
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add transformed Elo features required by the model and calibrator.
    """
    df = calendar_df.copy()

    df["elo_tanh_400"] = np.tanh(df["elo_diff"] / 400)
    df["abs_elo_tanh_400"] = np.abs(df["elo_tanh_400"])
    df["abs_elo_diff"] = np.abs(df["elo_diff"])

    return df


def add_calendar_parity_features(
    calendar_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add absolute-difference features used by the multinomial calibrator.
    """
    df = calendar_df.copy()

    df["abs_diff_attack_5"] = np.abs(df["diff_attack_5"])
    df["abs_diff_defense_5"] = np.abs(df["diff_defense_5"])
    df["abs_diff_points_5"] = np.abs(df["diff_points_5"])

    df["abs_diff_attack_10"] = np.abs(df["diff_attack_10"])
    df["abs_diff_defense_10"] = np.abs(df["diff_defense_10"])
    df["abs_diff_points_10"] = np.abs(df["diff_points_10"])

    return df


def fill_missing_calendar_features(
    calendar_df: pd.DataFrame,
    played_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Fill missing feature values for teams without enough history.

    This validates both MODEL_FEATURES and CALIBRATOR_CONTEXT_FEATURES,
    because future predictions need to work for both Poisson and the
    multinomial calibrator.
    """
    df = calendar_df.copy()

    latest_form = build_latest_team_form_features(played_matches)

    fallback_values = {
        "avg_goals_for_5": latest_form["avg_goals_for_5"].median(),
        "avg_goals_against_5": latest_form["avg_goals_against_5"].median(),
        "avg_points_5": latest_form["avg_points_5"].median(),
        "avg_goals_for_10": latest_form["avg_goals_for_10"].median(),
        "avg_goals_against_10": latest_form["avg_goals_against_10"].median(),
        "avg_points_10": latest_form["avg_points_10"].median(),
    }

    required_features = list(dict.fromkeys(
        MODEL_FEATURES + CALIBRATOR_CONTEXT_FEATURES
    ))

    for col in required_features:
        if col not in df.columns:
            raise ValueError(f"Missing feature in calendar dataset: {col}")

        if df[col].isna().any():
            base_col = (
                col.replace("home_", "")
                .replace("away_", "")
            )

            fill_value = fallback_values.get(base_col, df[col].median())
            df[col] = df[col].fillna(fill_value)

    return df


def predict_calendar(
    future_matches: pd.DataFrame,
    played_matches: pd.DataFrame,
    model_bundle: dict,
) -> pd.DataFrame:
    """
    Predict expected goals and probabilities for future matches.

    Outputs:
    - lambda_home / lambda_away
    - raw Poisson probabilities
    - blended probabilities as benchmark
    """
    calendar_df = build_calendar_features(
        future_matches=future_matches,
        played_matches=played_matches,
    )

    predictions_df = predict_expected_goals(
        model_bundle=model_bundle,
        model_df=calendar_df,
    )

    predictions_df = add_match_probabilities(predictions_df)

    predictions_df = calibrate_probabilities(
        predictions_df,
        base_probs=model_bundle.get("base_probs"),
    )

    return predictions_df


def monte_carlo_calendar_fast(
    calendar_predictions: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add Monte Carlo simulation results to calendar predictions.
    """
    return monte_carlo_dataframe_fast(calendar_predictions)


def save_calendar_predictions(
    predictions_df: pd.DataFrame,
    filename: str = "calendar_predictions.csv",
) -> None:
    """
    Save calendar predictions to outputs folder.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUTS_DIR / filename

    predictions_df.to_csv(output_path, index=False)

    print(f"Saved predictions to: {output_path}")


def get_next_pending_matches(
    future_matches: pd.DataFrame,
    date_window_days: int = 0,
) -> pd.DataFrame:
    """
    Get the next pending matches from the future calendar.

    By default, it selects matches from the earliest pending date.
    """
    if future_matches.empty:
        return future_matches.copy()

    df = future_matches.copy()
    df = df.sort_values("date").reset_index(drop=True)

    next_date = df["date"].min()
    max_date = next_date + pd.Timedelta(days=date_window_days)

    next_matches = df[
        (df["date"] >= next_date)
        & (df["date"] <= max_date)
    ].copy()

    return next_matches.reset_index(drop=True)


def save_next_round_predictions(
    predictions_df: pd.DataFrame,
    filename: str = "next_round_predictions.csv",
) -> None:
    """
    Save next round predictions to outputs folder.
    """
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    output_path = OUTPUTS_DIR / filename

    predictions_df.to_csv(output_path, index=False)

    print(f"Saved next round predictions to: {output_path}")