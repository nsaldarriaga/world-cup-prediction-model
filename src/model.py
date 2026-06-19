import numpy as np
import pandas as pd

from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import PoissonRegressor, LogisticRegression

from src.config import (
    TRAIN_START_DATE,
    POISSON_ALPHA,
    POISSON_MAX_ITER,
)

from src.features import (
    create_matches_long,
    add_rolling_features,
    add_tournament_features,
)

from src.elo import build_elo_features
from src.probabilities import get_base_probabilities


ROLLING_FEATURES = [
    "avg_goals_for_5",
    "avg_goals_against_5",
    "avg_points_5",
    "avg_goals_for_10",
    "avg_goals_against_10",
    "avg_points_10",
]

TOURNAMENT_FEATURES = [
    "is_friendly",
    "is_world_cup",
    "is_world_cup_qualifier",
    "is_continental_cup",
    "is_nations_league",
    "is_competitive",
]

DIFF_FEATURES = [
    "diff_attack_5",
    "diff_defense_5",
    "diff_points_5",
    "diff_attack_10",
    "diff_defense_10",
    "diff_points_10",
]

PARITY_FEATURES = [
    "abs_elo_tanh_400",
    "abs_elo_diff",
    "abs_diff_attack_5",
    "abs_diff_defense_5",
    "abs_diff_points_5",
    "abs_diff_attack_10",
    "abs_diff_defense_10",
    "abs_diff_points_10",
]

MODEL_FEATURES = [
    "home_avg_goals_for_5",
    "home_avg_goals_against_5",
    "home_avg_points_5",
    "home_avg_goals_for_10",
    "home_avg_goals_against_10",
    "home_avg_points_10",

    "away_avg_goals_for_5",
    "away_avg_goals_against_5",
    "away_avg_points_5",
    "away_avg_goals_for_10",
    "away_avg_goals_against_10",
    "away_avg_points_10",

    *DIFF_FEATURES,

    "neutral",
    "elo_tanh_400",

    *TOURNAMENT_FEATURES,
]

CALIBRATOR_CONTEXT_FEATURES = [
    "elo_tanh_400",
    "abs_elo_tanh_400",
    "abs_elo_diff",
    "neutral",

    "is_friendly",
    "is_world_cup",
    "is_world_cup_qualifier",
    "is_competitive",

    "diff_points_5",
    "diff_points_10",
    "abs_diff_points_5",
    "abs_diff_points_10",

    "diff_attack_10",
    "diff_defense_10",
    "abs_diff_attack_10",
    "abs_diff_defense_10",
]


def build_model_dataset(
    played_matches: pd.DataFrame,
    train_start_date: str = TRAIN_START_DATE,
) -> pd.DataFrame:
    """
    Build final model dataset with one row per match.

    Includes:
    - rolling form features for home and away teams
    - attack/defense/points differences
    - tournament features
    - Elo features
    - targets: home_score, away_score
    """
    df = played_matches.copy()
    df = df.sort_values("date").reset_index(drop=True)
    df["match_id"] = range(len(df))

    # 1. Elo features
    df = build_elo_features(df)
    df = add_elo_transformations(df)

    # 2. Tournament features
    df = add_tournament_features(df)

    # 3. Rolling features in long format
    matches_long = create_matches_long(df)
    matches_long = add_rolling_features(matches_long)

    # 4. Home rolling features
    home_features = (
        matches_long[matches_long["is_home"]]
        [["match_id", *ROLLING_FEATURES]]
        .copy()
    )

    home_features = home_features.rename(
        columns={col: f"home_{col}" for col in ROLLING_FEATURES}
    )

    # 5. Away rolling features
    away_features = (
        matches_long[~matches_long["is_home"]]
        [["match_id", *ROLLING_FEATURES]]
        .copy()
    )

    away_features = away_features.rename(
        columns={col: f"away_{col}" for col in ROLLING_FEATURES}
    )

    # 6. Merge back to match-level dataset
    model_df = df.merge(home_features, on="match_id", how="left")
    model_df = model_df.merge(away_features, on="match_id", how="left")

    # 7. Convert bool neutral to int
    model_df["neutral"] = model_df["neutral"].astype(int)

    # 8. Difference and parity features
    model_df = add_difference_features(model_df)
    model_df = add_parity_features(model_df)

    # 9. Target result
    model_df = add_actual_result(model_df)

    # 10. Filter training period after calculating historical features
    model_df = model_df[
        model_df["date"] >= pd.to_datetime(train_start_date)
    ].reset_index(drop=True)

    # 11. Fill early rolling NaNs
    model_df = fill_missing_feature_values(model_df)

    return model_df


def add_actual_result(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add H/D/A target.

    H = home win
    D = draw
    A = away win
    """
    df = model_df.copy()

    df["actual_result"] = np.where(
        df["home_score"] > df["away_score"], "H",
        np.where(df["home_score"] == df["away_score"], "D", "A")
    )

    return df


def add_elo_transformations(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add transformed Elo features.

    elo_tanh_400 is used instead of raw elo_diff to reduce extreme effects.
    """
    df = model_df.copy()

    df["elo_tanh_400"] = np.tanh(df["elo_diff"] / 400)
    df["abs_elo_tanh_400"] = np.abs(df["elo_tanh_400"])
    df["abs_elo_diff"] = np.abs(df["elo_diff"])

    return df


def add_difference_features(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add home-away differences for attack, defense and recent points.
    """
    df = model_df.copy()

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


def add_parity_features(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Add absolute-difference features.

    These help the calibrator detect balanced matches, which are more likely
    to end in draws.
    """
    df = model_df.copy()

    df["abs_diff_attack_5"] = np.abs(df["diff_attack_5"])
    df["abs_diff_defense_5"] = np.abs(df["diff_defense_5"])
    df["abs_diff_points_5"] = np.abs(df["diff_points_5"])

    df["abs_diff_attack_10"] = np.abs(df["diff_attack_10"])
    df["abs_diff_defense_10"] = np.abs(df["diff_defense_10"])
    df["abs_diff_points_10"] = np.abs(df["diff_points_10"])

    return df


def fill_missing_feature_values(model_df: pd.DataFrame) -> pd.DataFrame:
    """
    Fill missing features generated by teams with little history.

    For this educational project we use column medians.
    Later this can be improved using explicit priors.
    """
    df = model_df.copy()

    cols_to_fill = list(dict.fromkeys(
        MODEL_FEATURES + CALIBRATOR_CONTEXT_FEATURES
    ))

    for col in cols_to_fill:
        if col in df.columns and df[col].isna().any():
            df[col] = df[col].fillna(df[col].median())

    return df


def train_poisson_models(
    model_df: pd.DataFrame,
    feature_cols: list[str] = MODEL_FEATURES,
    alpha: float = POISSON_ALPHA,
    max_iter: int = POISSON_MAX_ITER,
) -> dict:
    """
    Train two Poisson regression models:
    - one for home goals
    - one for away goals

    The Poisson models are the source of expected goals:
    - lambda_home
    - lambda_away
    """
    X = model_df[feature_cols]
    y_home = model_df["home_score"]
    y_away = model_df["away_score"]

    home_model = Pipeline([
        ("scaler", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=alpha, max_iter=max_iter)),
    ])

    away_model = Pipeline([
        ("scaler", StandardScaler()),
        ("poisson", PoissonRegressor(alpha=alpha, max_iter=max_iter)),
    ])

    home_model.fit(X, y_home)
    away_model.fit(X, y_away)

    base_probs = get_base_probabilities(model_df)

    return {
        "home_model": home_model,
        "away_model": away_model,
        "feature_cols": feature_cols,
        "base_probs": base_probs,
        "alpha": alpha,
        "max_iter": max_iter,
    }


def predict_expected_goals(
    model_bundle: dict,
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Add expected goals predictions to a model dataset.

    lambda_home and lambda_away are the model's expected goals.
    lambda_home_int and lambda_away_int are rounded versions for easier interpretation.
    """
    df = model_df.copy()

    feature_cols = model_bundle["feature_cols"]
    X = df[feature_cols]

    df["lambda_home"] = model_bundle["home_model"].predict(X)
    df["lambda_away"] = model_bundle["away_model"].predict(X)

    df["lambda_home"] = df["lambda_home"].clip(lower=1e-15, upper=10)
    df["lambda_away"] = df["lambda_away"].clip(lower=1e-15, upper=10)

    df["lambda_home_int"] = df["lambda_home"].round().astype(int)
    df["lambda_away_int"] = df["lambda_away"].round().astype(int)

    return df


def build_calibrator_features(
    model_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build features for the multinomial probability calibrator.

    Required columns:
    - prob_home_win
    - prob_draw
    - prob_away_win
    - lambda_home
    - lambda_away

    The calibrator learns to adjust Poisson probabilities into final H/D/A
    probabilities.
    """
    required_cols = [
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "lambda_home",
        "lambda_away",
    ]

    missing = [col for col in required_cols if col not in model_df.columns]
    if missing:
        raise ValueError(
            f"Missing columns for calibrator features: {missing}"
        )

    df = model_df.copy()

    cal_df = pd.DataFrame(index=df.index)

    cal_df["prob_H_poisson"] = df["prob_home_win"]
    cal_df["prob_D_poisson"] = df["prob_draw"]
    cal_df["prob_A_poisson"] = df["prob_away_win"]

    cal_df["lambda_home"] = df["lambda_home"]
    cal_df["lambda_away"] = df["lambda_away"]
    cal_df["lambda_diff"] = df["lambda_home"] - df["lambda_away"]
    cal_df["abs_lambda_diff"] = np.abs(
        df["lambda_home"] - df["lambda_away"]
    )
    cal_df["lambda_sum"] = df["lambda_home"] + df["lambda_away"]

    for col in CALIBRATOR_CONTEXT_FEATURES:
        if col in df.columns:
            cal_df[col] = df[col].values

    return cal_df


def train_multinomial_calibrator(
    X_calibrator: pd.DataFrame,
    y_result: pd.Series,
    C: float = 1.0,
    max_iter: int = 1000,
) -> Pipeline:
    """
    Train multinomial logistic regression calibrator.

    This model receives Poisson probabilities, lambdas and context features,
    and outputs calibrated H/D/A probabilities.
    """
    calibrator = Pipeline([
        ("scaler", StandardScaler()),
        ("logreg", LogisticRegression(
            ##multi_class="multinomial",
            solver="lbfgs",
            C=C,
            max_iter=max_iter,
            class_weight=None,
        )),
    ])

    calibrator.fit(X_calibrator, y_result)

    return calibrator


def predict_calibrated_probabilities(
    calibrator: Pipeline,
    X_calibrator: pd.DataFrame,
) -> pd.DataFrame:
    """
    Predict calibrated probabilities in fixed order:
    - H
    - D
    - A
    """
    prob_original = calibrator.predict_proba(X_calibrator)

    classes = calibrator.named_steps["logreg"].classes_
    class_to_index = {cls: idx for idx, cls in enumerate(classes)}

    prob_h = prob_original[:, class_to_index["H"]]
    prob_d = prob_original[:, class_to_index["D"]]
    prob_a = prob_original[:, class_to_index["A"]]

    prob_df = pd.DataFrame({
        "prob_H_calibrated": prob_h,
        "prob_D_calibrated": prob_d,
        "prob_A_calibrated": prob_a,
    }, index=X_calibrator.index)

    row_sums = prob_df.sum(axis=1)

    prob_df["prob_H_calibrated"] = prob_df["prob_H_calibrated"] / row_sums
    prob_df["prob_D_calibrated"] = prob_df["prob_D_calibrated"] / row_sums
    prob_df["prob_A_calibrated"] = prob_df["prob_A_calibrated"] / row_sums

    return prob_df