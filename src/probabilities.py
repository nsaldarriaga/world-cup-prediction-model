import numpy as np
import pandas as pd

from scipy.stats import poisson
from sklearn.metrics import accuracy_score

from src.config import MAX_GOALS, BLENDING_WEIGHT_MODEL


# Fixed internal class order used across the whole project.
# H = home win, D = draw, A = away win
CLASS_LABELS = ["H", "D", "A"]

# Probability columns must always follow the same order as CLASS_LABELS.
RAW_PROB_COLS = [
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
]

BLENDED_PROB_COLS = [
    "prob_home_win_blended",
    "prob_draw_blended",
    "prob_away_win_blended",
]

CALIBRATED_PROB_COLS = [
    "prob_H_calibrated",
    "prob_D_calibrated",
    "prob_A_calibrated",
]

EPS = 1e-15


def match_probabilities(
    lambda_home: float,
    lambda_away: float,
    max_goals: int = MAX_GOALS,
) -> dict:
    """
    Convert expected goals into match result probabilities.

    Output columns follow fixed H/D/A order:
    - prob_home_win
    - prob_draw
    - prob_away_win
    """
    lambda_home = max(float(lambda_home), EPS)
    lambda_away = max(float(lambda_away), EPS)

    goals = np.arange(0, max_goals + 1)

    home_goal_probs = poisson.pmf(goals, lambda_home)
    away_goal_probs = poisson.pmf(goals, lambda_away)

    score_matrix = np.outer(home_goal_probs, away_goal_probs)

    prob_home_win = np.tril(score_matrix, k=-1).sum()
    prob_draw = np.trace(score_matrix)
    prob_away_win = np.triu(score_matrix, k=1).sum()

    probs = np.array(
        [prob_home_win, prob_draw, prob_away_win],
        dtype=float,
    )

    probs = np.clip(probs, EPS, 1 - EPS)
    probs = probs / probs.sum()

    return {
        "prob_home_win": probs[0],
        "prob_draw": probs[1],
        "prob_away_win": probs[2],
    }


def add_match_probabilities(
    df: pd.DataFrame,
    lambda_home_col: str = "lambda_home",
    lambda_away_col: str = "lambda_away",
    max_goals: int = MAX_GOALS,
) -> pd.DataFrame:
    """
    Add Poisson H/D/A probabilities to a dataframe containing expected goals.
    """
    df = df.copy()

    probabilities = df.apply(
        lambda row: match_probabilities(
            lambda_home=row[lambda_home_col],
            lambda_away=row[lambda_away_col],
            max_goals=max_goals,
        ),
        axis=1,
        result_type="expand",
    )

    df = pd.concat([df, probabilities], axis=1)

    return df


def add_actual_result(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add actual result label in fixed H/D/A format.

    H = home win
    D = draw
    A = away win
    """
    df = df.copy()

    conditions = [
        df["home_score"] > df["away_score"],
        df["home_score"] == df["away_score"],
        df["home_score"] < df["away_score"],
    ]

    df["actual_result"] = np.select(
        conditions,
        CLASS_LABELS,
        default="unknown",
    )

    return df


def add_predicted_result(
    df: pd.DataFrame,
    prob_cols: list[str] | None = None,
    output_col: str = "predicted_result",
) -> pd.DataFrame:
    """
    Add predicted H/D/A result based on the highest probability.

    The probability columns must follow the order:
    H, D, A.
    """
    df = df.copy()

    if prob_cols is None:
        prob_cols = RAW_PROB_COLS

    pred_idx = df[prob_cols].values.argmax(axis=1)

    df[output_col] = [CLASS_LABELS[i] for i in pred_idx]

    return df


def get_base_probabilities(df: pd.DataFrame) -> dict:
    """
    Calculate base probabilities from historical result frequencies.

    Returns base probabilities in H/D/A order.
    Used only as a benchmark or simple blending baseline.
    """
    df = add_actual_result(df)

    base_probs = (
        df["actual_result"]
        .value_counts(normalize=True)
        .reindex(CLASS_LABELS)
        .fillna(0)
        .to_dict()
    )

    return {
        "prob_home_win_base": base_probs["H"],
        "prob_draw_base": base_probs["D"],
        "prob_away_win_base": base_probs["A"],
    }


def calibrate_probabilities(
    df: pd.DataFrame,
    weight_model: float = BLENDING_WEIGHT_MODEL,
    base_probs: dict | None = None,
) -> pd.DataFrame:
    """
    Calibrate probabilities using simple blending:

    prob_blended = weight_model * prob_model + (1 - weight_model) * prob_base

    Note:
    This is now kept as a baseline benchmark. The preferred final probability
    model is the multinomial calibrator when available.
    """
    df = df.copy()

    if base_probs is None:
        base_probs = get_base_probabilities(df)

    df["prob_home_win_blended"] = (
        weight_model * df["prob_home_win"]
        + (1 - weight_model) * base_probs["prob_home_win_base"]
    )

    df["prob_draw_blended"] = (
        weight_model * df["prob_draw"]
        + (1 - weight_model) * base_probs["prob_draw_base"]
    )

    df["prob_away_win_blended"] = (
        weight_model * df["prob_away_win"]
        + (1 - weight_model) * base_probs["prob_away_win_base"]
    )

    df = normalize_probability_columns(df, BLENDED_PROB_COLS)

    return df


def normalize_probability_columns(
    df: pd.DataFrame,
    prob_cols: list[str],
) -> pd.DataFrame:
    """
    Normalize probability columns row-wise.
    """
    df = df.copy()

    df[prob_cols] = df[prob_cols].clip(lower=EPS, upper=1 - EPS)

    row_sums = df[prob_cols].sum(axis=1)
    df[prob_cols] = df[prob_cols].div(row_sums, axis=0)

    return df


def probability_matrix(
    df: pd.DataFrame,
    prob_cols: list[str],
) -> np.ndarray:
    """
    Return probability matrix in fixed H/D/A order.

    The prob_cols argument must already be in H/D/A order.
    """
    return df[prob_cols].to_numpy(dtype=float)


def custom_log_loss_hda(
    y_true: pd.Series | np.ndarray,
    prob_matrix: np.ndarray,
) -> float:
    """
    Compute log loss using explicit H/D/A probability order.

    This avoids sklearn column-order issues when labels are strings.

    Expected probability matrix order:
    column 0 = H
    column 1 = D
    column 2 = A
    """
    y_true_array = np.array(y_true)

    class_to_col = {
        "H": 0,
        "D": 1,
        "A": 2,
    }

    true_cols = np.array([class_to_col[y] for y in y_true_array])

    assigned_probs = prob_matrix[np.arange(len(y_true_array)), true_cols]
    assigned_probs = np.clip(assigned_probs, EPS, 1 - EPS)

    return -np.mean(np.log(assigned_probs))


def log_loss_by_class(
    df: pd.DataFrame,
    prob_cols: list[str],
    target_col: str = "actual_result",
) -> pd.DataFrame:
    """
    Calculate log loss by class using explicit H/D/A order.
    """
    if target_col not in df.columns:
        df = add_actual_result(df)

    y_true = np.array(df[target_col])
    prob_matrix = probability_matrix(df, prob_cols)

    rows = []

    for idx, label in enumerate(CLASS_LABELS):
        mask = y_true == label

        if mask.sum() == 0:
            class_log_loss = np.nan
            avg_assigned_prob = np.nan
        else:
            assigned_probs = prob_matrix[mask, idx]
            assigned_probs = np.clip(assigned_probs, EPS, 1 - EPS)

            class_log_loss = -np.mean(np.log(assigned_probs))
            avg_assigned_prob = assigned_probs.mean()

        rows.append({
            "class": label,
            "n": int(mask.sum()),
            "class_log_loss": class_log_loss,
            "avg_assigned_prob": avg_assigned_prob,
        })

    return pd.DataFrame(rows)


def evaluate_predictions(
    df: pd.DataFrame,
    prob_cols: list[str],
    prediction_col: str = "predicted_result",
) -> dict:
    """
    Evaluate H/D/A predictions using accuracy and custom log loss.

    Important:
    prob_cols must follow fixed H/D/A order.
    """
    df = df.copy()

    if "actual_result" not in df.columns:
        df = add_actual_result(df)

    if prediction_col not in df.columns:
        df = add_predicted_result(
            df,
            prob_cols=prob_cols,
            output_col=prediction_col,
        )

    y_true = df["actual_result"]
    y_pred = df[prediction_col]

    prob_matrix = probability_matrix(df, prob_cols)

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "log_loss": custom_log_loss_hda(y_true, prob_matrix),
        "draw_real_rate": np.mean(np.array(y_true) == "D"),
        "draw_pred_rate": np.mean(np.array(y_pred) == "D"),
    }


def evaluate_raw_and_blended_predictions(df: pd.DataFrame) -> dict:
    """
    Evaluate both raw Poisson probabilities and blended probabilities.
    """
    df = df.copy()

    df = add_actual_result(df)

    df = add_predicted_result(
        df,
        prob_cols=RAW_PROB_COLS,
        output_col="predicted_result_raw",
    )

    df = add_predicted_result(
        df,
        prob_cols=BLENDED_PROB_COLS,
        output_col="predicted_result_blended",
    )

    raw_metrics = evaluate_predictions(
        df,
        prob_cols=RAW_PROB_COLS,
        prediction_col="predicted_result_raw",
    )

    blended_metrics = evaluate_predictions(
        df,
        prob_cols=BLENDED_PROB_COLS,
        prediction_col="predicted_result_blended",
    )

    return {
        "raw": raw_metrics,
        "blended": blended_metrics,
    }


def add_final_prediction_columns(
    df: pd.DataFrame,
    prob_cols: list[str],
    prefix: str = "final",
) -> pd.DataFrame:
    """
    Add final prediction result and confidence from selected probability columns.

    Example:
    prob_cols = ["prob_H_calibrated", "prob_D_calibrated", "prob_A_calibrated"]

    Output:
    - predicted_result_final
    - confidence_final
    """
    df = df.copy()

    pred_idx = df[prob_cols].values.argmax(axis=1)

    df[f"predicted_result_{prefix}"] = [
        CLASS_LABELS[i] for i in pred_idx
    ]

    df[f"confidence_{prefix}"] = df[prob_cols].max(axis=1)

    return df