import pandas as pd

from sklearn.metrics import mean_absolute_error

from src.config import TRAIN_END_DATE, TEST_START_DATE

from src.model import (
    MODEL_FEATURES,
    train_poisson_models,
    predict_expected_goals,
    build_calibrator_features,
    train_multinomial_calibrator,
    predict_calibrated_probabilities,
)

from src.probabilities import (
    RAW_PROB_COLS,
    BLENDED_PROB_COLS,
    CALIBRATED_PROB_COLS,
    add_actual_result,
    add_match_probabilities,
    add_predicted_result,
    calibrate_probabilities,
    evaluate_predictions,
    evaluate_raw_and_blended_predictions,
)


def temporal_train_test_split(
    model_df: pd.DataFrame,
    train_end_date: str = TRAIN_END_DATE,
    test_start_date: str = TEST_START_DATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split model dataset using dates.

    Train:
        date <= train_end_date

    Test:
        date >= test_start_date
    """
    df = model_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    train_end_date = pd.to_datetime(train_end_date)
    test_start_date = pd.to_datetime(test_start_date)

    train_df = df[df["date"] <= train_end_date].copy()
    test_df = df[df["date"] >= test_start_date].copy()

    if train_df.empty:
        raise ValueError("Train dataset is empty. Check TRAIN_END_DATE.")

    if test_df.empty:
        raise ValueError("Test dataset is empty. Check TEST_START_DATE.")

    return train_df.reset_index(drop=True), test_df.reset_index(drop=True)


def add_predictions_and_probabilities(
    model_bundle: dict,
    df: pd.DataFrame,
    add_blended: bool = True,
) -> pd.DataFrame:
    """
    Add expected goals and raw Poisson result probabilities.

    Optionally adds blended probabilities as a benchmark.
    """
    predictions_df = predict_expected_goals(
        model_bundle=model_bundle,
        model_df=df,
    )

    predictions_df = add_match_probabilities(predictions_df)

    if add_blended:
        predictions_df = calibrate_probabilities(
            predictions_df,
            base_probs=model_bundle.get("base_probs"),
        )

    return predictions_df


def evaluate_goal_mae(predictions_df: pd.DataFrame) -> dict:
    """
    Evaluate expected goals using MAE.

    MAE compares:
    - home_score vs lambda_home
    - away_score vs lambda_away
    """
    home_mae = mean_absolute_error(
        predictions_df["home_score"],
        predictions_df["lambda_home"],
    )

    away_mae = mean_absolute_error(
        predictions_df["away_score"],
        predictions_df["lambda_away"],
    )

    overall_mae = mean_absolute_error(
        pd.concat(
            [
                predictions_df["home_score"],
                predictions_df["away_score"],
            ],
            ignore_index=True,
        ),
        pd.concat(
            [
                predictions_df["lambda_home"],
                predictions_df["lambda_away"],
            ],
            ignore_index=True,
        ),
    )

    return {
        "home_mae": home_mae,
        "away_mae": away_mae,
        "overall_mae": overall_mae,
    }


def make_temporal_oof_calibrator_data(
    train_df: pd.DataFrame,
    n_folds: int = 4,
    min_train_ratio: float = 0.50,
) -> tuple[pd.DataFrame, pd.Series, pd.DataFrame]:
    """
    Generate temporal out-of-fold predictions inside the training set.

    This avoids training the multinomial calibrator with in-sample Poisson
    predictions.

    For each fold:
    - train Poisson on an older block
    - predict a later validation block
    - build calibrator features from those predictions

    Returns:
    - X_calibrator_oof
    - y_oof
    - oof_debug_df
    """
    df = train_df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    df = add_actual_result(df)

    n = len(df)
    min_train_size = int(n * min_train_ratio)

    if n_folds <= 0:
        raise ValueError("n_folds must be greater than zero.")

    if min_train_size >= n:
        raise ValueError("min_train_ratio leaves no validation data.")

    fold_size = int((n - min_train_size) / n_folds)

    if fold_size <= 0:
        raise ValueError("Fold size is zero. Reduce n_folds or min_train_ratio.")

    oof_parts = []

    for fold in range(n_folds):
        train_end = min_train_size + fold * fold_size

        if fold == n_folds - 1:
            valid_end = n
        else:
            valid_end = train_end + fold_size

        inner_train_df = df.iloc[:train_end].copy()
        inner_valid_df = df.iloc[train_end:valid_end].copy()

        if inner_train_df.empty or inner_valid_df.empty:
            continue

        inner_model_bundle = train_poisson_models(inner_train_df)

        inner_valid_predictions = add_predictions_and_probabilities(
            model_bundle=inner_model_bundle,
            df=inner_valid_df,
            add_blended=False,
        )

        X_valid_calibrator = build_calibrator_features(inner_valid_predictions)

        fold_oof_df = X_valid_calibrator.copy()
        fold_oof_df["actual_result"] = inner_valid_predictions["actual_result"].values
        fold_oof_df["date"] = inner_valid_predictions["date"].values
        fold_oof_df["fold"] = fold

        oof_parts.append(fold_oof_df)

    if not oof_parts:
        raise ValueError("No OOF calibrator data was generated.")

    oof_debug_df = pd.concat(oof_parts, ignore_index=True)

    y_oof = oof_debug_df["actual_result"].copy()

    X_oof = oof_debug_df.drop(
        columns=["actual_result", "date", "fold"],
    )

    return X_oof, y_oof, oof_debug_df


def train_oof_multinomial_calibrator(
    train_df: pd.DataFrame,
    n_folds: int = 4,
    min_train_ratio: float = 0.50,
) -> tuple[object, pd.DataFrame]:
    """
    Train a multinomial calibrator using temporal OOF Poisson predictions.
    """
    X_oof, y_oof, oof_debug_df = make_temporal_oof_calibrator_data(
        train_df=train_df,
        n_folds=n_folds,
        min_train_ratio=min_train_ratio,
    )

    calibrator = train_multinomial_calibrator(
        X_calibrator=X_oof,
        y_result=y_oof,
    )

    return calibrator, oof_debug_df


def add_multinomial_calibrated_probabilities(
    predictions_df: pd.DataFrame,
    calibrator: object,
) -> pd.DataFrame:
    """
    Add multinomial calibrated probabilities to a predictions dataframe.

    Output columns:
    - prob_H_calibrated
    - prob_D_calibrated
    - prob_A_calibrated
    - predicted_result_calibrated
    - confidence_calibrated
    """
    df = predictions_df.copy()

    X_calibrator = build_calibrator_features(df)

    calibrated_probs = predict_calibrated_probabilities(
        calibrator=calibrator,
        X_calibrator=X_calibrator,
    )

    df = pd.concat([df, calibrated_probs], axis=1)

    df = add_predicted_result(
        df,
        prob_cols=CALIBRATED_PROB_COLS,
        output_col="predicted_result_calibrated",
    )

    df["confidence_calibrated"] = df[CALIBRATED_PROB_COLS].max(axis=1)

    return df


def evaluate_all_probability_models(
    predictions_df: pd.DataFrame,
) -> dict:
    """
    Evaluate raw Poisson, blended baseline and multinomial calibrated probabilities.
    """
    df = predictions_df.copy()
    df = add_actual_result(df)

    # Raw prediction
    df = add_predicted_result(
        df,
        prob_cols=RAW_PROB_COLS,
        output_col="predicted_result_raw",
    )

    raw_metrics = evaluate_predictions(
        df,
        prob_cols=RAW_PROB_COLS,
        prediction_col="predicted_result_raw",
    )

    metrics = {
        "raw": raw_metrics,
    }

    # Blended benchmark, if available
    if all(col in df.columns for col in BLENDED_PROB_COLS):
        df = add_predicted_result(
            df,
            prob_cols=BLENDED_PROB_COLS,
            output_col="predicted_result_blended",
        )

        blended_metrics = evaluate_predictions(
            df,
            prob_cols=BLENDED_PROB_COLS,
            prediction_col="predicted_result_blended",
        )

        metrics["blended"] = blended_metrics

    # Multinomial calibrator, if available
    if all(col in df.columns for col in CALIBRATED_PROB_COLS):
        if "predicted_result_calibrated" not in df.columns:
            df = add_predicted_result(
                df,
                prob_cols=CALIBRATED_PROB_COLS,
                output_col="predicted_result_calibrated",
            )

        calibrated_metrics = evaluate_predictions(
            df,
            prob_cols=CALIBRATED_PROB_COLS,
            prediction_col="predicted_result_calibrated",
        )

        metrics["oof_multinomial_calibrator"] = calibrated_metrics

    return metrics


def evaluate_model_temporal_split(
    model_df: pd.DataFrame,
    train_end_date: str = TRAIN_END_DATE,
    test_start_date: str = TEST_START_DATE,
    train_calibrator: bool = True,
    n_oof_folds: int = 4,
    min_train_ratio: float = 0.50,
) -> dict:
    """
    Train on historical period and evaluate on a later period.

    Evaluation includes:
    - raw Poisson probabilities
    - blended baseline probabilities
    - OOF multinomial calibrated probabilities, if train_calibrator=True
    """
    train_df, test_df = temporal_train_test_split(
        model_df=model_df,
        train_end_date=train_end_date,
        test_start_date=test_start_date,
    )

    train_df = add_actual_result(train_df)
    test_df = add_actual_result(test_df)

    model_bundle = train_poisson_models(train_df)

    calibrator = None
    oof_debug_df = None

    if train_calibrator:
        calibrator, oof_debug_df = train_oof_multinomial_calibrator(
            train_df=train_df,
            n_folds=n_oof_folds,
            min_train_ratio=min_train_ratio,
        )

        model_bundle["calibrator"] = calibrator

    train_predictions = add_predictions_and_probabilities(
        model_bundle=model_bundle,
        df=train_df,
        add_blended=True,
    )

    test_predictions = add_predictions_and_probabilities(
        model_bundle=model_bundle,
        df=test_df,
        add_blended=True,
    )

    if calibrator is not None:
        train_predictions = add_multinomial_calibrated_probabilities(
            predictions_df=train_predictions,
            calibrator=calibrator,
        )

        test_predictions = add_multinomial_calibrated_probabilities(
            predictions_df=test_predictions,
            calibrator=calibrator,
        )

    train_result_metrics = evaluate_all_probability_models(train_predictions)
    test_result_metrics = evaluate_all_probability_models(test_predictions)

    train_mae = evaluate_goal_mae(train_predictions)
    test_mae = evaluate_goal_mae(test_predictions)

    return {
        "model_bundle": model_bundle,
        "calibrator": calibrator,
        "oof_debug_df": oof_debug_df,
        "train_df": train_df,
        "test_df": test_df,
        "train_predictions": train_predictions,
        "test_predictions": test_predictions,
        "train_result_metrics": train_result_metrics,
        "test_result_metrics": test_result_metrics,
        "train_mae": train_mae,
        "test_mae": test_mae,
    }