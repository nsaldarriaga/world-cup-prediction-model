from src.data_loader import load_and_cache_results
from src.cleaning import (
    clean_results,
    split_played_and_future_matches,
)

from src.config import (
    CALIBRATOR_OOF_FOLDS,
    CALIBRATOR_MIN_TRAIN_RATIO,
)

from src.model import (
    build_model_dataset,
    MODEL_FEATURES,
)

from src.evaluation import (
    evaluate_model_temporal_split,
)

from src.probabilities import (
    RAW_PROB_COLS,
    BLENDED_PROB_COLS,
    CALIBRATED_PROB_COLS,
)


def print_metrics(metrics: dict, title: str) -> None:
    """
    Print probability metrics in a readable format.
    """
    print(f"\n{title}")
    print("-" * len(title))

    for model_name, model_metrics in metrics.items():
        print(f"\n{model_name}")
        print(f"Accuracy:        {model_metrics['accuracy']:.4f}")
        print(f"Log loss:        {model_metrics['log_loss']:.4f}")
        print(f"Draw real rate:  {model_metrics['draw_real_rate']:.4f}")
        print(f"Draw pred rate:  {model_metrics['draw_pred_rate']:.4f}")


def main():
    print("Loading data...")
    raw_df = load_and_cache_results()

    print("Cleaning data...")
    clean_df = clean_results(raw_df)

    print("Splitting played and future matches...")
    played_matches, future_matches = split_played_and_future_matches(clean_df)

    print(f"Played matches: {played_matches.shape}")
    print(f"Future matches: {future_matches.shape}")

    print("Building model dataset...")
    model_df = build_model_dataset(played_matches)

    print(f"Model dataset shape: {model_df.shape}")

    print("Training and evaluating temporal model split...")
    results = evaluate_model_temporal_split(
    model_df=model_df,
    train_calibrator=True,
    n_oof_folds=CALIBRATOR_OOF_FOLDS,
    min_train_ratio=CALIBRATOR_MIN_TRAIN_RATIO,
)

    train_predictions = results["train_predictions"]
    test_predictions = results["test_predictions"]

    print_metrics(
        metrics=results["train_result_metrics"],
        title="Train result metrics",
    )

    print_metrics(
        metrics=results["test_result_metrics"],
        title="Test result metrics",
    )

    print("\nGoal MAE")
    print("--------")
    print(f"Train home MAE:    {results['train_mae']['home_mae']:.4f}")
    print(f"Train away MAE:    {results['train_mae']['away_mae']:.4f}")
    print(f"Train overall MAE: {results['train_mae']['overall_mae']:.4f}")
    print(f"Test home MAE:     {results['test_mae']['home_mae']:.4f}")
    print(f"Test away MAE:     {results['test_mae']['away_mae']:.4f}")
    print(f"Test overall MAE:  {results['test_mae']['overall_mae']:.4f}")

    print("\nSample test predictions:")
    sample_cols = [
        "date",
        "home_team",
        "away_team",
        "home_score",
        "away_score",
        "actual_result",
        "lambda_home",
        "lambda_away",
        "lambda_home_int",
        "lambda_away_int",
        *RAW_PROB_COLS,
        *BLENDED_PROB_COLS,
        *CALIBRATED_PROB_COLS,
        "predicted_result_calibrated",
        "confidence_calibrated",
    ]

    existing_sample_cols = [
        col for col in sample_cols
        if col in test_predictions.columns
    ]

    print(test_predictions[existing_sample_cols].head(10))

    print("\nFeature columns used:")
    for col in MODEL_FEATURES:
        print(f"- {col}")

    print("\nTraining completed successfully.")


if __name__ == "__main__":
    main()