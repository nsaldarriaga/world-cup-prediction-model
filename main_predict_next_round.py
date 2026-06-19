from src.config import (
    NEXT_ROUND_DATE_WINDOW_DAYS,
    CALIBRATOR_OOF_FOLDS,
    CALIBRATOR_MIN_TRAIN_RATIO,
)

from src.data_loader import load_and_cache_results
from src.cleaning import (
    clean_results,
    split_played_and_future_matches,
)

from src.model import (
    build_model_dataset,
    train_poisson_models,
)

from src.evaluation import (
    train_oof_multinomial_calibrator,
    add_multinomial_calibrated_probabilities,
)

from src.calendar import (
    get_next_pending_matches,
    predict_calendar,
    monte_carlo_calendar_fast,
    save_next_round_predictions,
)


OUTPUT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "tournament",
    "city",
    "country",
    "neutral",

    # Expected goals from Poisson
    "lambda_home",
    "lambda_away",
    "lambda_home_int",
    "lambda_away_int",

    # Raw Poisson probabilities
    "prob_home_win",
    "prob_draw",
    "prob_away_win",

    # Blended benchmark probabilities
    "prob_home_win_blended",
    "prob_draw_blended",
    "prob_away_win_blended",

    # Final calibrated probabilities
    "prob_H_calibrated",
    "prob_D_calibrated",
    "prob_A_calibrated",
    "predicted_result_calibrated",
    "confidence_calibrated",

    # Monte Carlo simulation
    "sim_prob_home_win",
    "sim_prob_draw",
    "sim_prob_away_win",
    "sim_avg_home_goals",
    "sim_avg_away_goals",
    "sim_most_common_home_goals",
    "sim_most_common_away_goals",
]


def main():
    print("Loading updated data...")
    raw_df = load_and_cache_results()

    print("Cleaning data...")
    clean_df = clean_results(raw_df)

    print("Splitting real played matches and pending matches...")
    played_matches, future_matches = split_played_and_future_matches(clean_df)

    print(f"Played matches with real scores: {played_matches.shape}")
    print(f"Pending matches without scores: {future_matches.shape}")

    if future_matches.empty:
        print("No pending matches found.")
        return

    print("Selecting next pending round...")
    next_round_matches = get_next_pending_matches(
        future_matches=future_matches,
        date_window_days=NEXT_ROUND_DATE_WINDOW_DAYS,
    )

    next_round_date_min = next_round_matches["date"].min()
    next_round_date_max = next_round_matches["date"].max()

    print(
        f"Next round window: {next_round_date_min.date()} "
        f"to {next_round_date_max.date()}"
    )
    print(f"Matches selected: {next_round_matches.shape[0]}")

    print("Building model dataset using only real played matches...")
    model_df = build_model_dataset(played_matches)

    print(f"Model dataset shape: {model_df.shape}")

    print("Training Poisson models...")
    model_bundle = train_poisson_models(model_df)

    print("Training OOF multinomial calibrator...")
    calibrator, oof_debug_df = train_oof_multinomial_calibrator(
    train_df=model_df,
    n_folds=CALIBRATOR_OOF_FOLDS,
    min_train_ratio=CALIBRATOR_MIN_TRAIN_RATIO,
)

    model_bundle["calibrator"] = calibrator

    print(f"OOF calibration rows: {oof_debug_df.shape[0]}")

    print("Predicting next round with Poisson...")
    predictions_df = predict_calendar(
        future_matches=next_round_matches,
        played_matches=played_matches,
        model_bundle=model_bundle,
    )

    print("Adding multinomial calibrated probabilities...")
    predictions_df = add_multinomial_calibrated_probabilities(
        predictions_df=predictions_df,
        calibrator=calibrator,
    )

    print("Running Monte Carlo simulations...")
    predictions_df = monte_carlo_calendar_fast(predictions_df)

    available_columns = [
        col for col in OUTPUT_COLUMNS
        if col in predictions_df.columns
    ]

    output_df = predictions_df[available_columns].copy()

    print("\nNext round predictions:")
    print(output_df)

    save_next_round_predictions(
        output_df,
        filename="next_round_predictions.csv",
    )

    print("\nNext round prediction completed successfully.")


if __name__ == "__main__":
    main()