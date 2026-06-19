from src.data_loader import load_results
from src.cleaning import (
    clean_results,
    split_played_and_future_matches,
)

from src.model import (
    build_model_dataset,
    train_poisson_models,
)

from src.calendar import (
    predict_calendar,
    monte_carlo_calendar_fast,
    save_calendar_predictions,
)


OUTPUT_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "tournament",
    "city",
    "country",
    "neutral",
    "lambda_home",
    "lambda_away",
    "lambda_home_int",
    "lambda_away_int",
    "prob_home_win",
    "prob_draw",
    "prob_away_win",
    "prob_home_win_calibrated",
    "prob_draw_calibrated",
    "prob_away_win_calibrated",
    "sim_prob_home_win",
    "sim_prob_draw",
    "sim_prob_away_win",
    "sim_avg_home_goals",
    "sim_avg_away_goals",
    "sim_most_common_home_goals",
    "sim_most_common_away_goals",
]


def main():
    print("Loading data...")
    raw_df = load_results(source="local")

    print("Cleaning data...")
    clean_df = clean_results(raw_df)

    print("Splitting played and future matches...")
    played_matches, future_matches = split_played_and_future_matches(clean_df)

    print(f"Played matches: {played_matches.shape}")
    print(f"Future matches: {future_matches.shape}")

    if future_matches.empty:
        print("No future matches found in the dataset.")
        return

    print("Building model dataset...")
    model_df = build_model_dataset(played_matches)

    print("Training Poisson models...")
    model_bundle = train_poisson_models(model_df)

    print("Predicting future calendar...")
    predictions_df = predict_calendar(
        future_matches=future_matches,
        played_matches=played_matches,
        model_bundle=model_bundle,
    )

    print("Running Monte Carlo simulations...")
    predictions_df = monte_carlo_calendar_fast(predictions_df)

    available_columns = [
        col for col in OUTPUT_COLUMNS
        if col in predictions_df.columns
    ]

    output_df = predictions_df[available_columns].copy()

    print("\nSample calendar predictions:")
    print(output_df.head(20))

    save_calendar_predictions(
        output_df,
        filename="calendar_predictions.csv",
    )

    print("\nCalendar prediction completed successfully.")


if __name__ == "__main__":
    main()