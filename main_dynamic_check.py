import pandas as pd
import matplotlib.pyplot as plt

from src.config import (
    OUTPUTS_DIR,
    NEXT_ROUND_DATE_WINDOW_DAYS,
    CALIBRATOR_OOF_FOLDS,
    CALIBRATOR_MIN_TRAIN_RATIO,
)

from src.data_loader import load_and_cache_results

from src.cleaning import (
    clean_results,
    split_played_and_future_matches,
)

from src.config import (
    OUTPUTS_DIR,
    NEXT_ROUND_DATE_WINDOW_DAYS,
    CALIBRATOR_OOF_FOLDS,
    CALIBRATOR_MIN_TRAIN_RATIO,
    INITIAL_ELO,
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
)

from src.team_state import (
    build_team_state,
    team_state_to_dataframe,
)

WORLD_CUP_START_DATE = "2026-06-11"
WORLD_CUP_END_DATE = "2026-06-27"


def build_predictions_for_scenario(
    played_matches: pd.DataFrame,
    future_matches: pd.DataFrame,
    scenario_name: str,
) -> pd.DataFrame:
    """
    Build model dataset, train Poisson + OOF calibrator,
    and predict the next pending window.
    """
    model_df = build_model_dataset(played_matches)

    model_bundle = train_poisson_models(model_df)

    calibrator, _ = train_oof_multinomial_calibrator(
        train_df=model_df,
        n_folds=CALIBRATOR_OOF_FOLDS,
        min_train_ratio=CALIBRATOR_MIN_TRAIN_RATIO,
    )

    model_bundle["calibrator"] = calibrator

    next_matches = get_next_pending_matches(
        future_matches=future_matches,
        date_window_days=NEXT_ROUND_DATE_WINDOW_DAYS,
    )

    predictions_df = predict_calendar(
        future_matches=next_matches,
        played_matches=played_matches,
        model_bundle=model_bundle,
    )

    predictions_df = add_multinomial_calibrated_probabilities(
        predictions_df=predictions_df,
        calibrator=calibrator,
    )

    predictions_df["scenario"] = scenario_name

    return predictions_df


def build_prediction_comparison(
    predictions_before: pd.DataFrame,
    predictions_after: pd.DataFrame,
) -> pd.DataFrame:
    """
    Compare predictions before and after adding the latest real match.
    """
    key_cols = [
        "date",
        "home_team",
        "away_team",
    ]

    compare_cols = [
        "lambda_home",
        "lambda_away",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "prob_H_calibrated",
        "prob_D_calibrated",
        "prob_A_calibrated",
        "predicted_result_calibrated",
        "confidence_calibrated",
    ]

    before_small = predictions_before[key_cols + compare_cols].copy()
    after_small = predictions_after[key_cols + compare_cols].copy()

    comparison_df = before_small.merge(
        after_small,
        on=key_cols,
        suffixes=("_before", "_after"),
    )

    numeric_cols = [
        "lambda_home",
        "lambda_away",
        "prob_home_win",
        "prob_draw",
        "prob_away_win",
        "prob_H_calibrated",
        "prob_D_calibrated",
        "prob_A_calibrated",
        "confidence_calibrated",
    ]

    for col in numeric_cols:
        comparison_df[f"delta_{col}"] = (
            comparison_df[f"{col}_after"]
            - comparison_df[f"{col}_before"]
        )

    comparison_df["match"] = (
        comparison_df["home_team"] + " vs " + comparison_df["away_team"]
    )

    comparison_df["total_abs_probability_change"] = (
        comparison_df[
            [
                "delta_prob_H_calibrated",
                "delta_prob_D_calibrated",
                "delta_prob_A_calibrated",
            ]
        ]
        .abs()
        .sum(axis=1)
    )

    return comparison_df


def build_elo_comparison(
    played_before: pd.DataFrame,
    played_after: pd.DataFrame,
    last_match: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Build Elo ranking after latest match and Elo changes for the teams involved
    in the latest match.
    """
    state_before = build_team_state(played_before)
    state_after = build_team_state(played_after)

    elo_before = team_state_to_dataframe(state_before)
    elo_after = team_state_to_dataframe(state_after)

    elo_before = elo_before.rename(columns={"elo": "elo_before"})
    elo_after = elo_after.rename(columns={"elo": "elo_after"})

    ranking_after = elo_after.copy()
    ranking_after["elo_rank_after"] = ranking_after["elo_after"].rank(
        ascending=False,
        method="min",
    ).astype(int)

    ranking_after = ranking_after.sort_values(
        "elo_after",
        ascending=False,
    ).reset_index(drop=True)

    involved_teams = [
        last_match["home_team"],
        last_match["away_team"],
    ]

    elo_changes = elo_before[["team", "elo_before"]].merge(
        elo_after[["team", "elo_after"]],
        on="team",
        how="outer",
    )

    elo_changes["elo_change"] = (
        elo_changes["elo_after"] - elo_changes["elo_before"]
    )

    elo_changes = elo_changes[
        elo_changes["team"].isin(involved_teams)
    ].copy()

    elo_changes = elo_changes.sort_values(
        "elo_change",
        ascending=False,
    ).reset_index(drop=True)

    return ranking_after, elo_changes


def plot_prediction_changes(
    comparison_df: pd.DataFrame,
    output_path,
    top_n: int = 20,
) -> None:
    """
    Save a bar chart showing which future matches changed most after adding
    the latest real match.
    """
    plot_df = comparison_df.copy()

    plot_df = plot_df.sort_values(
        "total_abs_probability_change",
        ascending=False,
    ).head(top_n)

    if plot_df.empty:
        print("No prediction changes available for plotting.")
        return

    plt.figure(figsize=(12, 7))

    plt.barh(
        plot_df["match"],
        plot_df["total_abs_probability_change"],
    )

    plt.xlabel("Total absolute change in calibrated probabilities")
    plt.ylabel("Match")
    plt.title("Dynamic prediction impact after latest real match")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_elo_ranking(
    ranking_after: pd.DataFrame,
    output_path,
    top_n: int = 20,
) -> None:
    """
    Save a bar chart with the top Elo ranking after the latest real match.
    """
    plot_df = ranking_after.copy().head(top_n)

    if plot_df.empty:
        print("No Elo ranking available for plotting.")
        return

    plt.figure(figsize=(12, 7))

    plt.barh(
        plot_df["team"],
        plot_df["elo_after"],
    )

    plt.xlabel("Elo rating")
    plt.ylabel("Team")
    plt.title(f"Top {top_n} teams by Elo after latest real match")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def get_world_cup_matches_in_window(
    clean_df: pd.DataFrame,
    start_date: str = WORLD_CUP_START_DATE,
    end_date: str = WORLD_CUP_END_DATE,
) -> pd.DataFrame:
    """
    Get all World Cup matches in the configured tournament window.

    This includes both played and pending matches.
    """
    df = clean_df.copy()
    df["date"] = pd.to_datetime(df["date"])

    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    wc_matches = df[
        (df["date"] >= start_date)
        & (df["date"] <= end_date)
    ].copy()

    return wc_matches.sort_values("date").reset_index(drop=True)


def get_world_cup_participants(
    world_cup_matches: pd.DataFrame,
) -> list[str]:
    """
    Get all teams participating in the World Cup window.
    """
    teams = sorted(
        set(world_cup_matches["home_team"].dropna())
        | set(world_cup_matches["away_team"].dropna())
    )

    return teams


def get_played_world_cup_matches(
    world_cup_matches: pd.DataFrame,
) -> pd.DataFrame:
    """
    Keep only World Cup matches that already have real scores.
    """
    played_wc = world_cup_matches[
        world_cup_matches["home_score"].notna()
        & world_cup_matches["away_score"].notna()
    ].copy()

    return played_wc.sort_values("date").reset_index(drop=True)


def build_world_cup_team_results(
    played_world_cup_matches: pd.DataFrame,
    participants: list[str],
) -> pd.DataFrame:
    """
    Build World Cup results summary for participating teams.
    """
    rows = []

    for team in participants:
        team_matches = played_world_cup_matches[
            (played_world_cup_matches["home_team"] == team)
            | (played_world_cup_matches["away_team"] == team)
        ].copy()

        wins = 0
        draws = 0
        losses = 0
        goals_for = 0
        goals_against = 0
        points = 0

        for _, match in team_matches.iterrows():
            is_home = match["home_team"] == team

            if is_home:
                gf = int(match["home_score"])
                ga = int(match["away_score"])
            else:
                gf = int(match["away_score"])
                ga = int(match["home_score"])

            goals_for += gf
            goals_against += ga

            if gf > ga:
                wins += 1
                points += 3
            elif gf == ga:
                draws += 1
                points += 1
            else:
                losses += 1

        rows.append({
            "team": team,
            "world_cup_matches_played": len(team_matches),
            "wins": wins,
            "draws": draws,
            "losses": losses,
            "goals_for": goals_for,
            "goals_against": goals_against,
            "goal_difference": goals_for - goals_against,
            "points": points,
        })

    return pd.DataFrame(rows)


def build_world_cup_elo_evolution(
    clean_df: pd.DataFrame,
    played_matches: pd.DataFrame,
    start_date: str = WORLD_CUP_START_DATE,
    end_date: str = WORLD_CUP_END_DATE,
) -> pd.DataFrame:
    """
    Compare Elo before the World Cup vs Elo after the latest played World Cup match.

    This demonstrates the dynamic update:
    historical state before tournament -> updated state after real World Cup matches.
    """
    start_date = pd.to_datetime(start_date)
    end_date = pd.to_datetime(end_date)

    world_cup_matches = get_world_cup_matches_in_window(
        clean_df=clean_df,
        start_date=start_date,
        end_date=end_date,
    )

    participants = get_world_cup_participants(world_cup_matches)

    played_world_cup_matches = get_played_world_cup_matches(world_cup_matches)

    if played_world_cup_matches.empty:
        latest_wc_played_date = None
        played_until_latest_wc = played_matches[
            played_matches["date"] < start_date
        ].copy()
    else:
        latest_wc_played_date = played_world_cup_matches["date"].max()

        played_until_latest_wc = played_matches[
            played_matches["date"] <= latest_wc_played_date
        ].copy()

    played_before_world_cup = played_matches[
        played_matches["date"] < start_date
    ].copy()

    state_before = build_team_state(played_before_world_cup)
    state_after = build_team_state(played_until_latest_wc)

    rows = []

    for team in participants:
        elo_before = state_before.get(team, {}).get("elo", INITIAL_ELO)
        elo_after = state_after.get(team, {}).get("elo", INITIAL_ELO)

        rows.append({
            "team": team,
            "elo_before_world_cup": elo_before,
            "elo_after_latest_world_cup_match": elo_after,
            "elo_change": elo_after - elo_before,
            "latest_world_cup_played_date": latest_wc_played_date,
        })

    elo_df = pd.DataFrame(rows)

    elo_df["elo_rank_before"] = (
        elo_df["elo_before_world_cup"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    elo_df["elo_rank_after"] = (
        elo_df["elo_after_latest_world_cup_match"]
        .rank(ascending=False, method="min")
        .astype(int)
    )

    elo_df["rank_change"] = (
        elo_df["elo_rank_before"] - elo_df["elo_rank_after"]
    )

    results_df = build_world_cup_team_results(
        played_world_cup_matches=played_world_cup_matches,
        participants=participants,
    )

    elo_df = elo_df.merge(
        results_df,
        on="team",
        how="left",
    )

    elo_df = elo_df.sort_values(
        "elo_after_latest_world_cup_match",
        ascending=False,
    ).reset_index(drop=True)

    return elo_df


def plot_world_cup_elo_ranking(
    elo_evolution_df: pd.DataFrame,
    output_path,
    top_n: int = 30,
) -> None:
    """
    Save Elo ranking chart for World Cup participants after latest played match.
    """
    plot_df = (
        elo_evolution_df
        .sort_values("elo_after_latest_world_cup_match", ascending=False)
        .head(top_n)
        .copy()
    )

    if plot_df.empty:
        print("No World Cup Elo ranking available for plotting.")
        return

    plt.figure(figsize=(12, 9))

    plt.barh(
        plot_df["team"],
        plot_df["elo_after_latest_world_cup_match"],
    )

    plt.xlabel("Elo after latest World Cup match")
    plt.ylabel("Team")
    plt.title(f"World Cup Elo ranking after latest played match - Top {top_n}")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def plot_world_cup_elo_changes(
    elo_evolution_df: pd.DataFrame,
    output_path,
    top_n: int = 30,
) -> None:
    """
    Save Elo change chart for World Cup participants.
    """
    plot_df = elo_evolution_df.copy()

    plot_df = plot_df.reindex(
        plot_df["elo_change"].abs().sort_values(ascending=False).index
    ).head(top_n)

    if plot_df.empty:
        print("No World Cup Elo changes available for plotting.")
        return

    plt.figure(figsize=(12, 9))

    plt.barh(
        plot_df["team"],
        plot_df["elo_change"],
    )

    plt.axvline(0, linewidth=1)

    plt.xlabel("Elo change since World Cup start")
    plt.ylabel("Team")
    plt.title(f"Biggest Elo changes during World Cup window - Top {top_n}")
    plt.gca().invert_yaxis()

    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()


def main():
    OUTPUTS_DIR.mkdir(parents=True, exist_ok=True)

    print("Loading data...")
    raw_df = load_and_cache_results()

    print("Cleaning data...")
    clean_df = clean_results(raw_df)

    print("Splitting played and future matches...")
    played_matches, future_matches = split_played_and_future_matches(clean_df)

    if played_matches.empty:
        raise ValueError("No played matches found.")

    if future_matches.empty:
        raise ValueError("No future matches found.")

    played_matches = played_matches.sort_values("date").reset_index(drop=True)

    print("\nBuilding World Cup Elo evolution for participating teams...")

    world_cup_elo_evolution = build_world_cup_elo_evolution(
        clean_df=clean_df,
        played_matches=played_matches,
        start_date=WORLD_CUP_START_DATE,
        end_date=WORLD_CUP_END_DATE,
    )

    world_cup_elo_path = OUTPUTS_DIR / "dynamic_check_world_cup_elo_evolution.csv"
    world_cup_elo_ranking_plot_path = OUTPUTS_DIR / "dynamic_check_world_cup_elo_ranking.png"
    world_cup_elo_changes_plot_path = OUTPUTS_DIR / "dynamic_check_world_cup_elo_changes.png"

    world_cup_elo_evolution.to_csv(world_cup_elo_path, index=False)

    plot_world_cup_elo_ranking(
        elo_evolution_df=world_cup_elo_evolution,
        output_path=world_cup_elo_ranking_plot_path,
        top_n=30,
    )

    plot_world_cup_elo_changes(
        elo_evolution_df=world_cup_elo_evolution,
        output_path=world_cup_elo_changes_plot_path,
        top_n=30,
    )

    print(f"Saved World Cup Elo evolution to: {world_cup_elo_path}")
    print(f"Saved World Cup Elo ranking chart to: {world_cup_elo_ranking_plot_path}")
    print(f"Saved World Cup Elo changes chart to: {world_cup_elo_changes_plot_path}")

    print("\nWorld Cup Elo evolution sample:")
    print(
        world_cup_elo_evolution[
            [
                "team",
                "elo_before_world_cup",
                "elo_after_latest_world_cup_match",
                "elo_change",
                "elo_rank_before",
                "elo_rank_after",
                "rank_change",
                "world_cup_matches_played",
                "wins",
                "draws",
                "losses",
                "points",
            ]
        ].head(20)
    )

    last_match = played_matches.iloc[-1]

    print("\nLast played match used for dynamic check:")
    print(
        last_match[
            [
                "date",
                "home_team",
                "away_team",
                "home_score",
                "away_score",
                "tournament",
                "neutral",
            ]
        ]
    )

    # Scenario A: before latest real match
    played_before_last_match = played_matches.iloc[:-1].copy()

    # Scenario B: after latest real match
    played_after_last_match = played_matches.copy()

    print("\nBuilding Elo comparison...")
    ranking_after, elo_changes = build_elo_comparison(
        played_before=played_before_last_match,
        played_after=played_after_last_match,
        last_match=last_match,
    )

    ranking_path = OUTPUTS_DIR / "dynamic_check_elo_ranking_after.csv"
    elo_changes_path = OUTPUTS_DIR / "dynamic_check_elo_changes_last_match_teams.csv"

    ranking_after.to_csv(ranking_path, index=False)
    elo_changes.to_csv(elo_changes_path, index=False)

    print(f"Saved Elo ranking to: {ranking_path}")
    print(f"Saved Elo changes to: {elo_changes_path}")

    print("\nElo changes for teams involved in latest match:")
    print(elo_changes)

    elo_plot_path = OUTPUTS_DIR / "dynamic_check_elo_ranking_after.png"

    plot_elo_ranking(
        ranking_after=ranking_after,
        output_path=elo_plot_path,
        top_n=20,
    )

    print(f"Saved Elo ranking chart to: {elo_plot_path}")

    print("\nBuilding predictions BEFORE latest match...")
    predictions_before = build_predictions_for_scenario(
        played_matches=played_before_last_match,
        future_matches=future_matches,
        scenario_name="before_latest_match",
    )

    print("Building predictions AFTER latest match...")
    predictions_after = build_predictions_for_scenario(
        played_matches=played_after_last_match,
        future_matches=future_matches,
        scenario_name="after_latest_match",
    )

    comparison_df = build_prediction_comparison(
        predictions_before=predictions_before,
        predictions_after=predictions_after,
    )

    selected_output_cols = [
        "date",
        "home_team",
        "away_team",
        "match",

        "lambda_home_before",
        "lambda_home_after",
        "delta_lambda_home",

        "lambda_away_before",
        "lambda_away_after",
        "delta_lambda_away",

        "prob_H_calibrated_before",
        "prob_H_calibrated_after",
        "delta_prob_H_calibrated",

        "prob_D_calibrated_before",
        "prob_D_calibrated_after",
        "delta_prob_D_calibrated",

        "prob_A_calibrated_before",
        "prob_A_calibrated_after",
        "delta_prob_A_calibrated",

        "predicted_result_calibrated_before",
        "predicted_result_calibrated_after",

        "confidence_calibrated_before",
        "confidence_calibrated_after",
        "delta_confidence_calibrated",

        "total_abs_probability_change",
    ]

    selected_output_cols = [
        col for col in selected_output_cols
        if col in comparison_df.columns
    ]

    output_df = comparison_df[selected_output_cols].copy()

    prediction_csv_path = OUTPUTS_DIR / "dynamic_check_predictions.csv"

    output_df.to_csv(prediction_csv_path, index=False)

    print(f"\nSaved dynamic prediction comparison to: {prediction_csv_path}")

    prediction_plot_path = OUTPUTS_DIR / "dynamic_check_prediction_changes.png"

    plot_prediction_changes(
        comparison_df=comparison_df,
        output_path=prediction_plot_path,
        top_n=20,
    )

    print(f"Saved prediction changes chart to: {prediction_plot_path}")

    print("\nDynamic prediction comparison:")
    print(output_df)

    total_abs_change = comparison_df[
        [
            "delta_lambda_home",
            "delta_lambda_away",
            "delta_prob_H_calibrated",
            "delta_prob_D_calibrated",
            "delta_prob_A_calibrated",
        ]
    ].abs().sum().sum()

    print("\nDynamic check summary")
    print("---------------------")
    print(f"Total absolute prediction change: {total_abs_change:.8f}")

    if total_abs_change > 0:
        print("Result: model predictions changed after adding the latest match.")
    else:
        print(
            "Result: no numerical prediction change detected. "
            "Review whether the latest match affects teams in the selected future window."
        )

    if not elo_changes.empty:
        total_abs_elo_change = elo_changes["elo_change"].abs().sum()
        print(f"Total Elo change for latest match teams: {total_abs_elo_change:.8f}")
        print("Result: Elo state changed after adding the latest match.")


if __name__ == "__main__":
    main()