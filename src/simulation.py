import numpy as np
import pandas as pd

from src.config import N_SIMULATIONS, RANDOM_STATE


def simulate_match(
    lambda_home: float,
    lambda_away: float,
    random_state: int | None = None,
) -> dict:
    """
    Simulate a single football match using Poisson-distributed goals.

    Parameters
    ----------
    lambda_home : float
        Expected goals for the home team.
    lambda_away : float
        Expected goals for the away team.
    random_state : int | None
        Optional random seed.

    Returns
    -------
    dict
        Simulated home goals, away goals and result.
    """
    rng = np.random.default_rng(random_state)

    home_goals = rng.poisson(lambda_home)
    away_goals = rng.poisson(lambda_away)

    if home_goals > away_goals:
        result = "home_win"
    elif home_goals == away_goals:
        result = "draw"
    else:
        result = "away_win"

    return {
        "home_goals": int(home_goals),
        "away_goals": int(away_goals),
        "result": result,
    }


def monte_carlo_match_fast(
    lambda_home: float,
    lambda_away: float,
    n_simulations: int = N_SIMULATIONS,
    random_state: int = RANDOM_STATE,
) -> dict:
    """
    Fast Monte Carlo simulation for one match.

    The model should already have produced lambda_home and lambda_away.
    This function does not call the model internally.
    """
    rng = np.random.default_rng(random_state)

    home_goals = rng.poisson(lambda_home, size=n_simulations)
    away_goals = rng.poisson(lambda_away, size=n_simulations)

    home_wins = home_goals > away_goals
    draws = home_goals == away_goals
    away_wins = home_goals < away_goals

    prob_home_win = home_wins.mean()
    prob_draw = draws.mean()
    prob_away_win = away_wins.mean()

    avg_home_goals = home_goals.mean()
    avg_away_goals = away_goals.mean()

    most_common_home_goals = int(pd.Series(home_goals).mode().iloc[0])
    most_common_away_goals = int(pd.Series(away_goals).mode().iloc[0])

    return {
        "sim_prob_home_win": prob_home_win,
        "sim_prob_draw": prob_draw,
        "sim_prob_away_win": prob_away_win,
        "sim_avg_home_goals": avg_home_goals,
        "sim_avg_away_goals": avg_away_goals,
        "sim_most_common_home_goals": most_common_home_goals,
        "sim_most_common_away_goals": most_common_away_goals,
        "n_simulations": n_simulations,
    }


def simulate_scoreline_distribution(
    lambda_home: float,
    lambda_away: float,
    n_simulations: int = N_SIMULATIONS,
    random_state: int = RANDOM_STATE,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Simulate a match and return the most frequent scorelines.

    Useful for explaining predicted score distributions.
    """
    rng = np.random.default_rng(random_state)

    home_goals = rng.poisson(lambda_home, size=n_simulations)
    away_goals = rng.poisson(lambda_away, size=n_simulations)

    scorelines = pd.DataFrame({
        "home_goals": home_goals,
        "away_goals": away_goals,
    })

    scoreline_distribution = (
        scorelines
        .value_counts()
        .reset_index(name="count")
    )

    scoreline_distribution["probability"] = (
        scoreline_distribution["count"] / n_simulations
    )

    scoreline_distribution = scoreline_distribution.sort_values(
        "probability",
        ascending=False,
    ).head(top_n)

    return scoreline_distribution.reset_index(drop=True)


def monte_carlo_dataframe_fast(
    df: pd.DataFrame,
    lambda_home_col: str = "lambda_home",
    lambda_away_col: str = "lambda_away",
    n_simulations: int = N_SIMULATIONS,
    random_state: int = RANDOM_STATE,
) -> pd.DataFrame:
    """
    Apply fast Monte Carlo simulation to each row of a dataframe.

    The dataframe must already contain expected goals columns.
    """
    df = df.copy()

    simulation_results = []

    for idx, row in df.iterrows():
        result = monte_carlo_match_fast(
            lambda_home=row[lambda_home_col],
            lambda_away=row[lambda_away_col],
            n_simulations=n_simulations,
            random_state=random_state + idx,
        )

        simulation_results.append(result)

    simulation_df = pd.DataFrame(simulation_results)

    df = pd.concat(
        [
            df.reset_index(drop=True),
            simulation_df.reset_index(drop=True),
        ],
        axis=1,
    )

    return df