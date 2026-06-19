import pandas as pd


REQUIRED_COLUMNS = [
    "date",
    "home_team",
    "away_team",
    "home_score",
    "away_score",
    "tournament",
    "city",
    "country",
    "neutral",
]


def validate_columns(df: pd.DataFrame) -> None:
    """
    Validate that the raw dataset contains the expected columns.
    """
    missing_cols = [col for col in REQUIRED_COLUMNS if col not in df.columns]

    if missing_cols:
        raise ValueError(f"Missing required columns: {missing_cols}")


def clean_results(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply basic cleaning to the raw results dataset.

    This function does not remove future matches.
    It only standardizes types and ordering.
    """
    validate_columns(df)

    df = df.copy()

    df["date"] = pd.to_datetime(df["date"], errors="coerce")

    if df["date"].isna().any():
        bad_rows = df[df["date"].isna()]
        raise ValueError(
            f"Found rows with invalid dates. Example rows:\n{bad_rows.head()}"
        )

    df["home_team"] = df["home_team"].astype(str).str.strip()
    df["away_team"] = df["away_team"].astype(str).str.strip()
    df["tournament"] = df["tournament"].astype(str).str.strip()
    df["city"] = df["city"].astype(str).str.strip()
    df["country"] = df["country"].astype(str).str.strip()

    df["neutral"] = df["neutral"].astype(bool)

    df = df.sort_values("date").reset_index(drop=True)

    return df


def split_played_and_future_matches(
    df: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split dataset into played matches and future calendar matches.

    Played matches:
        rows with both home_score and away_score available.

    Future matches:
        rows where at least one score is missing.
    """
    df = df.copy()

    played_mask = df["home_score"].notna() & df["away_score"].notna()

    played_matches = df.loc[played_mask].copy()
    future_matches = df.loc[~played_mask].copy()

    played_matches["home_score"] = played_matches["home_score"].astype(int)
    played_matches["away_score"] = played_matches["away_score"].astype(int)

    return played_matches, future_matches


def filter_training_period(
    df: pd.DataFrame,
    start_date: str,
) -> pd.DataFrame:
    """
    Keep matches from a given start date.

    Used for training, for example from 2010 onward.
    """
    df = df.copy()
    start_date = pd.to_datetime(start_date)

    return df[df["date"] >= start_date].reset_index(drop=True)