import pandas as pd

from src.config import RESULTS_URL, RESULTS_LOCAL_PATH, RAW_DATA_DIR


def load_results(source: str = "url") -> pd.DataFrame:
    """
    Load international football results.

    Parameters
    ----------
    source : str
        "url"   -> load from GitHub raw CSV.
        "local" -> load from data/raw/results.csv.

    Returns
    -------
    pd.DataFrame
        Raw results dataset.
    """
    if source == "url":
        df = pd.read_csv(RESULTS_URL)

    elif source == "local":
        if not RESULTS_LOCAL_PATH.exists():
            raise FileNotFoundError(
                f"Local file not found: {RESULTS_LOCAL_PATH}. "
                "Download it first or use source='url'."
            )
        df = pd.read_csv(RESULTS_LOCAL_PATH)

    else:
        raise ValueError("source must be either 'url' or 'local'.")

    return df


def save_raw_results(df: pd.DataFrame) -> None:
    """
    Save raw results dataset into data/raw/results.csv.
    """
    RAW_DATA_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(RESULTS_LOCAL_PATH, index=False)


def load_and_cache_results() -> pd.DataFrame:
    """
    Load results from GitHub and save a local copy.
    """
    df = load_results(source="url")
    save_raw_results(df)
    return df

def load_and_cache_results() -> pd.DataFrame:
    """
    Load results from GitHub and save a local copy.
    """
    print("Loading latest results from GitHub...")
    df = load_results(source="url")

    print(f"Saving latest results to: {RESULTS_LOCAL_PATH}")
    save_raw_results(df)

    return df