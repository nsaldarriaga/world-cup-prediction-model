from pathlib import Path


# ============================================================
# Project paths
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"

OUTPUTS_DIR = PROJECT_ROOT / "outputs"

RESULTS_LOCAL_PATH = RAW_DATA_DIR / "results.csv"


# ============================================================
# Data source
# ============================================================

RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/"
    "international_results/master/results.csv"
)


# ============================================================
# Training configuration
# ============================================================

TRAIN_START_DATE = "2010-01-01"


# ============================================================
# Evaluation configuration
# ============================================================

TRAIN_END_DATE = "2022-12-31"
TEST_START_DATE = "2023-01-01"


# ============================================================
# Elo configuration
# ============================================================

INITIAL_ELO = 1500
ELO_K = 30
HOME_ADVANTAGE = 50


# ============================================================
# Poisson model configuration
# ============================================================

POISSON_ALPHA = 0.1
POISSON_MAX_ITER = 1000


# ============================================================
# Probability calibration configuration
# ============================================================

# Simple blending baseline.
# This is kept as a benchmark, not as the main final model.
BLENDING_WEIGHT_MODEL = 0.70

# Multinomial calibrator trained with temporal out-of-fold predictions.
CALIBRATOR_OOF_FOLDS = 4
CALIBRATOR_MIN_TRAIN_RATIO = 0.50
CALIBRATOR_C = 1.0
CALIBRATOR_MAX_ITER = 1000


# ============================================================
# Simulation configuration
# ============================================================

MAX_GOALS = 10
N_SIMULATIONS = 10_000
RANDOM_STATE = 42


# ============================================================
# Next round prediction configuration
# ============================================================

# Number of calendar days to include after the first pending match date.
# Example:
# 0  -> only the first pending date
# 6  -> first pending date plus the following 6 days
# 12 -> first pending date plus the following 12 days
NEXT_ROUND_DATE_WINDOW_DAYS = 12