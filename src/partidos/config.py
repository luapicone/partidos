from pathlib import Path


ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"

INTERNATIONAL_RESULTS_URL = (
    "https://raw.githubusercontent.com/martj42/international_results/master/results.csv"
)

ELO_BASE = 1500.0
ELO_K = 32.0
HOME_ADVANTAGE = 55.0
FORM_MATCHES = 8
POISSON_MAX_GOALS = 8
