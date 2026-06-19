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
TIME_DECAY_HALF_LIFE_DAYS = 540.0
MISMATCH_ELO_DIVISOR = 350.0
MAX_EXPECTED_GOALS = 4.8
MIN_EXPECTED_GOALS = 0.12

TOURNAMENT_WEIGHTS = {
    "friendly": 0.65,
    "uefa nations league": 1.05,
    "afc asian cup": 1.15,
    "copa america": 1.2,
    "gold cup": 1.15,
    "african cup of nations": 1.15,
    "fifa world cup qualification": 1.15,
    "uefa euro qualification": 1.15,
    "uefa euro": 1.2,
    "fifa world cup": 1.3,
    "confederations cup": 1.15,
}
