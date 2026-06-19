from __future__ import annotations

from pathlib import Path

import pandas as pd
import requests

from .config import INTERNATIONAL_RESULTS_URL, RAW_DIR


RESULTS_CSV = RAW_DIR / "international_results.csv"


def ensure_data_dir() -> None:
    RAW_DIR.mkdir(parents=True, exist_ok=True)


def download_results(force: bool = False) -> Path:
    ensure_data_dir()
    if RESULTS_CSV.exists() and not force:
        return RESULTS_CSV

    response = requests.get(INTERNATIONAL_RESULTS_URL, timeout=60)
    response.raise_for_status()
    RESULTS_CSV.write_bytes(response.content)
    return RESULTS_CSV


def load_results(force_download: bool = False) -> pd.DataFrame:
    csv_path = download_results(force=force_download)
    df = pd.read_csv(csv_path)
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).copy()
    df["home_score"] = pd.to_numeric(df["home_score"], errors="coerce")
    df["away_score"] = pd.to_numeric(df["away_score"], errors="coerce")
    df["neutral"] = (
        df["neutral"].astype(str).str.strip().str.lower().map({"true": True, "false": False})
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df
