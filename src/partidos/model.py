from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .config import (
    ELO_BASE,
    ELO_K,
    FORM_MATCHES,
    HOME_ADVANTAGE,
    MAX_EXPECTED_GOALS,
    MIN_EXPECTED_GOALS,
    MISMATCH_ELO_DIVISOR,
    POISSON_MAX_GOALS,
    TIME_DECAY_HALF_LIFE_DAYS,
    TOURNAMENT_WEIGHTS,
)


@dataclass
class TeamSnapshot:
    team: str
    elo: float
    matches: int
    wins: int
    draws: int
    losses: int
    goals_for_avg: float
    goals_against_avg: float
    recent_points_per_match: float
    recent_points_adjusted: float
    recent_goals_for_avg: float
    recent_goals_against_avg: float
    recent_goals_for_adjusted: float
    recent_goals_against_adjusted: float
    scoring_rate: float
    clean_sheet_rate: float


@dataclass
class Prediction:
    team_a: str
    team_b: str
    match_date: str
    neutral: bool
    team_a_snapshot: TeamSnapshot
    team_b_snapshot: TeamSnapshot
    expected_goals_a: float
    expected_goals_b: float
    win_prob_a: float
    draw_prob: float
    win_prob_b: float
    most_likely_score: tuple[int, int]


@dataclass
class BacktestResult:
    matches_evaluated: int
    accuracy: float
    log_loss: float
    brier_score: float
    avg_confidence: float
    baseline_home_accuracy: float
    baseline_elo_accuracy: float


def _result_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _goal_margin_multiplier(goal_diff: int) -> float:
    return math.log(goal_diff + 1.0) * 2.2


def _expected_from_elo(rating_a: float, rating_b: float) -> float:
    return 1.0 / (1.0 + 10 ** ((rating_b - rating_a) / 400.0))


def tournament_weight(tournament: str) -> float:
    normalized = str(tournament).strip().lower()
    if normalized in TOURNAMENT_WEIGHTS:
        return TOURNAMENT_WEIGHTS[normalized]
    if "qualification" in normalized or "qualifier" in normalized:
        return 1.15
    if "friendly" in normalized:
        return 0.65
    return 1.0


def _time_decay(match_date: pd.Timestamp, reference_date: pd.Timestamp) -> float:
    days = max((reference_date - match_date).days, 0)
    return 0.5 ** (days / TIME_DECAY_HALF_LIFE_DAYS)


def build_elo_history(results: pd.DataFrame) -> dict[str, float]:
    ratings: dict[str, float] = {}
    ordered = results.sort_values("date")

    for row in ordered.itertuples(index=False):
        home = row.home_team
        away = row.away_team
        ratings.setdefault(home, ELO_BASE)
        ratings.setdefault(away, ELO_BASE)

        home_rating = ratings[home]
        away_rating = ratings[away]

        home_advantage = 0.0 if bool(row.neutral) else HOME_ADVANTAGE
        expected_home = _expected_from_elo(home_rating + home_advantage, away_rating)
        expected_away = 1.0 - expected_home

        if row.home_score > row.away_score:
            actual_home, actual_away = 1.0, 0.0
        elif row.home_score < row.away_score:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home = actual_away = 0.5

        goal_diff = abs(int(row.home_score) - int(row.away_score))
        multiplier = _goal_margin_multiplier(goal_diff) if goal_diff else 1.0
        adjustment = ELO_K * multiplier * tournament_weight(row.tournament)

        ratings[home] = home_rating + adjustment * (actual_home - expected_home)
        ratings[away] = away_rating + adjustment * (actual_away - expected_away)

    return ratings


def _team_matches(results: pd.DataFrame, team: str) -> pd.DataFrame:
    mask = (results["home_team"] == team) | (results["away_team"] == team)
    matches = results.loc[mask].copy()
    if matches.empty:
        return matches.sort_values("date")
    matches["is_home"] = matches["home_team"] == team
    matches["goals_for"] = matches.apply(
        lambda row: row["home_score"] if row["is_home"] else row["away_score"], axis=1
    )
    matches["goals_against"] = matches.apply(
        lambda row: row["away_score"] if row["is_home"] else row["home_score"], axis=1
    )
    matches["points"] = matches.apply(
        lambda row: _result_points(int(row["goals_for"]), int(row["goals_against"])), axis=1
    )
    matches["opponent"] = matches.apply(
        lambda row: row["away_team"] if row["is_home"] else row["home_team"], axis=1
    )
    return matches.sort_values("date")


def _weighted_average(series: pd.Series, weights: pd.Series, fallback: float) -> float:
    if series.empty:
        return fallback
    total_weight = float(weights.sum())
    if total_weight <= 0:
        return fallback
    return float((series * weights).sum() / total_weight)


def build_team_snapshot(
    results: pd.DataFrame,
    ratings: dict[str, float],
    team: str,
    reference_date: pd.Timestamp,
) -> TeamSnapshot:
    matches = _team_matches(results, team)
    if matches.empty:
        return TeamSnapshot(
            team=team,
            elo=ratings.get(team, ELO_BASE),
            matches=0,
            wins=0,
            draws=0,
            losses=0,
            goals_for_avg=1.0,
            goals_against_avg=1.0,
            recent_points_per_match=1.0,
            recent_points_adjusted=1.0,
            recent_goals_for_avg=1.0,
            recent_goals_against_avg=1.0,
            recent_goals_for_adjusted=1.0,
            recent_goals_against_adjusted=1.0,
            scoring_rate=0.5,
            clean_sheet_rate=0.25,
        )

    recent = matches.tail(FORM_MATCHES)
    recent = recent.copy()
    recent["opponent_elo"] = recent["opponent"].map(lambda name: ratings.get(name, ELO_BASE))
    recent["time_weight"] = recent["date"].map(lambda value: _time_decay(value, reference_date))
    recent["tournament_weight"] = recent["tournament"].map(tournament_weight)
    recent["sample_weight"] = recent["time_weight"] * recent["tournament_weight"]
    recent["opponent_factor"] = recent["opponent_elo"].map(
        lambda value: _clamp(value / ELO_BASE, 0.8, 1.25)
    )
    recent["adjusted_points"] = recent["points"] * recent["opponent_factor"]
    recent["adjusted_goals_for"] = recent["goals_for"] * recent["opponent_factor"]
    recent["adjusted_goals_against"] = recent["goals_against"] * recent["opponent_factor"].map(
        lambda value: _clamp(2.05 - value, 0.8, 1.25)
    )

    wins = int((matches["points"] == 3).sum())
    draws = int((matches["points"] == 1).sum())
    losses = int((matches["points"] == 0).sum())

    return TeamSnapshot(
        team=team,
        elo=ratings.get(team, ELO_BASE),
        matches=len(matches),
        wins=wins,
        draws=draws,
        losses=losses,
        goals_for_avg=float(matches["goals_for"].mean()),
        goals_against_avg=float(matches["goals_against"].mean()),
        recent_points_per_match=_weighted_average(recent["points"], recent["sample_weight"], 1.0),
        recent_points_adjusted=_weighted_average(
            recent["adjusted_points"], recent["sample_weight"], 1.0
        ),
        recent_goals_for_avg=_weighted_average(recent["goals_for"], recent["sample_weight"], 1.0),
        recent_goals_against_avg=_weighted_average(
            recent["goals_against"], recent["sample_weight"], 1.0
        ),
        recent_goals_for_adjusted=_weighted_average(
            recent["adjusted_goals_for"], recent["sample_weight"], 1.0
        ),
        recent_goals_against_adjusted=_weighted_average(
            recent["adjusted_goals_against"], recent["sample_weight"], 1.0
        ),
        scoring_rate=_weighted_average(
            (recent["goals_for"] > 0).astype(float), recent["sample_weight"], 0.5
        ),
        clean_sheet_rate=_weighted_average(
            (recent["goals_against"] == 0).astype(float), recent["sample_weight"], 0.25
        ),
    )


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _poisson_probability(lmbda: float, goals: int) -> float:
    return math.exp(-lmbda) * (lmbda**goals) / math.factorial(goals)


def _probability_matrix(lambda_a: float, lambda_b: float) -> tuple[float, float, float, tuple[int, int]]:
    win_a = 0.0
    draw = 0.0
    win_b = 0.0
    best_prob = -1.0
    best_score = (0, 0)

    for goals_a in range(POISSON_MAX_GOALS + 1):
        for goals_b in range(POISSON_MAX_GOALS + 1):
            prob = _poisson_probability(lambda_a, goals_a) * _poisson_probability(lambda_b, goals_b)
            if goals_a > goals_b:
                win_a += prob
            elif goals_a == goals_b:
                draw += prob
            else:
                win_b += prob
            if prob > best_prob:
                best_prob = prob
                best_score = (goals_a, goals_b)

    total = win_a + draw + win_b
    return win_a / total, draw / total, win_b / total, best_score


def predict_match(
    results: pd.DataFrame,
    team_a: str,
    team_b: str,
    match_date: str,
    neutral: bool = False,
) -> Prediction:
    match_ts = pd.Timestamp(match_date)
    history = results.loc[
        (results["date"] < match_ts)
        & results["home_score"].notna()
        & results["away_score"].notna()
    ].copy()
    if history.empty:
        raise ValueError("No hay historial previo a la fecha indicada.")

    ratings = build_elo_history(history)
    team_a_snapshot = build_team_snapshot(history, ratings, team_a, match_ts)
    team_b_snapshot = build_team_snapshot(history, ratings, team_b, match_ts)

    base_goals = float((history["home_score"].mean() + history["away_score"].mean()) / 2.0)
    attack_a = _clamp(team_a_snapshot.recent_goals_for_adjusted / max(base_goals, 0.1), 0.42, 2.35)
    defense_a = _clamp(team_a_snapshot.recent_goals_against_adjusted / max(base_goals, 0.1), 0.35, 2.0)
    attack_b = _clamp(team_b_snapshot.recent_goals_for_adjusted / max(base_goals, 0.1), 0.42, 2.35)
    defense_b = _clamp(team_b_snapshot.recent_goals_against_adjusted / max(base_goals, 0.1), 0.35, 2.0)

    elo_edge = team_a_snapshot.elo - team_b_snapshot.elo + (0 if neutral else HOME_ADVANTAGE)
    elo_factor_a = _clamp(math.exp(elo_edge / 700.0), 0.68, 1.65)
    elo_factor_b = _clamp(math.exp(-elo_edge / 700.0), 0.55, 1.45)

    form_factor_a = _clamp(0.83 + (team_a_snapshot.recent_points_adjusted - 1.2) * 0.16, 0.72, 1.28)
    form_factor_b = _clamp(0.83 + (team_b_snapshot.recent_points_adjusted - 1.2) * 0.16, 0.72, 1.28)

    matchup_gap = elo_edge / MISMATCH_ELO_DIVISOR
    mismatch_boost_a = _clamp(1.0 + max(matchup_gap, 0.0) * 0.18, 1.0, 1.32)
    mismatch_boost_b = _clamp(1.0 + max(-matchup_gap, 0.0) * 0.18, 1.0, 1.32)
    suppression_a = _clamp(1.0 - max(-matchup_gap, 0.0) * 0.12, 0.72, 1.0)
    suppression_b = _clamp(1.0 - max(matchup_gap, 0.0) * 0.12, 0.55, 1.0)

    scoring_factor_a = _clamp(0.85 + team_a_snapshot.scoring_rate * 0.35, 0.85, 1.20)
    scoring_factor_b = _clamp(0.85 + team_b_snapshot.scoring_rate * 0.35, 0.85, 1.20)
    clean_sheet_pressure_a = _clamp(1.08 - team_b_snapshot.clean_sheet_rate * 0.14, 0.92, 1.08)
    clean_sheet_pressure_b = _clamp(1.08 - team_a_snapshot.clean_sheet_rate * 0.20, 0.80, 1.08)

    lambda_a = _clamp(
        base_goals
        * attack_a
        * defense_b
        * elo_factor_a
        * form_factor_a
        * mismatch_boost_a
        * suppression_a
        * scoring_factor_a
        * clean_sheet_pressure_a,
        MIN_EXPECTED_GOALS,
        MAX_EXPECTED_GOALS,
    )
    lambda_b = _clamp(
        base_goals
        * attack_b
        * defense_a
        * elo_factor_b
        * form_factor_b
        * mismatch_boost_b
        * scoring_factor_b
        * clean_sheet_pressure_b
        * suppression_b,
        MIN_EXPECTED_GOALS,
        MAX_EXPECTED_GOALS,
    )

    win_a, draw, win_b, best_score = _probability_matrix(lambda_a, lambda_b)

    return Prediction(
        team_a=team_a,
        team_b=team_b,
        match_date=str(match_ts.date()),
        neutral=neutral,
        team_a_snapshot=team_a_snapshot,
        team_b_snapshot=team_b_snapshot,
        expected_goals_a=lambda_a,
        expected_goals_b=lambda_b,
        win_prob_a=win_a,
        draw_prob=draw,
        win_prob_b=win_b,
        most_likely_score=best_score,
    )


def _actual_outcome(row: pd.Series) -> str:
    if row["home_score"] > row["away_score"]:
        return "home"
    if row["home_score"] < row["away_score"]:
        return "away"
    return "draw"


def _evaluate_single_match(row: pd.Series, prediction: Prediction) -> dict[str, float]:
    probs = {
        "home": prediction.win_prob_a,
        "draw": prediction.draw_prob,
        "away": prediction.win_prob_b,
    }
    actual = _actual_outcome(row)
    predicted = max(probs, key=probs.get)

    home_baseline = 1.0 if actual == "home" else 0.0
    elo_home = prediction.team_a_snapshot.elo + (0.0 if bool(row["neutral"]) else HOME_ADVANTAGE)
    elo_away = prediction.team_b_snapshot.elo
    elo_predicted = "home" if elo_home >= elo_away else "away"
    elo_baseline = 1.0 if elo_predicted == actual else 0.0

    target = {"home": 0.0, "draw": 0.0, "away": 0.0}
    target[actual] = 1.0

    return {
        "hit": float(predicted == actual),
        "confidence": probs[predicted],
        "log_loss": -math.log(max(probs[actual], 1e-9)),
        "brier": sum((probs[key] - target[key]) ** 2 for key in probs),
        "baseline_home_hit": home_baseline,
        "baseline_elo_hit": elo_baseline,
    }


def _finalize_backtest(metrics: list[dict[str, float]]) -> BacktestResult:
    evaluated = len(metrics)
    if evaluated == 0:
        raise ValueError("No hubo suficientes partidos historicos para evaluar el backtest.")

    return BacktestResult(
        matches_evaluated=evaluated,
        accuracy=sum(item["hit"] for item in metrics) / evaluated,
        log_loss=sum(item["log_loss"] for item in metrics) / evaluated,
        brier_score=sum(item["brier"] for item in metrics) / evaluated,
        avg_confidence=sum(item["confidence"] for item in metrics) / evaluated,
        baseline_home_accuracy=sum(item["baseline_home_hit"] for item in metrics) / evaluated,
        baseline_elo_accuracy=sum(item["baseline_elo_hit"] for item in metrics) / evaluated,
    )


def run_backtest(
    results: pd.DataFrame,
    matches_to_test: int = 200,
    min_history_matches: int = 500,
) -> BacktestResult:
    clean = results.loc[results["home_score"].notna() & results["away_score"].notna()].copy()
    clean = clean.sort_values("date").reset_index(drop=True)
    test_slice = clean.iloc[-matches_to_test:]

    metrics: list[dict[str, float]] = []

    for _, row in test_slice.iterrows():
        history = clean.loc[clean["date"] < row["date"]]
        if len(history) < min_history_matches:
            continue

        prediction = predict_match(
            results=history,
            team_a=row["home_team"],
            team_b=row["away_team"],
            match_date=str(row["date"].date()),
            neutral=bool(row["neutral"]),
        )
        metrics.append(_evaluate_single_match(row, prediction))

    return _finalize_backtest(metrics)


def run_rolling_backtest(
    results: pd.DataFrame,
    folds: int = 5,
    min_history_matches: int = 500,
) -> list[BacktestResult]:
    if folds <= 0:
        raise ValueError("folds debe ser mayor que 0.")

    clean = results.loc[results["home_score"].notna() & results["away_score"].notna()].copy()
    clean = clean.sort_values("date").reset_index(drop=True)
    candidate = clean.iloc[min_history_matches:].copy()
    if candidate.empty:
        raise ValueError("No hay suficientes partidos para ejecutar rolling backtest.")

    fold_size = len(candidate) // folds
    if fold_size == 0:
        raise ValueError("No hay suficientes partidos para repartir en la cantidad de folds solicitada.")

    reports: list[BacktestResult] = []
    start = 0
    for fold_index in range(folds):
        end = len(candidate) if fold_index == folds - 1 else start + fold_size
        fold_rows = candidate.iloc[start:end]
        if fold_rows.empty:
            break

        metrics: list[dict[str, float]] = []
        for row_index, row in fold_rows.iterrows():
            history = clean.iloc[:row_index]
            if len(history) < min_history_matches:
                continue

            prediction = predict_match(
                results=history,
                team_a=row["home_team"],
                team_b=row["away_team"],
                match_date=str(row["date"].date()),
                neutral=bool(row["neutral"]),
            )
            metrics.append(_evaluate_single_match(row, prediction))

        reports.append(_finalize_backtest(metrics))
        start = end

    return reports
