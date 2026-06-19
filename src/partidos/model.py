from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .config import (
    ELO_BASE,
    ELO_K,
    FORM_MATCHES,
    HOME_ADVANTAGE,
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


def _result_points(goals_for: int, goals_against: int) -> int:
    if goals_for > goals_against:
        return 3
    if goals_for == goals_against:
        return 1
    return 0


def _goal_margin_multiplier(goal_diff: int) -> float:
    return math.log(goal_diff + 1.0) * (2.2 / 2.2)


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
    attack_a = _clamp(team_a_snapshot.recent_goals_for_avg / max(base_goals, 0.1), 0.45, 1.9)
    defense_a = _clamp(team_a_snapshot.recent_goals_against_avg / max(base_goals, 0.1), 0.45, 1.9)
    attack_b = _clamp(team_b_snapshot.recent_goals_for_avg / max(base_goals, 0.1), 0.45, 1.9)
    defense_b = _clamp(team_b_snapshot.recent_goals_against_avg / max(base_goals, 0.1), 0.45, 1.9)

    elo_edge = team_a_snapshot.elo - team_b_snapshot.elo + (0 if neutral else HOME_ADVANTAGE)
    elo_factor_a = _clamp(math.exp(elo_edge / 800.0), 0.75, 1.35)
    elo_factor_b = _clamp(math.exp(-elo_edge / 800.0), 0.75, 1.35)

    form_factor_a = _clamp(0.83 + (team_a_snapshot.recent_points_adjusted - 1.2) * 0.16, 0.72, 1.28)
    form_factor_b = _clamp(0.83 + (team_b_snapshot.recent_points_adjusted - 1.2) * 0.16, 0.72, 1.28)

    lambda_a = _clamp(base_goals * attack_a * defense_b * elo_factor_a * form_factor_a, 0.2, 3.4)
    lambda_b = _clamp(base_goals * attack_b * defense_a * elo_factor_b * form_factor_b, 0.2, 3.4)

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


def run_backtest(
    results: pd.DataFrame,
    matches_to_test: int = 200,
    min_history_matches: int = 500,
) -> BacktestResult:
    clean = results.loc[results["home_score"].notna() & results["away_score"].notna()].copy()
    clean = clean.sort_values("date").reset_index(drop=True)
    test_slice = clean.iloc[-matches_to_test:]

    hits = 0
    log_loss_total = 0.0
    brier_total = 0.0
    confidence_total = 0.0
    evaluated = 0

    for row in test_slice.itertuples(index=False):
        history = clean.loc[clean["date"] < row.date]
        if len(history) < min_history_matches:
            continue

        prediction = predict_match(
            results=history,
            team_a=row.home_team,
            team_b=row.away_team,
            match_date=str(row.date.date()),
            neutral=bool(row.neutral),
        )

        probs = {
            "home": prediction.win_prob_a,
            "draw": prediction.draw_prob,
            "away": prediction.win_prob_b,
        }
        actual = "home" if row.home_score > row.away_score else "away" if row.home_score < row.away_score else "draw"
        predicted = max(probs, key=probs.get)

        hits += int(predicted == actual)
        confidence_total += probs[predicted]
        actual_prob = max(probs[actual], 1e-9)
        log_loss_total += -math.log(actual_prob)

        target = {"home": 0.0, "draw": 0.0, "away": 0.0}
        target[actual] = 1.0
        brier_total += sum((probs[key] - target[key]) ** 2 for key in probs)
        evaluated += 1

    if evaluated == 0:
        raise ValueError("No hubo suficientes partidos historicos para evaluar el backtest.")

    return BacktestResult(
        matches_evaluated=evaluated,
        accuracy=hits / evaluated,
        log_loss=log_loss_total / evaluated,
        brier_score=brier_total / evaluated,
        avg_confidence=confidence_total / evaluated,
    )
