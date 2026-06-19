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
    shrinkage_weight: float


@dataclass
class BacktestResult:
    matches_evaluated: int
    accuracy: float
    log_loss: float
    brier_score: float
    avg_confidence: float
    baseline_home_accuracy: float
    baseline_elo_accuracy: float
    tournament_accuracy: dict[str, float]
    elo_gap_accuracy: dict[str, float]


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


def _build_team_snapshot_from_matches(
    matches: pd.DataFrame,
    ratings: dict[str, float],
    team: str,
    reference_date: pd.Timestamp,
) -> TeamSnapshot:
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


def build_team_snapshot(
    results: pd.DataFrame,
    ratings: dict[str, float],
    team: str,
    reference_date: pd.Timestamp,
) -> TeamSnapshot:
    matches = _team_matches(results, team)
    return _build_team_snapshot_from_matches(matches, ratings, team, reference_date)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _poisson_probability(lmbda: float, goals: int) -> float:
    return math.exp(-lmbda) * (lmbda**goals) / math.factorial(goals)


def _dixon_coles_tau(
    goals_a: int,
    goals_b: int,
    lambda_a: float,
    lambda_b: float,
    rho: float,
) -> float:
    if goals_a == 0 and goals_b == 0:
        tau = 1.0 - lambda_a * lambda_b * rho
        return max(tau, 1e-6)
    if goals_a == 1 and goals_b == 0:
        return 1.0 + lambda_b * rho
    if goals_a == 0 and goals_b == 1:
        return 1.0 + lambda_a * rho
    if goals_a == 1 and goals_b == 1:
        return 1.0 - rho
    return 1.0


def _probability_matrix(
    lambda_a: float, lambda_b: float, rho: float = 0.1
) -> tuple[float, float, float, tuple[int, int]]:
    win_a = 0.0
    draw = 0.0
    win_b = 0.0
    best_prob = -1.0
    best_score = (0, 0)

    for goals_a in range(POISSON_MAX_GOALS + 1):
        for goals_b in range(POISSON_MAX_GOALS + 1):
            prob = (
                _poisson_probability(lambda_a, goals_a)
                * _poisson_probability(lambda_b, goals_b)
                * _dixon_coles_tau(goals_a, goals_b, lambda_a, lambda_b, rho)
            )
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


def _shrinkage_weight(elo_gap: float, midpoint: float = 150.0, steepness: float = 0.018) -> float:
    return 1.0 / (1.0 + math.exp(-steepness * (elo_gap - midpoint)))


def predict_match(
    results: pd.DataFrame,
    team_a: str,
    team_b: str,
    match_date: str,
    neutral: bool = False,
    form_base: float = 0.75,
    form_ref: float = 1.5,
    form_scale: float = 0.10,
    rho: float = 0.1,
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

    form_factor_a = _clamp(
        form_base + (team_a_snapshot.recent_points_adjusted - form_ref) * form_scale, 0.72, 1.28
    )
    form_factor_b = _clamp(
        form_base + (team_b_snapshot.recent_points_adjusted - form_ref) * form_scale, 0.72, 1.28
    )

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

    win_a, draw, win_b, best_score = _probability_matrix(lambda_a, lambda_b, rho=rho)
    elo_gap = abs(elo_edge)
    weight = _shrinkage_weight(elo_gap)
    neutral_prob = 1.0 / 3.0
    win_a = weight * win_a + (1.0 - weight) * neutral_prob
    draw = weight * draw + (1.0 - weight) * neutral_prob
    win_b = weight * win_b + (1.0 - weight) * neutral_prob

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
        shrinkage_weight=weight,
    )


def _actual_outcome(row: pd.Series) -> str:
    if row["home_score"] > row["away_score"]:
        return "home"
    if row["home_score"] < row["away_score"]:
        return "away"
    return "draw"


def _elo_gap_bucket(elo_gap: float) -> str:
    if elo_gap <= 50:
        return "0-50"
    if elo_gap <= 100:
        return "51-100"
    if elo_gap <= 200:
        return "101-200"
    if elo_gap <= 350:
        return "201-350"
    return "350+"


def _evaluate_single_match(row: pd.Series, prediction: Prediction) -> dict[str, float | str]:
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
    elo_gap = abs(
        prediction.team_a_snapshot.elo
        - prediction.team_b_snapshot.elo
        + (HOME_ADVANTAGE if not row["neutral"] else 0.0)
    )

    target = {"home": 0.0, "draw": 0.0, "away": 0.0}
    target[actual] = 1.0

    return {
        "hit": float(predicted == actual),
        "confidence": probs[predicted],
        "log_loss": -math.log(max(probs[actual], 1e-9)),
        "brier": sum((probs[key] - target[key]) ** 2 for key in probs),
        "baseline_home_hit": home_baseline,
        "baseline_elo_hit": elo_baseline,
        "tournament": str(row["tournament"]),
        "elo_gap": elo_gap,
    }


def _finalize_backtest(metrics: list[dict[str, float | str]]) -> BacktestResult:
    evaluated = len(metrics)
    if evaluated == 0:
        raise ValueError("No hubo suficientes partidos historicos para evaluar el backtest.")

    tournament_totals: dict[str, list[float]] = {}
    elo_gap_totals: dict[str, list[float]] = {
        "0-50": [],
        "51-100": [],
        "101-200": [],
        "201-350": [],
        "350+": [],
    }

    for item in metrics:
        tournament = str(item["tournament"])
        tournament_totals.setdefault(tournament, []).append(float(item["hit"]))
        elo_gap_totals[_elo_gap_bucket(float(item["elo_gap"]))].append(float(item["hit"]))

    tournament_accuracy = {
        tournament: sum(values) / len(values)
        for tournament, values in tournament_totals.items()
        if len(values) >= 5
    }
    elo_gap_accuracy = {
        bucket: sum(values) / len(values)
        for bucket, values in elo_gap_totals.items()
        if values
    }

    return BacktestResult(
        matches_evaluated=evaluated,
        accuracy=sum(float(item["hit"]) for item in metrics) / evaluated,
        log_loss=sum(float(item["log_loss"]) for item in metrics) / evaluated,
        brier_score=sum(float(item["brier"]) for item in metrics) / evaluated,
        avg_confidence=sum(float(item["confidence"]) for item in metrics) / evaluated,
        baseline_home_accuracy=sum(float(item["baseline_home_hit"]) for item in metrics) / evaluated,
        baseline_elo_accuracy=sum(float(item["baseline_elo_hit"]) for item in metrics) / evaluated,
        tournament_accuracy=tournament_accuracy,
        elo_gap_accuracy=elo_gap_accuracy,
    )


def run_backtest(
    results: pd.DataFrame,
    matches_to_test: int = 200,
    min_history_matches: int = 500,
    form_base: float = 0.75,
    form_ref: float = 1.5,
    form_scale: float = 0.10,
    rho: float = 0.1,
) -> BacktestResult:
    clean = results.loc[results["home_score"].notna() & results["away_score"].notna()].copy()
    clean = clean.sort_values("date").reset_index(drop=True)
    test_slice = clean.iloc[-matches_to_test:]

    metrics: list[dict[str, float | str]] = []

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
            form_base=form_base,
            form_ref=form_ref,
            form_scale=form_scale,
            rho=rho,
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

        metrics: list[dict[str, float | str]] = []
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


def _build_elo_cache(
    clean: pd.DataFrame,
    test_slice: pd.DataFrame,
    min_history_matches: int,
) -> dict[int, dict[str, float]]:
    target_indices = set(test_slice.index.tolist())
    elo_cache: dict[int, dict[str, float]] = {}
    ratings: dict[str, float] = {}

    for row_index, row in clean.iterrows():
        if row_index in target_indices and row_index >= min_history_matches:
            elo_cache[row_index] = ratings.copy()

        home = row["home_team"]
        away = row["away_team"]
        ratings.setdefault(home, ELO_BASE)
        ratings.setdefault(away, ELO_BASE)

        home_rating = ratings[home]
        away_rating = ratings[away]

        home_advantage = 0.0 if bool(row["neutral"]) else HOME_ADVANTAGE
        expected_home = _expected_from_elo(home_rating + home_advantage, away_rating)
        expected_away = 1.0 - expected_home

        if row["home_score"] > row["away_score"]:
            actual_home, actual_away = 1.0, 0.0
        elif row["home_score"] < row["away_score"]:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home = actual_away = 0.5

        goal_diff = abs(int(row["home_score"]) - int(row["away_score"]))
        multiplier = _goal_margin_multiplier(goal_diff) if goal_diff else 1.0
        adjustment = ELO_K * multiplier * tournament_weight(row["tournament"])

        ratings[home] = home_rating + adjustment * (actual_home - expected_home)
        ratings[away] = away_rating + adjustment * (actual_away - expected_away)

    return elo_cache


def _build_calibration_contexts(
    clean: pd.DataFrame,
    test_slice: pd.DataFrame,
    elo_cache: dict[int, dict[str, float]],
    min_history_matches: int,
) -> list[dict[str, object]]:
    contexts: list[dict[str, object]] = []
    teams = set(test_slice["home_team"].tolist()) | set(test_slice["away_team"].tolist())
    team_matches_cache = {team: _team_matches(clean, team) for team in teams}

    for row_index, row in test_slice.iterrows():
        if row_index not in elo_cache:
            continue

        history = clean.loc[clean["date"] < row["date"]]
        if len(history) < min_history_matches:
            continue

        ratings = elo_cache[row_index]
        match_ts = row["date"]
        team_a_matches = team_matches_cache[row["home_team"]].loc[
            team_matches_cache[row["home_team"]]["date"] < match_ts
        ]
        team_b_matches = team_matches_cache[row["away_team"]].loc[
            team_matches_cache[row["away_team"]]["date"] < match_ts
        ]
        team_a_snapshot = _build_team_snapshot_from_matches(
            team_a_matches, ratings, row["home_team"], match_ts
        )
        team_b_snapshot = _build_team_snapshot_from_matches(
            team_b_matches, ratings, row["away_team"], match_ts
        )
        base_goals = float((history["home_score"].mean() + history["away_score"].mean()) / 2.0)
        attack_a = _clamp(team_a_snapshot.recent_goals_for_adjusted / max(base_goals, 0.1), 0.42, 2.35)
        defense_a = _clamp(team_a_snapshot.recent_goals_against_adjusted / max(base_goals, 0.1), 0.35, 2.0)
        attack_b = _clamp(team_b_snapshot.recent_goals_for_adjusted / max(base_goals, 0.1), 0.42, 2.35)
        defense_b = _clamp(team_b_snapshot.recent_goals_against_adjusted / max(base_goals, 0.1), 0.35, 2.0)
        elo_edge = team_a_snapshot.elo - team_b_snapshot.elo + (0 if bool(row["neutral"]) else HOME_ADVANTAGE)
        elo_factor_a = _clamp(math.exp(elo_edge / 700.0), 0.68, 1.65)
        elo_factor_b = _clamp(math.exp(-elo_edge / 700.0), 0.55, 1.45)
        matchup_gap = elo_edge / MISMATCH_ELO_DIVISOR
        mismatch_boost_a = _clamp(1.0 + max(matchup_gap, 0.0) * 0.18, 1.0, 1.32)
        mismatch_boost_b = _clamp(1.0 + max(-matchup_gap, 0.0) * 0.18, 1.0, 1.32)
        suppression_a = _clamp(1.0 - max(-matchup_gap, 0.0) * 0.12, 0.72, 1.0)
        suppression_b = _clamp(1.0 - max(matchup_gap, 0.0) * 0.12, 0.55, 1.0)
        scoring_factor_a = _clamp(0.85 + team_a_snapshot.scoring_rate * 0.35, 0.85, 1.20)
        scoring_factor_b = _clamp(0.85 + team_b_snapshot.scoring_rate * 0.35, 0.85, 1.20)
        clean_sheet_pressure_a = _clamp(1.08 - team_b_snapshot.clean_sheet_rate * 0.14, 0.92, 1.08)
        clean_sheet_pressure_b = _clamp(1.08 - team_a_snapshot.clean_sheet_rate * 0.20, 0.80, 1.08)
        elo_gap = abs(elo_edge)
        shrinkage_weight = _shrinkage_weight(elo_gap)

        contexts.append(
            {
                "row": row,
                "team_a_snapshot": team_a_snapshot,
                "team_b_snapshot": team_b_snapshot,
                "base_goals": base_goals,
                "attack_a": attack_a,
                "defense_a": defense_a,
                "attack_b": attack_b,
                "defense_b": defense_b,
                "elo_edge": elo_edge,
                "elo_factor_a": elo_factor_a,
                "elo_factor_b": elo_factor_b,
                "mismatch_boost_a": mismatch_boost_a,
                "mismatch_boost_b": mismatch_boost_b,
                "suppression_a": suppression_a,
                "suppression_b": suppression_b,
                "scoring_factor_a": scoring_factor_a,
                "scoring_factor_b": scoring_factor_b,
                "clean_sheet_pressure_a": clean_sheet_pressure_a,
                "clean_sheet_pressure_b": clean_sheet_pressure_b,
                "shrinkage_weight": shrinkage_weight,
            }
        )

    return contexts


def _evaluate_with_cached_elo(
    clean: pd.DataFrame,
    test_slice: pd.DataFrame,
    elo_cache: dict[int, dict[str, float]],
    min_history_matches: int,
    matches_to_test: int,
    form_base: float,
    form_ref: float,
    form_scale: float,
    rho: float = 0.1,
) -> BacktestResult:
    metrics: list[dict[str, float | str]] = []
    contexts = _build_calibration_contexts(clean, test_slice, elo_cache, min_history_matches)
    return _evaluate_with_cached_contexts(
        contexts=contexts,
        form_base=form_base,
        form_ref=form_ref,
        form_scale=form_scale,
        rho=rho,
    )


def _evaluate_with_cached_contexts(
    contexts: list[dict[str, object]],
    form_base: float,
    form_ref: float,
    form_scale: float,
    rho: float = 0.1,
) -> BacktestResult:
    metrics: list[dict[str, float | str]] = []

    for context in contexts:
        row = context["row"]
        team_a_snapshot = context["team_a_snapshot"]
        team_b_snapshot = context["team_b_snapshot"]
        base_goals = float(context["base_goals"])
        match_ts = row["date"]
        attack_a = float(context["attack_a"])
        defense_a = float(context["defense_a"])
        attack_b = float(context["attack_b"])
        defense_b = float(context["defense_b"])
        elo_edge = float(context["elo_edge"])
        elo_factor_a = float(context["elo_factor_a"])
        elo_factor_b = float(context["elo_factor_b"])

        form_factor_a = _clamp(
            form_base + (team_a_snapshot.recent_points_adjusted - form_ref) * form_scale, 0.72, 1.28
        )
        form_factor_b = _clamp(
            form_base + (team_b_snapshot.recent_points_adjusted - form_ref) * form_scale, 0.72, 1.28
        )
        mismatch_boost_a = float(context["mismatch_boost_a"])
        mismatch_boost_b = float(context["mismatch_boost_b"])
        suppression_a = float(context["suppression_a"])
        suppression_b = float(context["suppression_b"])
        scoring_factor_a = float(context["scoring_factor_a"])
        scoring_factor_b = float(context["scoring_factor_b"])
        clean_sheet_pressure_a = float(context["clean_sheet_pressure_a"])
        clean_sheet_pressure_b = float(context["clean_sheet_pressure_b"])

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

        win_a, draw, win_b, best_score = _probability_matrix(lambda_a, lambda_b, rho=rho)
        weight = float(context["shrinkage_weight"])
        neutral_prob = 1.0 / 3.0
        win_a = weight * win_a + (1.0 - weight) * neutral_prob
        draw = weight * draw + (1.0 - weight) * neutral_prob
        win_b = weight * win_b + (1.0 - weight) * neutral_prob

        prediction = Prediction(
            team_a=row["home_team"],
            team_b=row["away_team"],
            match_date=str(match_ts.date()),
            neutral=bool(row["neutral"]),
            team_a_snapshot=team_a_snapshot,
            team_b_snapshot=team_b_snapshot,
            expected_goals_a=lambda_a,
            expected_goals_b=lambda_b,
            win_prob_a=win_a,
            draw_prob=draw,
            win_prob_b=win_b,
            most_likely_score=best_score,
            shrinkage_weight=weight,
        )
        metrics.append(_evaluate_single_match(row, prediction))

    return _finalize_backtest(metrics)


def calibrate_time_decay(
    results: pd.DataFrame,
    candidates: list[float] | None = None,
    matches_to_test: int = 300,
    min_history_matches: int = 500,
) -> dict[float, float]:
    candidate_values = candidates or [180.0, 270.0, 360.0, 450.0, 540.0, 630.0, 720.0]
    original_value = TIME_DECAY_HALF_LIFE_DAYS
    scores: dict[float, float] = {}

    for candidate in candidate_values:
        try:
            globals()["TIME_DECAY_HALF_LIFE_DAYS"] = candidate
            report = run_backtest(
                results=results,
                matches_to_test=matches_to_test,
                min_history_matches=min_history_matches,
            )
            scores[candidate] = report.log_loss
        finally:
            globals()["TIME_DECAY_HALF_LIFE_DAYS"] = original_value

    return scores


def calibrate_form_constants(
    results: pd.DataFrame,
    matches_to_test: int = 300,
    min_history_matches: int = 500,
) -> dict[tuple[float, float, float, float], float]:
    form_bases = [0.75, 0.80, 0.83, 0.87, 0.90]
    form_refs = [1.0, 1.1, 1.2, 1.3, 1.5]
    form_scales = [0.10, 0.13, 0.16, 0.20, 0.25]
    rhos = [0.05, 0.10, 0.15, 0.20]
    clean = (
        results.loc[results["home_score"].notna() & results["away_score"].notna()]
        .copy()
        .sort_values("date")
        .reset_index(drop=True)
    )

    test_slice = clean.iloc[-matches_to_test:]
    elo_cache = _build_elo_cache(clean, test_slice, min_history_matches)
    contexts = _build_calibration_contexts(clean, test_slice, elo_cache, min_history_matches)

    scores: dict[tuple[float, float, float, float], float] = {}
    for form_base in form_bases:
        for form_ref in form_refs:
            for form_scale in form_scales:
                for rho in rhos:
                    report = _evaluate_with_cached_contexts(
                        contexts=contexts,
                        form_base=form_base,
                        form_ref=form_ref,
                        form_scale=form_scale,
                        rho=rho,
                    )
                    scores[(form_base, form_ref, form_scale, rho)] = report.log_loss

    best = min(scores, key=scores.get)
    print("Calibracion constantes de forma y rho (top 10 por log-loss)")
    print(f"{'base':>6} {'ref':>6} {'scale':>6} {'rho':>6} {'log_loss':>10}")
    for combo, log_loss in sorted(scores.items(), key=lambda item: item[1])[:10]:
        print(
            f"{combo[0]:>6.2f} {combo[1]:>6.2f} {combo[2]:>6.2f} {combo[3]:>6.2f} {log_loss:>10.4f}"
        )
    print(
        "\nMejor combinacion: "
        f"form_base={best[0]}, form_ref={best[1]}, form_scale={best[2]}, rho={best[3]}"
    )

    return scores
