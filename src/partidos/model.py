from __future__ import annotations

import math
from dataclasses import dataclass

import pandas as pd

from .config import ELO_BASE, ELO_K, FORM_MATCHES, HOME_ADVANTAGE, POISSON_MAX_GOALS


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

        expected_home = _expected_from_elo(home_rating + HOME_ADVANTAGE, away_rating)
        expected_away = 1.0 - expected_home

        if row.home_score > row.away_score:
            actual_home, actual_away = 1.0, 0.0
        elif row.home_score < row.away_score:
            actual_home, actual_away = 0.0, 1.0
        else:
            actual_home = actual_away = 0.5

        goal_diff = abs(int(row.home_score) - int(row.away_score))
        multiplier = _goal_margin_multiplier(goal_diff) if goal_diff else 1.0
        adjustment = ELO_K * multiplier

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
    return matches.sort_values("date")


def build_team_snapshot(
    results: pd.DataFrame,
    ratings: dict[str, float],
    team: str,
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
            recent_goals_for_avg=1.0,
            recent_goals_against_avg=1.0,
        )

    recent = matches.tail(FORM_MATCHES)
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
        recent_points_per_match=float(recent["points"].mean()),
        recent_goals_for_avg=float(recent["goals_for"].mean()),
        recent_goals_against_avg=float(recent["goals_against"].mean()),
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
    team_a_snapshot = build_team_snapshot(history, ratings, team_a)
    team_b_snapshot = build_team_snapshot(history, ratings, team_b)

    base_goals = float((history["home_score"].mean() + history["away_score"].mean()) / 2.0)
    attack_a = _clamp(team_a_snapshot.recent_goals_for_avg / max(base_goals, 0.1), 0.45, 1.9)
    defense_a = _clamp(team_a_snapshot.recent_goals_against_avg / max(base_goals, 0.1), 0.45, 1.9)
    attack_b = _clamp(team_b_snapshot.recent_goals_for_avg / max(base_goals, 0.1), 0.45, 1.9)
    defense_b = _clamp(team_b_snapshot.recent_goals_against_avg / max(base_goals, 0.1), 0.45, 1.9)

    elo_edge = team_a_snapshot.elo - team_b_snapshot.elo + (0 if neutral else HOME_ADVANTAGE)
    elo_factor_a = _clamp(math.exp(elo_edge / 800.0), 0.75, 1.35)
    elo_factor_b = _clamp(math.exp(-elo_edge / 800.0), 0.75, 1.35)

    form_factor_a = _clamp(0.85 + (team_a_snapshot.recent_points_per_match - 1.2) * 0.18, 0.75, 1.25)
    form_factor_b = _clamp(0.85 + (team_b_snapshot.recent_points_per_match - 1.2) * 0.18, 0.75, 1.25)

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
