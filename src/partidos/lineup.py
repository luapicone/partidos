from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime

import requests


API_BASE_URL = "https://v3.football.api-sports.io"


@dataclass
class PlayerAvailability:
    player_id: int
    name: str
    is_starter: bool
    is_injured: bool
    reason: str


@dataclass
class LineupData:
    fixture_id: int
    team_a: str
    team_b: str
    team_a_starters: int
    team_b_starters: int
    team_a_missing: int
    team_b_missing: int
    available: bool


def _normalize_name(value: str) -> str:
    return " ".join(str(value).strip().lower().split())


def _name_matches(expected: str, candidate: str) -> bool:
    left = _normalize_name(expected)
    right = _normalize_name(candidate)
    return bool(left and right and (left in right or right in left))


def _request_api(path: str, api_key: str, params: dict[str, object]) -> list[dict[str, object]]:
    if not api_key:
        return []

    try:
        response = requests.get(
            f"{API_BASE_URL}{path}",
            headers={"x-apisports-key": api_key},
            params=params,
            timeout=10,
        )
        payload = response.json()
    except (ValueError, requests.RequestException):
        return []

    if not isinstance(payload, dict):
        return []

    items = payload.get("response")
    if not isinstance(items, list):
        return []

    return [item for item in items if isinstance(item, dict)]


def _extract_team_name(item: dict[str, object]) -> str:
    team = item.get("team")
    if not isinstance(team, dict):
        return ""
    name = team.get("name")
    return str(name) if name is not None else ""


def _count_missing_players(entries: list[dict[str, object]]) -> int:
    total = 0
    for item in entries:
        player = item.get("player")
        if not isinstance(player, dict):
            continue
        reason_type = str(player.get("type", "")).strip().lower()
        if reason_type in {"injury", "suspension"}:
            total += 1
    return total


def _lookup_team_count(values: dict[str, int], team_name: str) -> int:
    direct = values.get(team_name)
    if direct is not None:
        return direct

    for candidate, count in values.items():
        if _name_matches(team_name, candidate):
            return count

    return 0


def _extract_team_id(item: dict[str, object]) -> int | None:
    team = item.get("team")
    if not isinstance(team, dict):
        return None
    team_id = team.get("id")
    if isinstance(team_id, int):
        return team_id
    if isinstance(team_id, float):
        return int(team_id)
    return None


def _extract_fixture_id(item: dict[str, object]) -> int | None:
    fixture = item.get("fixture")
    if not isinstance(fixture, dict):
        return None
    fixture_id = fixture.get("id")
    if isinstance(fixture_id, int):
        return fixture_id
    if isinstance(fixture_id, float):
        return int(fixture_id)
    return None


def _parse_xg_value(value: object) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def find_fixture_id(
    team_a: str,
    team_b: str,
    match_date: str,
    api_key: str,
) -> int | None:
    responses = _request_api(
        "/fixtures",
        api_key,
        {"date": match_date, "type": "International"},
    )
    if not responses:
        responses = _request_api("/fixtures", api_key, {"date": match_date})

    for item in responses:
        teams = item.get("teams")
        fixture = item.get("fixture")
        if not isinstance(teams, dict) or not isinstance(fixture, dict):
            continue
        home = teams.get("home")
        away = teams.get("away")
        if not isinstance(home, dict) or not isinstance(away, dict):
            continue

        home_name = str(home.get("name", ""))
        away_name = str(away.get("name", ""))
        same_order = _name_matches(team_a, home_name) and _name_matches(team_b, away_name)
        swapped_order = _name_matches(team_a, away_name) and _name_matches(team_b, home_name)
        if not same_order and not swapped_order:
            continue

        fixture_id = fixture.get("id")
        if isinstance(fixture_id, int):
            return fixture_id
        if isinstance(fixture_id, float):
            return int(fixture_id)

    return None


def fetch_lineup(
    fixture_id: int,
    team_a: str,
    team_b: str,
    api_key: str,
) -> LineupData:
    responses = _request_api("/fixtures/lineups", api_key, {"fixture": fixture_id})
    if not responses:
        return LineupData(
            fixture_id=fixture_id,
            team_a=team_a,
            team_b=team_b,
            team_a_starters=0,
            team_b_starters=0,
            team_a_missing=0,
            team_b_missing=0,
            available=False,
        )

    team_a_starters = 0
    team_b_starters = 0

    for item in responses:
        team_name = _extract_team_name(item)
        starters = item.get("startXI")
        starter_count = len(starters) if isinstance(starters, list) else 0

        if _name_matches(team_a, team_name):
            team_a_starters = starter_count
        elif _name_matches(team_b, team_name):
            team_b_starters = starter_count

    return LineupData(
        fixture_id=fixture_id,
        team_a=team_a,
        team_b=team_b,
        team_a_starters=team_a_starters,
        team_b_starters=team_b_starters,
        team_a_missing=0,
        team_b_missing=0,
        available=True,
    )


def fetch_injuries(
    fixture_id: int,
    api_key: str,
) -> dict[str, int]:
    responses = _request_api("/injuries", api_key, {"fixture": fixture_id})
    if not responses:
        return {}

    counts: dict[str, int] = {}
    for item in responses:
        team_name = _extract_team_name(item)
        if not team_name:
            continue
        counts[team_name] = counts.get(team_name, 0) + _count_missing_players([item])

    return {team_name: count for team_name, count in counts.items() if count > 0}


def get_lineup_factor(
    team_a: str,
    team_b: str,
    match_date: str,
    api_key: str,
    missing_penalty: float = 0.04,
    max_penalty: float = 0.20,
    use_xg: bool = False,
) -> tuple[float, float, bool, float | None, float | None]:
    xg_a = None
    xg_b = None
    if use_xg:
        xg_a, xg_b = get_xg_factors(team_a=team_a, team_b=team_b, match_date=match_date, api_key=api_key)

    fixture_id = find_fixture_id(team_a=team_a, team_b=team_b, match_date=match_date, api_key=api_key)
    if fixture_id is None:
        return 1.0, 1.0, False, xg_a, xg_b

    lineup = fetch_lineup(fixture_id=fixture_id, team_a=team_a, team_b=team_b, api_key=api_key)
    if not lineup.available:
        return 1.0, 1.0, False, xg_a, xg_b

    injuries = fetch_injuries(fixture_id=fixture_id, api_key=api_key)
    missing_a = _lookup_team_count(injuries, team_a) + max(0, 11 - lineup.team_a_starters)
    missing_b = _lookup_team_count(injuries, team_b) + max(0, 11 - lineup.team_b_starters)

    factor_a = max(1.0 - missing_a * missing_penalty, 1.0 - max_penalty)
    factor_b = max(1.0 - missing_b * missing_penalty, 1.0 - max_penalty)

    return max(factor_a, 0.80), max(factor_b, 0.80), True, xg_a, xg_b


def fetch_team_xg_history(
    team_name: str,
    api_key: str,
    season: int,
    league_id: int = 1,
    last_n: int = 8,
) -> float | None:
    team_responses = _request_api("/teams", api_key, {"name": team_name})
    team_id = None
    for item in team_responses:
        candidate_name = _extract_team_name(item)
        if _name_matches(team_name, candidate_name):
            team_id = _extract_team_id(item)
            if team_id is not None:
                break
    if team_id is None and team_responses:
        team_id = _extract_team_id(team_responses[0])
    if team_id is None:
        return None

    fixtures = _request_api(
        "/fixtures",
        api_key,
        {"team": team_id, "season": season, "league": league_id, "last": last_n},
    )

    values: list[float] = []
    for item in fixtures:
        fixture_id = _extract_fixture_id(item)
        if fixture_id is None:
            continue

        statistics_responses = _request_api("/fixtures/statistics", api_key, {"fixture": fixture_id})
        for stats_item in statistics_responses:
            candidate_name = _extract_team_name(stats_item)
            if not _name_matches(team_name, candidate_name):
                continue

            statistics = stats_item.get("statistics")
            if not isinstance(statistics, list):
                continue

            for stat in statistics:
                if not isinstance(stat, dict):
                    continue
                if str(stat.get("type", "")).strip().lower() != "expected_goals".lower():
                    continue
                xg_value = _parse_xg_value(stat.get("value"))
                if xg_value is not None:
                    values.append(xg_value)
                break
            if len(values) >= 5:
                break
        if len(values) >= 5:
            break

    if not values:
        return None

    return sum(values) / len(values)


def get_xg_factors(
    team_a: str,
    team_b: str,
    match_date: str,
    api_key: str,
) -> tuple[float | None, float | None]:
    match_ts = datetime.strptime(match_date, "%Y-%m-%d")
    season = match_ts.year if match_ts.month >= 6 else match_ts.year - 1
    xg_a = fetch_team_xg_history(team_name=team_a, api_key=api_key, season=season)
    xg_b = fetch_team_xg_history(team_name=team_b, api_key=api_key, season=season)
    return xg_a, xg_b
