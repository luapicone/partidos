from __future__ import annotations

import re
from pathlib import Path

from .model import Prediction


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_prediction(prediction: Prediction) -> str:
    a = prediction.team_a_snapshot
    b = prediction.team_b_snapshot
    score_a, score_b, _ = prediction.top_scores[0]

    if prediction.win_prob_a >= prediction.win_prob_b and prediction.win_prob_a >= prediction.draw_prob:
        headline = f"Gana {prediction.team_a}"
    elif prediction.win_prob_b >= prediction.draw_prob:
        headline = f"Gana {prediction.team_b}"
    else:
        headline = "Empate"

    return "\n".join(
        [
            f"Prediccion: {prediction.team_a} vs {prediction.team_b}",
            f"Fecha del partido: {prediction.match_date}",
            f"Sede neutral: {'si' if prediction.neutral else 'no'}",
            "",
            f"Resultado probable: {headline}",
            f"Marcador mas probable: {prediction.team_a} {score_a} - {score_b} {prediction.team_b}",
            "",
            "Top 3 marcadores:",
            f" 1. {prediction.team_a} {score_a} - {score_b} {prediction.team_b} ({prediction.top_scores[0][2]*100:.1f}%)",
            f" 2. {prediction.team_a} {prediction.top_scores[1][0]} - {prediction.top_scores[1][1]} {prediction.team_b} ({prediction.top_scores[1][2]*100:.1f}%)",
            f" 3. {prediction.team_a} {prediction.top_scores[2][0]} - {prediction.top_scores[2][1]} {prediction.team_b} ({prediction.top_scores[2][2]*100:.1f}%)",
            "",
            "Probabilidades:",
            f"- {prediction.team_a}: {_pct(prediction.win_prob_a)}",
            f"- Empate: {_pct(prediction.draw_prob)}",
            f"- {prediction.team_b}: {_pct(prediction.win_prob_b)}",
            "",
            "Senales del modelo:",
            f"- Elo {prediction.team_a}: {a.elo:.0f}",
            f"- Elo {prediction.team_b}: {b.elo:.0f}",
            f"- Goles esperados {prediction.team_a}: {prediction.expected_goals_a:.2f}",
            f"- Goles esperados {prediction.team_b}: {prediction.expected_goals_b:.2f}",
            f"- Forma reciente {prediction.team_a}: {a.recent_points_per_match:.2f} pts/partido",
            f"- Forma reciente {prediction.team_b}: {b.recent_points_per_match:.2f} pts/partido",
            f"- Forma ajustada rival {prediction.team_a}: {a.recent_points_adjusted:.2f}",
            f"- Forma ajustada rival {prediction.team_b}: {b.recent_points_adjusted:.2f}",
        ]
    )


def render_tiktok_script(prediction: Prediction) -> str:
    score_a, score_b, _ = prediction.top_scores[0]

    if prediction.win_prob_a >= prediction.win_prob_b and prediction.win_prob_a >= prediction.draw_prob:
        verdict = f"mi prediccion es que gana {prediction.team_a}"
    elif prediction.win_prob_b >= prediction.draw_prob:
        verdict = f"mi prediccion es que gana {prediction.team_b}"
    else:
        verdict = "mi prediccion es empate"

    return (
        f"Prediccion {prediction.team_a} vs {prediction.team_b}. "
        f"Segun el modelo, {verdict}. "
        f"La probabilidad marca {prediction.win_prob_a * 100:.0f}% para {prediction.team_a}, "
        f"{prediction.draw_prob * 100:.0f}% de empate y "
        f"{prediction.win_prob_b * 100:.0f}% para {prediction.team_b}. "
        f"El marcador mas probable es {prediction.team_a} {score_a} a {score_b} {prediction.team_b}. "
        f"Esto sale de rating Elo, forma reciente y promedio de goles a favor y en contra."
    )


def _slugify(value: str) -> str:
    value = value.lower().strip()
    value = re.sub(r"[^a-z0-9]+", "-", value)
    return value.strip("-") or "chart"


def render_probability_chart_svg(prediction: Prediction) -> str:
    probs = [
        (prediction.team_a, prediction.win_prob_a, "#1f6feb"),
        ("Empate", prediction.draw_prob, "#f59e0b"),
        (prediction.team_b, prediction.win_prob_b, "#16a34a"),
    ]
    width = 1080
    height = 1350
    bar_left = 250
    bar_width = 650
    bar_height = 84
    gap = 120
    start_y = 370

    bars = []
    for index, (label, prob, color) in enumerate(probs):
        y = start_y + index * gap
        fill_width = max(14, bar_width * prob)
        percent_text = f"{prob * 100:.1f}%"
        bars.append(
            f"""
            <text x="120" y="{y + 54}" font-size="42" font-weight="700" fill="#e5eefb">{label}</text>
            <rect x="{bar_left}" y="{y}" rx="24" ry="24" width="{bar_width}" height="{bar_height}" fill="#1f2937"/>
            <rect x="{bar_left}" y="{y}" rx="24" ry="24" width="{fill_width:.1f}" height="{bar_height}" fill="{color}"/>
            <text x="940" y="{y + 54}" text-anchor="end" font-size="44" font-weight="800" fill="#ffffff">{percent_text}</text>
            """
        )

    score_a, score_b, _ = prediction.top_scores[0]
    headline = f"{prediction.team_a} vs {prediction.team_b}"
    subtitle = f"Prediccion para {prediction.match_date}"
    footer = f"Marcador mas probable: {prediction.team_a} {score_a} - {score_b} {prediction.team_b}"

    return f"""<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">
  <defs>
    <linearGradient id="bg" x1="0%" y1="0%" x2="100%" y2="100%">
      <stop offset="0%" stop-color="#081120"/>
      <stop offset="100%" stop-color="#142642"/>
    </linearGradient>
  </defs>
  <rect width="100%" height="100%" fill="url(#bg)"/>
  <circle cx="930" cy="180" r="180" fill="#1d4ed8" opacity="0.18"/>
  <circle cx="150" cy="1130" r="220" fill="#16a34a" opacity="0.14"/>
  <text x="120" y="130" font-size="44" font-weight="700" fill="#93c5fd">MODELO DE PREDICCION</text>
  <text x="120" y="220" font-size="72" font-weight="900" fill="#ffffff">{headline}</text>
  <text x="120" y="285" font-size="34" font-weight="500" fill="#cbd5e1">{subtitle}</text>
  <text x="120" y="995" font-size="32" font-weight="600" fill="#93c5fd">Resultado mas probable</text>
  <text x="120" y="1055" font-size="58" font-weight="900" fill="#ffffff">{footer}</text>
  <text x="120" y="1170" font-size="28" font-weight="500" fill="#cbd5e1">Basado en Elo, forma reciente ajustada y peso por torneo</text>
  {''.join(bars)}
</svg>
"""


def write_probability_chart_svg(prediction: Prediction, output_path: str | None = None) -> Path:
    if output_path is None:
        filename = (
            f"{prediction.match_date}-"
            f"{_slugify(prediction.team_a)}-vs-{_slugify(prediction.team_b)}.svg"
        )
        path = Path("charts") / filename
    else:
        path = Path(output_path)

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_probability_chart_svg(prediction), encoding="utf-8")
    return path
