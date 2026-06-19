from __future__ import annotations

from .model import Prediction


def _pct(value: float) -> str:
    return f"{value * 100:.1f}%"


def render_prediction(prediction: Prediction) -> str:
    a = prediction.team_a_snapshot
    b = prediction.team_b_snapshot
    score_a, score_b = prediction.most_likely_score

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
        ]
    )


def render_tiktok_script(prediction: Prediction) -> str:
    score_a, score_b = prediction.most_likely_score

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
