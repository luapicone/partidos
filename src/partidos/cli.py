from __future__ import annotations

import argparse
from pathlib import Path

from .data import download_results, load_results
from .model import (
    calibrate_h2h_matches,
    calibrate_form_constants,
    calibrate_shrinkage,
    calibrate_time_decay,
    predict_match,
    run_ablation,
    run_backtest,
    run_rolling_backtest,
)
from .output import render_prediction, render_tiktok_script, write_probability_chart_svg


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="partidos",
        description="Predicciones de futbol internacional basadas en resultados historicos",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    update_parser = subparsers.add_parser("update-data", help="Descarga el dataset historico")
    update_parser.add_argument("--force", action="store_true", help="Vuelve a descargar el CSV")

    backtest_parser = subparsers.add_parser("backtest", help="Evalua historicamente el modelo")
    backtest_parser.add_argument(
        "--matches",
        type=int,
        default=200,
        help="Cantidad de partidos recientes a evaluar",
    )
    backtest_parser.add_argument(
        "--min-history",
        type=int,
        default=500,
        help="Minimo de partidos historicos previos requeridos",
    )
    backtest_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Actualiza el dataset antes del backtest",
    )

    rolling_parser = subparsers.add_parser(
        "backtest-rolling",
        help="Evalua el modelo en ventanas temporales consecutivas",
    )
    rolling_parser.add_argument("--folds", type=int, default=5)
    rolling_parser.add_argument("--min-history", type=int, default=500)
    rolling_parser.add_argument("--force-download", action="store_true")

    calibrate_parser = subparsers.add_parser(
        "calibrate",
        help="Encuentra el TIME_DECAY_HALF_LIFE optimo por log-loss",
    )
    calibrate_parser.add_argument(
        "--matches", type=int, default=300, help="Partidos a usar en cada evaluacion"
    )
    calibrate_parser.add_argument("--min-history", type=int, default=500)
    calibrate_parser.add_argument("--force-download", action="store_true")

    calibrate_form_parser = subparsers.add_parser(
        "calibrate-form",
        help="Encuentra las constantes de forma optimas por log-loss",
    )
    calibrate_form_parser.add_argument("--matches", type=int, default=300)
    calibrate_form_parser.add_argument("--min-history", type=int, default=500)
    calibrate_form_parser.add_argument("--force-download", action="store_true")

    shrinkage_parser = subparsers.add_parser(
        "calibrate-shrinkage",
        help="Encuentra los parametros optimos de shrinkage por log-loss",
    )
    shrinkage_parser.add_argument("--matches", type=int, default=300)
    shrinkage_parser.add_argument("--min-history", type=int, default=500)
    shrinkage_parser.add_argument("--force-download", action="store_true")

    h2h_parser = subparsers.add_parser(
        "calibrate-h2h",
        help="Encuentra el max_matches optimo para head-to-head",
    )
    h2h_parser.add_argument("--matches", type=int, default=300)
    h2h_parser.add_argument("--min-history", type=int, default=500)
    h2h_parser.add_argument("--force-download", action="store_true")

    ablation_parser = subparsers.add_parser(
        "ablation",
        help="Mide el impacto de cada factor del modelo por separado",
    )
    ablation_parser.add_argument("--matches", type=int, default=200)
    ablation_parser.add_argument("--min-history", type=int, default=500)
    ablation_parser.add_argument("--force-download", action="store_true")

    predict_parser = subparsers.add_parser("predict", help="Predice un partido")
    predict_parser.add_argument("--team-a", required=True, help="Equipo local o equipo A")
    predict_parser.add_argument("--team-b", required=True, help="Equipo visitante o equipo B")
    predict_parser.add_argument("--date", required=True, help="Fecha del partido en formato YYYY-MM-DD")
    predict_parser.add_argument(
        "--neutral",
        action="store_true",
        help="Marca el partido como sede neutral",
    )
    predict_parser.add_argument(
        "--force-download",
        action="store_true",
        help="Actualiza el dataset antes de calcular",
    )
    predict_parser.add_argument(
        "--tiktok-script",
        action="store_true",
        help="Agrega una salida corta lista para narrar en video",
    )
    predict_parser.add_argument(
        "--chart",
        action="store_true",
        help="Genera un grafico SVG con las probabilidades del partido",
    )
    predict_parser.add_argument(
        "--chart-output",
        help="Ruta del SVG a generar. Si no se indica, se guarda en charts/",
    )
    predict_parser.add_argument(
        "--with-lineup",
        action="store_true",
        help="Consulta alineaciones y lesiones en API-Football (requiere API_FOOTBALL_KEY)",
    )
    predict_parser.add_argument(
        "--use-xg",
        action="store_true",
        help="Usa xG historico real de API-Football como input (consume mas calls)",
    )

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "update-data":
        path = download_results(force=args.force)
        print(f"Dataset listo en: {path}")
        return

    if args.command == "backtest":
        results = load_results(force_download=args.force_download)
        report = run_backtest(
            results=results,
            matches_to_test=args.matches,
            min_history_matches=args.min_history,
        )
        print("Backtest del modelo")
        print(f"- Partidos evaluados: {report.matches_evaluated}")
        print(f"- Accuracy 1X2: {report.accuracy * 100:.2f}%")
        print(f"- Log loss: {report.log_loss:.4f}")
        print(f"- Brier score: {report.brier_score:.4f}")
        print(f"- Confianza promedio del pick: {report.avg_confidence * 100:.2f}%")
        print(f"--- Baselines de comparacion ---")
        print(f"- Baseline siempre local: {report.baseline_home_accuracy * 100:.2f}%")
        print(f"- Baseline Elo puro: {report.baseline_elo_accuracy * 100:.2f}%")
        print("--- Accuracy por torneo (min 5 partidos) ---")
        for torneo, acc in sorted(report.tournament_accuracy.items(), key=lambda x: -x[1]):
            print(f" {torneo}: {acc * 100:.1f}%")
        print("--- Accuracy por brecha de Elo ---")
        for rango in ["0-50", "51-100", "101-200", "201-350", "350+"]:
            if rango in report.elo_gap_accuracy:
                print(f" {rango}: {report.elo_gap_accuracy[rango] * 100:.1f}%")
        return

    if args.command == "backtest-rolling":
        results = load_results(force_download=args.force_download)
        reports = run_rolling_backtest(
            results=results,
            folds=args.folds,
            min_history_matches=args.min_history,
        )
        print(f"Rolling backtest ({args.folds} folds)")
        for index, report in enumerate(reports, start=1):
            print(
                f"Fold {index}: "
                f"accuracy={report.accuracy * 100:.2f}% "
                f"log_loss={report.log_loss:.4f} "
                f"brier={report.brier_score:.4f}"
            )
        average_accuracy = sum(report.accuracy for report in reports) / len(reports)
        average_log_loss = sum(report.log_loss for report in reports) / len(reports)
        average_brier = sum(report.brier_score for report in reports) / len(reports)
        print(
            f"Promedio: accuracy={average_accuracy * 100:.2f}% "
            f"log_loss={average_log_loss:.4f} "
            f"brier={average_brier:.4f}"
        )
        return

    if args.command == "calibrate":
        results = load_results(force_download=args.force_download)
        scores = calibrate_time_decay(
            results=results,
            matches_to_test=args.matches,
            min_history_matches=args.min_history,
        )
        print("Calibracion TIME_DECAY_HALF_LIFE_DAYS")
        print(f"{'Valor':>10} {'Log loss':>10}")
        for value, log_loss in sorted(scores.items()):
            marker = " <-- actual" if value == 540.0 else ""
            print(f"{value:>10.0f} {log_loss:>10.4f}{marker}")
        best = min(scores, key=scores.get)
        print(f"\nMejor valor encontrado: {best:.0f} dias (log_loss={scores[best]:.4f})")
        print("Para aplicarlo: edita TIME_DECAY_HALF_LIFE_DAYS en config.py")
        return

    if args.command == "calibrate-form":
        results = load_results(force_download=args.force_download)
        scores = calibrate_form_constants(
            results=results,
            matches_to_test=args.matches,
            min_history_matches=args.min_history,
        )
        best = min(scores, key=scores.get)
        current = (0.83, 1.2, 0.16)
        print("Calibracion constantes de forma (top 10 por log-loss)")
        print(f"{'base':>6} {'ref':>6} {'scale':>6} {'log_loss':>10}")
        for combo, ll in sorted(scores.items(), key=lambda x: x[1])[:10]:
            marker = " <-- actual" if combo == current else ""
            print(f"{combo[0]:>6.2f} {combo[1]:>6.2f} {combo[2]:>6.2f} {ll:>10.4f}{marker}")
        print(
            f"\nMejor combinacion: form_base={best[0]}, form_ref={best[1]}, form_scale={best[2]}"
        )
        print("Para aplicarlo: edita predict_match en model.py lineas 301-302")
        return

    if args.command == "calibrate-shrinkage":
        results = load_results(force_download=args.force_download)
        scores = calibrate_shrinkage(
            results=results,
            matches_to_test=args.matches,
            min_history_matches=args.min_history,
        )
        print("Calibracion shrinkage (top 10 por log-loss)")
        print(f"{'midpoint':>10} {'steepness':>10} {'log_loss':>10}")
        for (mid, steep), ll in sorted(scores.items(), key=lambda x: x[1])[:10]:
            marker = " <-- actual" if mid == 150.0 and steep == 0.018 else ""
            print(f"{mid:>10.0f} {steep:>10.3f} {ll:>10.4f}{marker}")
        best = min(scores, key=scores.get)
        print(f"\nMejor combinacion: midpoint={best[0]:.0f}, steepness={best[1]:.3f}")
        print("Para aplicarlo: edita _shrinkage_weight en model.py")
        return

    if args.command == "calibrate-h2h":
        results = load_results(force_download=args.force_download)
        scores = calibrate_h2h_matches(
            results=results,
            matches_to_test=args.matches,
            min_history_matches=args.min_history,
        )
        print("Calibracion h2h max_matches")
        print(f"{'max_matches':>12} {'log_loss':>10}")
        for candidate, ll in sorted(scores.items()):
            marker = " <-- actual" if candidate == 10 else ""
            print(f"{candidate:>12} {ll:>10.4f}{marker}")
        best = min(scores, key=scores.get)
        print(f"\nMejor valor: max_matches={best}")
        print("Para aplicarlo: edita _head_to_head_factor en model.py")
        return

    if args.command == "ablation":
        results = load_results(force_download=args.force_download)
        report = run_ablation(
            results=results,
            matches_to_test=args.matches,
            min_history_matches=args.min_history,
        )
        print(f"{'Experimento':<22} {'Accuracy':>9} {'Log loss':>10} {'Brier':>8}")
        print("-" * 53)
        for name, result in report.items():
            print(
                f"{name:<22} {result.accuracy * 100:>8.2f}% "
                f"{result.log_loss:>10.4f} {result.brier_score:>8.4f}"
            )
        return

    if args.command == "predict":
        results = load_results(force_download=args.force_download)
        prediction = predict_match(
            results=results,
            team_a=args.team_a,
            team_b=args.team_b,
            match_date=args.date,
            neutral=args.neutral,
            use_lineup=args.with_lineup,
            use_xg=args.use_xg,
        )
        print(render_prediction(prediction))
        if args.with_lineup:
            if prediction.lineup_available:
                print("Alineacion confirmada: datos de API-Football aplicados.")
            else:
                print("Alineacion no disponible aun: prediccion basada solo en historial.")
        if args.tiktok_script:
            print("\nGuion TikTok:\n")
            print(render_tiktok_script(prediction))
        if args.chart or args.chart_output:
            chart_path = write_probability_chart_svg(prediction, args.chart_output)
            print(f"\nGrafico SVG: {Path(chart_path).resolve()}")
        return


if __name__ == "__main__":
    main()
