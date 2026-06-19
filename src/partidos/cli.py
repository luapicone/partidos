from __future__ import annotations

import argparse
from pathlib import Path

from .data import download_results, load_results
from .model import predict_match, run_backtest, run_rolling_backtest
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

    if args.command == "predict":
        results = load_results(force_download=args.force_download)
        prediction = predict_match(
            results=results,
            team_a=args.team_a,
            team_b=args.team_b,
            match_date=args.date,
            neutral=args.neutral,
        )
        print(render_prediction(prediction))
        if args.tiktok_script:
            print("\nGuion TikTok:\n")
            print(render_tiktok_script(prediction))
        if args.chart or args.chart_output:
            chart_path = write_probability_chart_svg(prediction, args.chart_output)
            print(f"\nGrafico SVG: {Path(chart_path).resolve()}")
        return


if __name__ == "__main__":
    main()
