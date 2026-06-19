from __future__ import annotations

import argparse

from .data import download_results, load_results
from .model import predict_match
from .output import render_prediction, render_tiktok_script


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="partidos",
        description="Predicciones de futbol internacional basadas en resultados historicos",
    )

    subparsers = parser.add_subparsers(dest="command", required=True)

    update_parser = subparsers.add_parser("update-data", help="Descarga el dataset historico")
    update_parser.add_argument("--force", action="store_true", help="Vuelve a descargar el CSV")

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

    return parser


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.command == "update-data":
        path = download_results(force=args.force)
        print(f"Dataset listo en: {path}")
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
        return


if __name__ == "__main__":
    main()
