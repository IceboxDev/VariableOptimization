"""Command-line interface.

    vopt [data options] <command> [command options]

Commands: train, preview, eval, mark-anomalies, refresh.
Data options select where game data comes from (Sheets, xlsx clone, cache).
"""

import argparse
import logging
import sys
from pathlib import Path

from . import constants
from .loader import DataSettings, DataSourceError, load_database, resolve_credentials

log = logging.getLogger(__name__)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="vopt",
        description="Pub-quiz score prediction: train models, preview games, "
        "evaluate players, mark anomalies.",
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    data = parser.add_argument_group("data source")
    data.add_argument(
        "--source",
        choices=("auto", "sheets", "xlsx", "cache"),
        default="auto",
        help="auto: fresh cache, else Google Sheets (default); "
        "xlsx: local spreadsheet clone; cache: cached snapshot regardless of age",
    )
    data.add_argument(
        "--xlsx",
        type=Path,
        default=Path(constants.XLSX_DEFAULT_PATH),
        help=f"path to the xlsx clone (default: {constants.XLSX_DEFAULT_PATH!r})",
    )
    data.add_argument(
        "--credentials",
        type=Path,
        default=None,
        help=f"service-account key (default: ${constants.CREDENTIALS_ENV_VAR} "
        f"or {constants.CREDENTIALS_GLOB})",
    )
    data.add_argument(
        "--cache",
        dest="cache_path",
        type=Path,
        default=Path(constants.CACHE_DEFAULT_PATH),
        help=f"snapshot cache path (default: {constants.CACHE_DEFAULT_PATH!r})",
    )
    data.add_argument(
        "--refresh",
        action="store_true",
        help="bypass the cache and re-fetch from Google Sheets",
    )

    commands = parser.add_subparsers(dest="command", required=True)

    train = commands.add_parser("train", help="train a model, keep the best of N runs")
    train.add_argument("--best-of", type=int, default=100)
    train.add_argument("--seed", type=int, default=None, help="reproducible runs (workers=1)")
    train.add_argument("--workers", type=int, default=1, help="parallel training threads")

    preview = commands.add_parser("preview", help="pretty-print all games by year")
    preview.add_argument(
        "--model",
        nargs="?",
        const="latest",
        default=None,
        help="add prediction columns ('latest' or a model path)",
    )

    evaluate = commands.add_parser("eval", help="rank players by predicted team scores")
    evaluate.add_argument("--model", default="latest")
    evaluate.add_argument("--min-games", type=int, default=None)
    evaluate.add_argument("--year", type=int, default=None)

    anomalies = commands.add_parser(
        "mark-anomalies", help="detect statistical outliers; dry-run unless --write"
    )
    anomalies.add_argument("--model", default="latest")
    anomalies.add_argument("--threshold", type=float, default=2.0, help="z-score cutoff")
    anomalies.add_argument(
        "--write", action="store_true", help="write flags back to the Google Sheet"
    )

    commands.add_parser("refresh", help="force-refresh the snapshot cache from Sheets")

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s" if args.verbose else "%(message)s",
    )

    settings = DataSettings(
        source=args.source,
        xlsx_path=args.xlsx,
        credentials=args.credentials,
        cache_path=args.cache_path,
        refresh=args.refresh,
    )

    try:
        return run_command(args, settings)
    except (DataSourceError, FileNotFoundError, RuntimeError) as error:
        log.error("%s", error)
        return 1


def run_command(args: argparse.Namespace, settings: DataSettings) -> int:
    # Imported here so `vopt refresh` etc. don't pay the torch import cost.
    if args.command == "refresh":
        refresh_settings = DataSettings(**{**vars(settings), "source": "sheets"})
        database = load_database(refresh_settings)
        print(
            f"Cache refreshed: {len(database.games)} games "
            f"({len(database.scored_games)} scored), {len(database.players)} players."
        )
        return 0

    database = load_database(settings)

    from .ai import ArtificialIntelligence
    from . import reports

    if args.command == "train":
        ai = ArtificialIntelligence(database)
        model_path, best_loss = ai.train(
            best_of=args.best_of, seed=args.seed, workers=args.workers
        )
        print(f"Best loss {best_loss:.0f} — saved {model_path}")
        return 0

    if args.command == "preview":
        ai = None
        if args.model is not None:
            ai = ArtificialIntelligence(database)
            ai.load(args.model)
        reports.preview_games(database, ai)
        return 0

    if args.command == "eval":
        ai = ArtificialIntelligence(database)
        ai.load(args.model)
        reports.evaluate_players(
            database, ai, min_games=args.min_games, year=args.year
        )
        return 0

    if args.command == "mark-anomalies":
        ai = ArtificialIntelligence(database)
        ai.load(args.model)
        flags = reports.find_anomalies(database, ai, threshold=args.threshold)
        if args.write:
            from .sources import SheetsSource

            source = SheetsSource(resolve_credentials(settings.credentials))
            source.save_anomalies(flags)
            print(f"Wrote {sum(flags.values())} anomaly flags to the sheet.")
        else:
            print("Dry run — pass --write to update the sheet.")
        return 0

    raise AssertionError(f"Unhandled command {args.command!r}")


if __name__ == "__main__":
    sys.exit(main())
