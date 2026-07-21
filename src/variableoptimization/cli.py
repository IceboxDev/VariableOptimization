"""Command-line interface.

    vopt [data options] <command> [command options]

Commands: train, preview, eval, mark-anomalies, refresh.
Data options select where game data comes from (Sheets, xlsx clone, cache).
Model references: ``deployed`` (the promoted model, default), ``latest``
(newest completed run), or an explicit path.
"""

import argparse
import datetime
import logging
import statistics
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

    train = commands.add_parser(
        "train", help="train a model as a tracked run under output/runs/"
    )
    train.add_argument("--best-of", type=int, default=100)
    train.add_argument("--seed", type=int, default=None, help="reproducible runs (workers=1)")
    train.add_argument("--workers", type=int, default=1, help="parallel training threads")
    train.add_argument("--epochs", type=int, default=1000, help="epochs per candidate")
    train.add_argument("--note", default=None, help="changelog note for this run")
    train.add_argument("--suffix", default=None, help="run-id suffix (e.g. 'demo')")
    train.add_argument(
        "--include-anomalies",
        action="store_true",
        help="train on anomaly-flagged games too (excluded by default)",
    )

    preview = commands.add_parser("preview", help="pretty-print all games by year")
    preview.add_argument(
        "--model",
        nargs="?",
        const="deployed",
        default=None,
        help="add prediction columns (deployed | latest | path)",
    )

    evaluate = commands.add_parser("eval", help="rank players by predicted team scores")
    evaluate.add_argument("--model", default="deployed")
    evaluate.add_argument("--min-games", type=int, default=None)
    evaluate.add_argument("--year", type=int, default=None)

    rank = commands.add_parser(
        "rank", help="derive player strengths from a model; dry-run unless --write"
    )
    rank.add_argument("--model", default="deployed")
    rank.add_argument("--samples", type=int, default=500, help="shared random teams per player")
    rank.add_argument("--seed", type=int, default=0, help="companion-set sampling seed")
    rank.add_argument(
        "--write", action="store_true",
        help="write the ranking into the sheet's BE/BF block",
    )

    anomalies = commands.add_parser(
        "mark-anomalies", help="detect statistical outliers; dry-run unless --write"
    )
    anomalies.add_argument("--model", default="deployed")
    anomalies.add_argument("--threshold", type=float, default=2.0, help="z-score cutoff")
    anomalies.add_argument(
        "--top", type=int, default=None,
        help="flag only the N most extreme outliers",
    )
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
    except (DataSourceError, FileNotFoundError, FileExistsError, RuntimeError) as error:
        log.error("%s", error)
        return 1


def run_train(args: argparse.Namespace, settings: DataSettings) -> int:
    """The tracked-run pipeline: run dir -> train -> gate -> manifest ->
    changelog -> latest. A failed run keeps only its log and stays invisible
    to `latest`, the changelog, and previous-run lookups."""
    import json

    from . import promotion, runs, snapshot as snapshot_module
    from .ai import ArtificialIntelligence, save_loss_plot

    output_dir = Path(constants.OUTPUT_DIR)
    root = runs.runs_root(output_dir)
    root.mkdir(parents=True, exist_ok=True)

    run_id = runs.generate_run_id(suffix=args.suffix)
    paths = runs.create_run_dir(root, run_id)

    handler = logging.FileHandler(paths.log_path)
    handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(message)s"))
    logging.getLogger().addHandler(handler)
    try:
        log.info("Run %s started", run_id)
        database = load_database(settings)

        config = {
            "best_of": args.best_of,
            "seed": args.seed,
            "workers": args.workers,
            "epochs": args.epochs,
            "source": settings.source,
            "include_anomalies": args.include_anomalies,
        }
        with open(paths.config_path, "w", encoding="utf-8") as handle:
            json.dump(config, handle, indent=2)
            handle.write("\n")

        intelligence = ArtificialIntelligence(
            database, include_anomalies=args.include_anomalies
        )
        result = intelligence.train(
            best_of=args.best_of,
            seed=args.seed,
            workers=args.workers,
            epochs=args.epochs,
        )
        result.predictor.save(paths.model_path, paths.roster_path)
        save_loss_plot(result.loss_history, result.best_loss, paths.plot_path)

        metrics = {
            "best_loss": result.best_loss,
            "mean_loss": statistics.mean(result.loss_history),
            "std_loss": statistics.pstdev(result.loss_history),
        }
        fingerprint = snapshot_module.fingerprint(database.snapshot)

        previous = runs.previous_manifest(root, run_id)
        delta_vs_prev = None
        if previous is not None:
            comparable = previous["dataset"]["fingerprint"] == fingerprint
            delta_vs_prev = {
                "prev_run_id": previous["run_id"],
                "best_loss": (
                    result.best_loss - previous["metrics"]["best_loss"]
                    if comparable
                    else None
                ),
                "comparable": comparable,
            }

        decision = promotion.decide_promotion(
            promotion.load_status_quo(output_dir), fingerprint, result.best_loss
        )
        if decision.promote:
            promotion.apply_promotion(paths, output_dir, run_id, fingerprint, metrics)

        sha, dirty = runs.git_state()
        manifest = {
            "run_id": run_id,
            "created_at": datetime.datetime.now(datetime.UTC).isoformat(),
            "git": {"sha": sha, "dirty": dirty},
            "note": args.note,
            "dataset": {
                "fingerprint": fingerprint,
                "games": len(database.games),
                "scored_games": len(database.scored_games),
                "excluded_anomalies": intelligence.excluded_anomalies,
                "players": len(database.players),
            },
            "config": config,
            "metrics": metrics,
            "delta_vs_prev": delta_vs_prev,
            "promoted": decision.promote,
            "promotion_reason": decision.reason,
        }
        runs.write_manifest(paths, manifest)
        runs.prepend_changelog(
            runs.changelog_path(output_dir), runs.format_changelog_entry(manifest)
        )
        runs.update_latest(root, run_id)

        stamp = "✅ promoted" if decision.promote else "❌ not promoted"
        print(
            f"Run {run_id}: best loss {result.best_loss:.0f} — "
            f"{stamp} ({decision.reason})\n"
            f"Artifacts: {paths.root}"
        )
        return 0
    finally:
        logging.getLogger().removeHandler(handler)
        handler.close()


def run_command(args: argparse.Namespace, settings: DataSettings) -> int:
    # Heavy imports happen per-command so data-only commands stay fast.
    if args.command == "refresh":
        refresh_settings = DataSettings(**{**vars(settings), "source": "sheets"})
        database = load_database(refresh_settings)
        print(
            f"Cache refreshed: {len(database.games)} games "
            f"({len(database.scored_games)} scored), {len(database.players)} players."
        )
        return 0

    if args.command == "train":
        return run_train(args, settings)

    database = load_database(settings)

    from . import reports
    from .ai import Predictor, resolve_model

    output_dir = Path(constants.OUTPUT_DIR)

    if args.command == "preview":
        predictor = None
        if args.model is not None:
            predictor = Predictor.load(resolve_model(args.model, output_dir))
        reports.preview_games(database, predictor)
        return 0

    if args.command == "eval":
        predictor = Predictor.load(resolve_model(args.model, output_dir))
        reports.evaluate_players(
            database, predictor, min_games=args.min_games, year=args.year
        )
        return 0

    if args.command == "rank":
        predictor = Predictor.load(resolve_model(args.model, output_dir))
        rankings = reports.rank_players(
            database, predictor, samples=args.samples, seed=args.seed
        )
        if not rankings:
            return 1
        if args.write:
            from .sources import SheetsSource

            source = SheetsSource(resolve_credentials(settings.credentials))
            source.save_rankings(rankings)
            print(f"Wrote {len(rankings)} player strengths to the sheet.")
        else:
            print("Dry run — pass --write to update the sheet.")
        return 0

    if args.command == "mark-anomalies":
        predictor = Predictor.load(resolve_model(args.model, output_dir))
        flags = reports.find_anomalies(
            database, predictor, threshold=args.threshold, top=args.top
        )
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
