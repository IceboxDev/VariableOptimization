"""User-facing reports: game preview tables, player rankings, anomaly detection.

All reports predict through a :class:`Predictor` and therefore work with any
trained model, including one older than the live roster — games involving
players the model never saw show no prediction instead of a wrong one.
"""

import datetime
import itertools
import logging
import math
from collections import defaultdict

import numpy
import tqdm
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import constants
from .ai import Predictor
from .database import Database
from .domain import Game

log = logging.getLogger(__name__)

console = Console()


def preview_games(database: Database, predictor: Predictor | None = None) -> None:
    """Print one panel per year. Column widths are computed globally so every
    panel lines up."""
    headers = ["#", "Date", "Score", "Anomaly", "Players"]
    if predictor is not None:
        headers += ["Prediction", "Δ(Pred)"]
    widths = {header: len(header) for header in headers}

    games_by_year: dict[str, list[Game]] = defaultdict(list)
    for game in database.games:
        games_by_year[str(game.date.year) if game.date else "Unknown"].append(game)

    rows_by_year: dict[str, list[dict[str, str]]] = {}
    for year, games in games_by_year.items():
        games = sorted(
            games,
            key=lambda game: (
                game.date or datetime.date.min,
                game.score if game.score is not None else -1,
            ),
        )
        rows = []
        for index, game in enumerate(games, start=1):
            row = {
                "#": str(index),
                "Date": game.date.isoformat() if game.date else "-",
                "Score": str(game.score) if game.has_score() else "-",
                "Anomaly": "✅" if game.is_anomaly else "",
                "Players": ", ".join(player.name for player in game.players),
            }
            if predictor is not None:
                prediction = predictor.infer(game)
                if prediction is None:
                    row["Prediction"] = "-"
                    row["Δ(Pred)"] = "-"
                else:
                    row["Prediction"] = f"{prediction:.2f}"
                    if game.has_score():
                        delta = prediction - game.score
                        color = "red" if delta > 0 else "green" if delta < 0 else None
                        plain = f"{delta:+.2f}"
                        row["Δ(Pred)"] = (
                            f"[{color}]{plain}[/{color}]" if color else plain
                        )
                        # Width bookkeeping must use plain text, not markup.
                        widths["Δ(Pred)"] = max(widths["Δ(Pred)"], len(plain))
                    else:
                        row["Δ(Pred)"] = "-"
            for header in headers:
                if header != "Δ(Pred)":
                    widths[header] = max(widths[header], len(row[header]))
            rows.append(row)
        rows_by_year[year] = rows

    for year in sorted(rows_by_year):
        table = Table(box=box.SIMPLE_HEAVY, show_lines=False)
        table.add_column("#", justify="right", style="bold", width=widths["#"], no_wrap=True)
        table.add_column("Date", width=widths["Date"], no_wrap=True)
        table.add_column("Score", justify="right", width=widths["Score"], no_wrap=True)
        table.add_column("Anomaly", justify="center", width=widths["Anomaly"], no_wrap=True)
        table.add_column("Players", width=widths["Players"])
        if predictor is not None:
            table.add_column("Prediction", justify="right", width=widths["Prediction"], no_wrap=True)
            table.add_column("Δ(Pred)", justify="right", width=widths["Δ(Pred)"], no_wrap=True)

        for row in rows_by_year[year]:
            table.add_row(*(row[header] for header in headers))
        console.print(Panel.fit(table, title=f"[bold cyan]Games in {year}[/bold cyan]"))


def evaluate_players(
    database: Database,
    predictor: Predictor,
    min_games: int | None = None,
    year: int | None = None,
    team_size: int = constants.TEAM_SIZE,
    max_teams: int = constants.EVAL_MAX_TEAMS,
) -> None:
    """Rank players by their mean predicted score over all possible teams.

    Teams are enumerated combinatorially, so the eligible roster must be
    narrow — an unfiltered roster produces billions of teams and is refused.
    Inference runs in batches (one forward pass per EVAL_BATCH_SIZE teams).
    """

    def relevant_games(player) -> list[Game]:
        games = player.scored_games
        if year is not None:
            games = [game for game in games if game.date and game.date.year == year]
        return games

    eligible = [
        player
        for player in database.players.values()
        if min_games is None or len(relevant_games(player)) >= min_games
    ]

    outside_roster = [p.name for p in eligible if not predictor.known(p)]
    if outside_roster:
        log.warning(
            "Excluding %d player(s) unknown to the trained roster: %s",
            len(outside_roster), ", ".join(sorted(outside_roster)),
        )
        eligible = [player for player in eligible if predictor.known(player)]

    if len(eligible) < team_size:
        console.print(
            f"Only {len(eligible)} players match the filters — "
            f"need at least {team_size} to form a team."
        )
        return

    team_count = math.comb(len(eligible), team_size)
    if team_count > max_teams:
        console.print(
            f"[red]{len(eligible)} eligible players form "
            f"{team_count:,} possible teams — refusing to evaluate more than "
            f"{max_teams:,}.[/red]\n"
            "Narrow the roster, e.g. 'task eval MIN_GAMES=3 YEAR=2025'."
        )
        return

    # Map each eligible player to their feature-vector column once.
    columns = numpy.array([predictor.column_of(player) for player in eligible])

    counts = numpy.zeros(len(eligible))
    sums = numpy.zeros(len(eligible))
    squares = numpy.zeros(len(eligible))

    teams = itertools.combinations(range(len(eligible)), team_size)
    batches = range(0, team_count, constants.EVAL_BATCH_SIZE)
    for _ in tqdm.tqdm(batches, desc="Evaluating", disable=len(batches) < 4):
        batch = numpy.array(
            list(itertools.islice(teams, constants.EVAL_BATCH_SIZE))
        )
        features = numpy.zeros(
            (len(batch), len(predictor.roster)), dtype=numpy.float32
        )
        rows = numpy.repeat(numpy.arange(len(batch)), team_size)
        features[rows, columns[batch.ravel()]] = 1.0

        predictions = predictor.infer_features(features)
        member = batch.ravel()
        per_member = numpy.repeat(predictions, team_size)
        numpy.add.at(counts, member, 1)
        numpy.add.at(sums, member, per_member)
        numpy.add.at(squares, member, per_member**2)

    means = sums / counts
    # Sample standard deviation from running sums; 0 when only one sample.
    variance = numpy.where(
        counts > 1,
        numpy.maximum(squares - counts * means**2, 0.0) / numpy.maximum(counts - 1, 1),
        0.0,
    )
    deviations = numpy.sqrt(variance)

    title = "Player evaluation"
    if year is not None:
        title += f" (year={year})"
    if min_games is not None:
        title += f" (min_games={min_games})"

    table = Table(title=title, box=box.SIMPLE_HEAVY)
    table.add_column("Player")
    table.add_column("Mean", justify="right")
    table.add_column("± Std", justify="right")
    table.add_column("Samples", justify="right")

    for index in numpy.argsort(-means):
        table.add_row(
            eligible[index].name,
            f"{means[index]:.2f}",
            f"{deviations[index]:.2f}",
            str(int(counts[index])),
        )
    console.print(table)


def rank_players(
    database: Database,
    predictor: Predictor,
    samples: int = 500,
    seed: int = 0,
    team_size: int = constants.TEAM_SIZE,
) -> list[tuple[str, float]]:
    """Derive a per-player strength score from the trained model.

    A network has no explicit per-player weight, so strength is measured by
    probing: every player is inserted into the SAME ``samples`` random
    companion sets (common random numbers — comparable across players and
    low-variance) and their mean predicted team score is the strength.
    Prints the ranking and returns (name, mean predicted points), best first.
    """
    pool = [
        player for player in database.players.values() if predictor.known(player)
    ]
    excluded = len(database.players) - len(pool)
    if excluded:
        log.warning(
            "%d player(s) unknown to the trained roster are excluded "
            "from the ranking", excluded,
        )
    if len(pool) < team_size:
        console.print(f"Need at least {team_size} rankable players.")
        return []

    generator = numpy.random.default_rng(seed)
    columns = numpy.array([predictor.column_of(player) for player in pool])
    companions = numpy.array([
        generator.choice(len(pool), size=team_size - 1, replace=False)
        for _ in range(samples)
    ])
    base = numpy.zeros((samples, len(predictor.roster)), dtype=numpy.float32)
    rows = numpy.repeat(numpy.arange(samples), team_size - 1)
    base[rows, columns[companions.ravel()]] = 1.0

    rankings = []
    for index, player in enumerate(pool):
        features = base.copy()
        features[:, columns[index]] = 1.0
        predictions = predictor.infer_features(features)
        # Companion sets that already contain the player would score a
        # 4-player team — drop them from their own average.
        valid = ~(companions == index).any(axis=1)
        if not valid.any():
            valid = numpy.ones(samples, dtype=bool)
        rankings.append((player.name, float(predictions[valid].mean())))
    rankings.sort(key=lambda item: item[1], reverse=True)

    table = Table(
        title=f"Player strength (mean predicted score over {samples} "
        f"shared random teams)",
        box=box.SIMPLE_HEAVY,
    )
    table.add_column("#", justify="right", style="bold")
    table.add_column("Player")
    table.add_column("Strength", justify="right")
    for position, (name, value) in enumerate(rankings, start=1):
        table.add_row(str(position), name, f"{value:.2f}")
    console.print(table)

    return rankings


def find_anomalies(
    database: Database,
    predictor: Predictor,
    threshold: float = 2.0,
) -> dict[int, bool]:
    """Flag scored games whose prediction residual is a statistical outlier.

    Prints a report and returns {worksheet row: is_anomaly} for every game
    with a known row — ready for SheetsSource.save_anomalies. Unscored games
    and games outside the trained roster are always False.
    """
    scored = database.scored_games
    predicted = [
        (game, prediction)
        for game in scored
        if (prediction := predictor.infer(game)) is not None
    ]
    skipped = len(scored) - len(predicted)
    if skipped:
        log.warning(
            "%d scored game(s) contain players outside the trained roster "
            "and were skipped", skipped,
        )
    if len(predicted) < 3:
        console.print("Not enough predictable games for anomaly statistics.")
        return {
            game.sheet_row: False
            for game in database.games
            if game.sheet_row is not None
        }

    predictions = numpy.array([prediction for _, prediction in predicted])
    residuals = predictions - numpy.array([game.score for game, _ in predicted])

    deviation = residuals.std()
    z_scores = (residuals - residuals.mean()) / deviation if deviation else residuals * 0.0
    flagged = {
        game: (z, prediction)
        for (game, prediction), z in zip(predicted, z_scores)
        if abs(z) >= threshold
    }

    table = Table(
        title=f"Anomalies (|z| ≥ {threshold}, {len(flagged)}/{len(predicted)} games)",
        box=box.SIMPLE_HEAVY,
    )
    for header in ("Date", "Players", "Actual", "Predicted", "z", "Was"):
        table.add_column(header, justify="right" if header not in ("Date", "Players") else "left")
    for game, (z, prediction) in sorted(flagged.items(), key=lambda item: -abs(item[1][0])):
        table.add_row(
            game.date.isoformat() if game.date else "-",
            ", ".join(player.name for player in game.players),
            str(game.score),
            f"{prediction:.1f}",
            f"{z:+.2f}",
            "✅" if game.is_anomaly else "",
        )
    console.print(table)

    return {
        game.sheet_row: game in flagged
        for game in database.games
        if game.sheet_row is not None
    }
