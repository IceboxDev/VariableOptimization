"""User-facing reports: game preview tables, player rankings, anomaly detection."""

import datetime
import itertools
import math
from collections import defaultdict

import numpy
import tqdm
from rich import box
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from . import constants
from .ai import ArtificialIntelligence
from .database import Database
from .domain import Game

console = Console()


def preview_games(database: Database, ai: ArtificialIntelligence | None = None) -> None:
    """Print one panel per year. Column widths are computed globally so every
    panel lines up."""
    headers = ["#", "Date", "Score", "Anomaly", "Players"]
    if ai is not None:
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
            if ai is not None:
                prediction = ai.infer(game)
                row["Prediction"] = f"{prediction:.2f}"
                if game.has_score():
                    delta = prediction - game.score
                    color = "red" if delta > 0 else "green" if delta < 0 else None
                    plain = f"{delta:+.2f}"
                    row["Δ(Pred)"] = f"[{color}]{plain}[/{color}]" if color else plain
                    # Width bookkeeping must use the plain text, not the markup.
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
        if ai is not None:
            table.add_column("Prediction", justify="right", width=widths["Prediction"], no_wrap=True)
            table.add_column("Δ(Pred)", justify="right", width=widths["Δ(Pred)"], no_wrap=True)

        for row in rows_by_year[year]:
            table.add_row(*(row[header] for header in headers))
        console.print(Panel.fit(table, title=f"[bold cyan]Games in {year}[/bold cyan]"))


def evaluate_players(
    database: Database,
    ai: ArtificialIntelligence,
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
    columns = numpy.array([ai.players.index(player) for player in eligible])

    counts = numpy.zeros(len(eligible))
    sums = numpy.zeros(len(eligible))
    squares = numpy.zeros(len(eligible))

    teams = itertools.combinations(range(len(eligible)), team_size)
    batches = range(0, team_count, constants.EVAL_BATCH_SIZE)
    for _ in tqdm.tqdm(batches, desc="Evaluating", disable=len(batches) < 4):
        batch = numpy.array(
            list(itertools.islice(teams, constants.EVAL_BATCH_SIZE))
        )
        features = numpy.zeros((len(batch), len(ai.players)), dtype=numpy.float32)
        rows = numpy.repeat(numpy.arange(len(batch)), team_size)
        features[rows, columns[batch.ravel()]] = 1.0

        predictions = ai.infer_features(features)
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


def find_anomalies(
    database: Database,
    ai: ArtificialIntelligence,
    threshold: float = 2.0,
) -> dict[int, bool]:
    """Flag scored games whose prediction residual is a statistical outlier.

    Prints a report and returns {worksheet row: is_anomaly} for every game
    with a known row — ready for SheetsSource.save_anomalies (unscored games
    are always False, and previously set flags get cleared).
    """
    scored = database.scored_games
    predictions = numpy.array([ai.infer(game) for game in scored])
    residuals = predictions - numpy.array([game.score for game in scored])

    deviation = residuals.std()
    z_scores = (residuals - residuals.mean()) / deviation if deviation else residuals * 0.0
    flagged = {
        game: (z, prediction)
        for game, z, prediction in zip(scored, z_scores, predictions)
        if abs(z) >= threshold
    }

    table = Table(
        title=f"Anomalies (|z| ≥ {threshold}, {len(flagged)}/{len(scored)} games)",
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
