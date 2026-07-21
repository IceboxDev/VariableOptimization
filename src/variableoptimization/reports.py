"""User-facing reports: game preview tables, player rankings, anomaly detection."""

import datetime
import itertools
import statistics
from collections import defaultdict

import numpy
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
) -> None:
    """Rank players by their mean predicted score over all possible teams."""

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
    if len(eligible) < constants.TEAM_SIZE:
        console.print(
            f"Only {len(eligible)} players match the filters — "
            f"need at least {constants.TEAM_SIZE} to form a team."
        )
        return

    scores: dict[str, list[float]] = defaultdict(list)
    for team in itertools.combinations(eligible, constants.TEAM_SIZE):
        prediction = ai.infer(Game(date=None, duration=None, score=None, players=team))
        for player in team:
            scores[player.name].append(prediction)

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

    ranking = sorted(scores.items(), key=lambda item: statistics.mean(item[1]), reverse=True)
    for name, values in ranking:
        deviation = statistics.stdev(values) if len(values) > 1 else 0.0
        table.add_row(
            name,
            f"{statistics.mean(values):.2f}",
            f"{deviation:.2f}",
            str(len(values)),
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
