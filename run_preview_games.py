# -*- coding: utf-8 -*-
# run_preview_games.py

import os
import sys
import datetime
from collections import defaultdict
from typing import Optional, Dict, List

from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich import box

import artificial_intelligence
import modules


console = Console()


def main() -> None:
    # 1) Reconstruct model filename from ALL args after script name
    #    so `[ 799] Model.pt` (even unquoted) becomes one string again.
    model_file: Optional[str]
    if len(sys.argv) >= 2:
        model_file = " ".join(sys.argv[1:]).strip()
        if not model_file:
            model_file = None
    else:
        model_file = None

    credentials_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    database = modules.Database(credentials_path)
    ai = None

    if model_file:
        ai = artificial_intelligence.ArtificialIntelligence(database)
        ai.load(artificial_intelligence.NeuralNetwork, model_file)

    # Group games by year
    games_by_year: Dict[str, List[modules.Game]] = defaultdict(list)
    for game in database.games:
        if game.date is None:
            key = "Unknown"
        else:
            key = str(game.date.year)
        games_by_year[key].append(game)

    term_width = console.width

    for year in sorted(games_by_year.keys()):
        games = sorted(
            games_by_year[year],
            key=lambda g: (
                g.date or datetime.date.min,
                g.score if g.score is not None else -1,
            ),
        )

        # Fixed overall width so all tables are same size
        table = Table(
            box=box.SIMPLE_HEAVY,
            show_lines=False,
            width=term_width,
        )

        # Fixed widths for all but Players, so column layout is consistent
        table.add_column("#", justify="right", style="bold", width=1, no_wrap=True)
        table.add_column("Date", justify="left", width=7, no_wrap=True)
        table.add_column("Score", justify="right", width=3, no_wrap=True)
        table.add_column("Anomaly", justify="center", width=5, no_wrap=True)

        # Players column fills remaining space
        table.add_column("Players", justify="left", no_wrap=False)

        if ai is not None:
            table.add_column("Prediction", justify="left", width=7, no_wrap=True)
            table.add_column("Δ(Pred)", justify="left", width=7, no_wrap=True)

        for idx, game in enumerate(games, start=1):
            date_str = game.date.isoformat() if game.date else "-"
            score_str = "-" if game.score is None or game.score == -1 else str(game.score)
            anomaly_str = "✅" if getattr(game, "is_anomaly", False) else ""
            players_str = ", ".join(p.name for p in game.players)

            row = [
                str(idx),
                date_str,
                score_str,
                anomaly_str,
                players_str,
            ]

            if ai is not None:
                pred = ai.infer(game)
                row.append(f"{pred:.2f}")

                if game.score is not None and game.score != -1:
                    delta = pred - game.score
                    # + red (worse than expected), - green (better)
                    if delta > 0:
                        delta_str = f"[red]{delta:+.2f}[/red]"
                    elif delta < 0:
                        delta_str = f"[green]{delta:+.2f}[/green]"
                    else:
                        delta_str = f"{delta:+.2f}"
                else:
                    delta_str = "-"

                row.append(delta_str)

            table.add_row(*row)

        panel_title = f"[bold cyan]Games in {year}[/bold cyan]"
        console.print(Panel(table, title=panel_title, width=term_width))


if __name__ == "__main__":
    main()