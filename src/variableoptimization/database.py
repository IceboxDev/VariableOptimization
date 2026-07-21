"""Database: parses a Snapshot into domain objects.

This is the only place raw cell strings become dates, scores, and players.
Malformed cells are logged and degraded (bad score -> unscored, bad duration
-> None); a malformed date skips the row with a warning.
"""

import datetime
import logging

from . import constants
from .domain import Game, Player
from .snapshot import GameRecord, Snapshot

log = logging.getLogger(__name__)


def _parse_score(record: GameRecord) -> int | None:
    """'' and negative values (the sheet uses -1) both mean 'not scored'."""
    if not record.score:
        return None
    try:
        score = int(float(record.score))
    except ValueError:
        log.warning(
            "Row %d: score %r is not a number — treating as unscored",
            record.row, record.score,
        )
        return None
    return score if score >= 0 else None


def _parse_duration(record: GameRecord) -> datetime.timedelta | None:
    if not record.duration:
        return None
    try:
        start_raw, end_raw = record.duration.split(constants.DURATION_SEPARATOR)
        start = datetime.datetime.strptime(start_raw, constants.TIME_FORMAT)
        end = datetime.datetime.strptime(end_raw, constants.TIME_FORMAT)
    except ValueError:
        log.warning(
            "Row %d: duration %r is not '%s%s%s' — ignoring",
            record.row, record.duration,
            "HH:MM", constants.DURATION_SEPARATOR, "HH:MM",
        )
        return None
    duration = end - start
    if duration < datetime.timedelta(0):  # game ran past midnight
        duration += datetime.timedelta(days=1)
    return duration


class Database:
    def __init__(self, snapshot: Snapshot) -> None:
        self.snapshot = snapshot
        self.players: dict[str, Player] = {}
        self.games: list[Game] = []

        self._build_players(snapshot)
        self._build_games(snapshot)

    @property
    def scored_games(self) -> list[Game]:
        return [game for game in self.games if game.has_score()]

    def _build_players(self, snapshot: Snapshot) -> None:
        names = sorted({
            name
            for record in snapshot.games
            for name in record.players
            if name != constants.NO_PLAYER
        })
        # Sorted order is load-bearing: it fixes the feature-vector column
        # order the models are trained and inferred with.
        self.players = {name: Player(name) for name in names}

    def _build_games(self, snapshot: Snapshot) -> None:
        for record in snapshot.games:
            try:
                date = datetime.datetime.strptime(
                    record.date, constants.DATE_FORMAT
                ).date()
            except ValueError:
                log.warning(
                    "Row %d: date %r does not match %r — skipping row",
                    record.row, record.date, constants.DATE_FORMAT,
                )
                continue

            game = Game(
                date=date,
                duration=_parse_duration(record),
                score=_parse_score(record),
                players=tuple(
                    self.players[name]
                    for name in record.players
                    if name != constants.NO_PLAYER
                ),
                is_anomaly=record.anomaly.strip().upper() in constants.TRUE_VALUES,
                sheet_row=record.row,
            )
            self.games.append(game)
            for player in game.players:
                player.games.add(game)
