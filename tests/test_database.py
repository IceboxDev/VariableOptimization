"""Snapshot -> domain parsing: defaults, sentinels, registration."""

import datetime

from variableoptimization.database import Database
from variableoptimization.domain import Game
from variableoptimization.snapshot import GameRecord, Snapshot, WeightRecord


def test_players_and_weights(fixture_snapshot):
    database = Database(fixture_snapshot)
    assert sorted(database.players) == ["Alice", "Bob", "Carol", "Dave"]  # no N/A
    assert database.players["Alice"].weight == 0.5
    assert database.players["Dave"].weight == 0.0  # absent from weights block


def test_missing_overlap_defaults_to_zero(fixture_snapshot, caplog):
    with caplog.at_level("WARNING"):
        database = Database(fixture_snapshot)
    assert database.overlap == 0.0
    assert any("Overlap" in message for message in caplog.messages)


def test_scores_and_sentinels(fixture_snapshot):
    database = Database(fixture_snapshot)
    assert [game.score for game in database.games] == [40, None, None, 55, 33, None]
    # -1 and blank both mean unscored.
    assert len(database.scored_games) == 3


def test_durations(fixture_snapshot):
    database = Database(fixture_snapshot)
    assert database.games[0].duration == datetime.timedelta(hours=1, minutes=45)
    # Overnight games wrap past midnight instead of going negative.
    assert database.games[2].duration == datetime.timedelta(hours=1, minutes=15)
    assert database.games[1].duration is None


def test_anomaly_flags(fixture_snapshot):
    database = Database(fixture_snapshot)
    assert [game.is_anomaly for game in database.games] == [
        False, False, False, True, False, False,
    ]


def test_sheet_rows_survive_for_writebacks(fixture_snapshot):
    database = Database(fixture_snapshot)
    assert [game.sheet_row for game in database.games] == [2, 3, 4, 5, 6, 7]


def test_real_games_register_with_players(fixture_snapshot):
    database = Database(fixture_snapshot)
    alice = database.players["Alice"]
    assert len(alice.games) == 4
    assert len(alice.scored_games) == 2  # rows 2 and 5


def test_hypothetical_games_do_not_pollute_histories(fixture_snapshot):
    database = Database(fixture_snapshot)
    alice = database.players["Alice"]
    games_before = set(alice.games)

    Game(date=None, duration=None, score=None, players=(alice,))

    assert alice.games == games_before


def test_bad_cells_degrade_without_crashing(caplog):
    snapshot = Snapshot(
        games=(
            GameRecord(row=2, date="not-a-date", duration="", score="1", anomaly="", players=("A",)),
            GameRecord(row=3, date="01/05/24", duration="junk", score="abc", anomaly="", players=("A",)),
        ),
        weights=(WeightRecord(name="Overlap", weight="0.1"),),
    )
    with caplog.at_level("WARNING"):
        database = Database(snapshot)

    assert len(database.games) == 1  # bad date skipped with a warning
    assert database.games[0].score is None  # bad score -> unscored
    assert database.games[0].duration is None  # bad duration -> None
    assert database.overlap == 0.1
    assert len(caplog.messages) == 3
