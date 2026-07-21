"""Dataset fingerprint: stable, and blind to what training doesn't see."""

import dataclasses

from variableoptimization.snapshot import GameRecord, fingerprint


def test_deterministic(fixture_snapshot):
    assert fingerprint(fixture_snapshot) == fingerprint(fixture_snapshot)
    assert len(fingerprint(fixture_snapshot)) == 16


def test_unscored_games_do_not_change_it(fixture_snapshot):
    upcoming = GameRecord(
        row=99, date="12/31/26", duration="", score="", anomaly="",
        players=("Alice", "Bob"),
    )
    grown = dataclasses.replace(
        fixture_snapshot, games=fixture_snapshot.games + (upcoming,)
    )
    assert fingerprint(grown) == fingerprint(fixture_snapshot)


def test_scored_game_changes_it(fixture_snapshot):
    played = GameRecord(
        row=99, date="12/31/26", duration="", score="50", anomaly="",
        players=("Alice", "Bob"),
    )
    grown = dataclasses.replace(
        fixture_snapshot, games=fixture_snapshot.games + (played,)
    )
    assert fingerprint(grown) != fingerprint(fixture_snapshot)


def test_new_player_changes_it(fixture_snapshot):
    games = list(fixture_snapshot.games)
    games[0] = dataclasses.replace(games[0], players=games[0].players + ("Eve",))
    assert fingerprint(
        dataclasses.replace(fixture_snapshot, games=tuple(games))
    ) != fingerprint(fixture_snapshot)
