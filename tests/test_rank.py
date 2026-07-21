"""Player-strength ranking: determinism, coverage, and sheet write-back."""

import pytest

from variableoptimization import constants
from variableoptimization.ai import NeuralNetwork, Predictor
from variableoptimization.database import Database
from variableoptimization.reports import rank_players


@pytest.fixture()
def database_and_predictor(fixture_snapshot):
    database = Database(fixture_snapshot)
    roster = list(database.players)
    predictor = Predictor(roster, NeuralNetwork(len(roster)))
    return database, predictor


def test_ranks_every_known_player(database_and_predictor):
    database, predictor = database_and_predictor
    rankings = rank_players(database, predictor, samples=50, team_size=2)
    assert sorted(name for name, _ in rankings) == ["Alice", "Bob", "Carol", "Dave"]
    assert all(isinstance(value, float) for _, value in rankings)
    # Sorted best-first.
    values = [value for _, value in rankings]
    assert values == sorted(values, reverse=True)


def test_same_seed_is_deterministic(database_and_predictor):
    database, predictor = database_and_predictor
    first = rank_players(database, predictor, samples=50, seed=7, team_size=2)
    second = rank_players(database, predictor, samples=50, seed=7, team_size=2)
    assert first == second


def test_too_small_pool_returns_empty(database_and_predictor, capsys):
    database, predictor = database_and_predictor
    assert rank_players(database, predictor, samples=10, team_size=9) == []
    assert "at least 9" in capsys.readouterr().out


def test_unknown_players_are_excluded(fixture_snapshot, caplog):
    database = Database(fixture_snapshot)
    trained_on_three = ["Alice", "Bob", "Carol"]  # Dave joined later
    predictor = Predictor(trained_on_three, NeuralNetwork(3))

    with caplog.at_level("WARNING"):
        rankings = rank_players(database, predictor, samples=50, team_size=2)

    assert sorted(name for name, _ in rankings) == trained_on_three
    assert any("excluded" in message for message in caplog.messages)


def test_save_rankings_overwrites_the_whole_block():
    from test_sheets_source import FakeWorksheet
    from variableoptimization.sources import SheetsSource

    source = SheetsSource("unused.json", worksheet=FakeWorksheet([]))
    source.save_rankings([("Alice", 38.911), ("Bob", 36.5)])

    [(cell_range, values)] = source.worksheet.updates
    assert cell_range == "BE3:BF1000"
    assert values[0] == ["Alice", 38.91]
    assert values[1] == ["Bob", 36.5]
    assert values[2] == ["", ""]  # stale rows below get cleared
    assert len(values) == constants.SHEET_MAX_ROW - constants.RANKINGS_FIRST_ROW + 1
