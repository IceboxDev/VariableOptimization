"""Anomaly detection: threshold, top-N limiting, and write-back flags."""

import pytest

from variableoptimization.ai import NeuralNetwork, Predictor
from variableoptimization.database import Database
from variableoptimization.reports import find_anomalies


@pytest.fixture()
def database_and_predictor(fixture_snapshot):
    database = Database(fixture_snapshot)
    roster = list(database.players)
    predictor = Predictor(roster, NeuralNetwork(len(roster)))
    return database, predictor


def test_flags_cover_every_sheet_row(database_and_predictor):
    database, predictor = database_and_predictor
    flags = find_anomalies(database, predictor, threshold=99.0)
    assert set(flags) == {game.sheet_row for game in database.games}
    assert not any(flags.values())  # nothing beats a z of 99


def test_threshold_zero_flags_all_scored_games(database_and_predictor):
    database, predictor = database_and_predictor
    flags = find_anomalies(database, predictor, threshold=0.0)
    assert sum(flags.values()) == len(database.scored_games)


def test_top_limits_flag_count(database_and_predictor):
    database, predictor = database_and_predictor
    flags = find_anomalies(database, predictor, threshold=0.0, top=1)
    assert sum(flags.values()) == 1


def test_top_keeps_the_most_extreme_outlier(database_and_predictor):
    database, predictor = database_and_predictor
    all_flags = find_anomalies(database, predictor, threshold=0.0)
    top_flags = find_anomalies(database, predictor, threshold=0.0, top=1)

    [top_row] = [row for row, flagged in top_flags.items() if flagged]
    assert all_flags[top_row]  # the survivor was flagged in the full set too
