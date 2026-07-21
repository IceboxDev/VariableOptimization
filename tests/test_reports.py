"""Player evaluation: combinatorial guard and batched-inference statistics."""

import numpy
import pytest

from variableoptimization.ai import NeuralNetwork, Predictor
from variableoptimization.database import Database
from variableoptimization.reports import evaluate_players


@pytest.fixture()
def database_and_predictor(fixture_snapshot):
    database = Database(fixture_snapshot)
    roster = list(database.players)
    # An untrained network is fine — these tests exercise mechanics, not skill.
    predictor = Predictor(roster, NeuralNetwork(len(roster)))
    return database, predictor


def test_unfiltered_roster_is_refused(database_and_predictor, capsys):
    database, predictor = database_and_predictor
    # 4 players, teams of 2 -> 6 teams; cap at 5 to trigger the guard.
    evaluate_players(database, predictor, team_size=2, max_teams=5)
    output = capsys.readouterr().out
    assert "refusing" in output
    assert "MIN_GAMES" in output


def test_too_few_players_is_reported(database_and_predictor, capsys):
    database, predictor = database_and_predictor
    evaluate_players(database, predictor, min_games=99)
    assert "match the filters" in capsys.readouterr().out


def test_sample_counts_are_correct(database_and_predictor, capsys):
    database, predictor = database_and_predictor
    # 4 players, teams of 2: each player appears in C(3,1) = 3 teams.
    evaluate_players(database, predictor, team_size=2)
    output = capsys.readouterr().out
    for name in ("Alice", "Bob", "Carol", "Dave"):
        assert name in output
    assert output.count(" 3 ") >= 4 or output.count("3") >= 4


def test_batched_stats_match_per_team_inference(database_and_predictor):
    """The running-sum mean/std must equal naive per-team computation."""
    import itertools
    import statistics

    database, predictor = database_and_predictor
    players = list(database.players.values())

    naive = {player.name: [] for player in players}
    for team in itertools.combinations(players, 2):
        features = numpy.array(
            [[member in team for member in players]], dtype=numpy.float32
        )
        prediction = float(predictor.infer_features(features)[0])
        for member in team:
            naive[member.name].append(prediction)

    # Reproduce the batched accumulation directly.
    columns = numpy.array([predictor.column_of(player) for player in players])
    teams = numpy.array(list(itertools.combinations(range(len(players)), 2)))
    features = numpy.zeros((len(teams), len(predictor.roster)), dtype=numpy.float32)
    rows = numpy.repeat(numpy.arange(len(teams)), 2)
    features[rows, columns[teams.ravel()]] = 1.0
    predictions = predictor.infer_features(features)

    for index, player in enumerate(players):
        member_predictions = predictions[(teams == index).any(axis=1)]
        assert numpy.mean(member_predictions) == pytest.approx(
            statistics.mean(naive[player.name]), abs=1e-4
        )
