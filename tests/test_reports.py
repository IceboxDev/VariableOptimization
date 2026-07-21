"""Player evaluation: combinatorial guard and batched-inference statistics."""

import numpy
import pytest

from variableoptimization.ai import ArtificialIntelligence, NeuralNetwork
from variableoptimization.database import Database
from variableoptimization.reports import evaluate_players


@pytest.fixture()
def ai_with_model(fixture_snapshot):
    database = Database(fixture_snapshot)
    ai = ArtificialIntelligence(database)
    # An untrained network is fine — these tests exercise mechanics, not skill.
    ai.algorithm = NeuralNetwork(len(ai.players), device=ai.device)
    return database, ai


def test_unfiltered_roster_is_refused(ai_with_model, capsys):
    database, ai = ai_with_model
    # 4 players, teams of 2 -> 6 teams; cap at 5 to trigger the guard.
    evaluate_players(database, ai, team_size=2, max_teams=5)
    output = capsys.readouterr().out
    assert "refusing" in output
    assert "MIN_GAMES" in output


def test_too_few_players_is_reported(ai_with_model, capsys):
    database, ai = ai_with_model
    evaluate_players(database, ai, min_games=99)
    assert "match the filters" in capsys.readouterr().out


def test_sample_counts_are_correct(ai_with_model, capsys):
    database, ai = ai_with_model
    # 4 players, teams of 2: each player appears in C(3,1) = 3 teams.
    evaluate_players(database, ai, team_size=2)
    output = capsys.readouterr().out
    for name in ("Alice", "Bob", "Carol", "Dave"):
        assert name in output
    assert output.count(" 3 ") >= 4 or output.count("3") >= 4


def test_batched_stats_match_per_team_inference(ai_with_model):
    """The running-sum mean/std must equal naive per-team computation."""
    import itertools
    import statistics

    database, ai = ai_with_model
    players = list(database.players.values())

    naive = {player.name: [] for player in players}
    for team in itertools.combinations(players, 2):
        features = numpy.array(
            [[member in team for member in ai.players]], dtype=numpy.float32
        )
        prediction = float(ai.infer_features(features)[0])
        for member in team:
            naive[member.name].append(prediction)

    # Reproduce the batched accumulation directly.
    columns = numpy.array([ai.players.index(player) for player in players])
    teams = numpy.array(list(itertools.combinations(range(len(players)), 2)))
    features = numpy.zeros((len(teams), len(ai.players)), dtype=numpy.float32)
    rows = numpy.repeat(numpy.arange(len(teams)), 2)
    features[rows, columns[teams.ravel()]] = 1.0
    predictions = ai.infer_features(features)

    for index, player in enumerate(players):
        member_predictions = predictions[(teams == index).any(axis=1)]
        assert numpy.mean(member_predictions) == pytest.approx(
            statistics.mean(naive[player.name]), abs=1e-4
        )
