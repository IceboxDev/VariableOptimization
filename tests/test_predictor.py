"""Predictor: roster-decoupled inference — old models survive roster growth."""

import dataclasses
import json

import pytest

from variableoptimization.ai import NeuralNetwork, Predictor
from variableoptimization.database import Database
from variableoptimization.domain import Game
from variableoptimization.reports import evaluate_players
from variableoptimization.snapshot import GameRecord

ROSTER = ["Alice", "Bob", "Carol", "Dave"]


@pytest.fixture()
def saved_model(tmp_path):
    predictor = Predictor(ROSTER, NeuralNetwork(len(ROSTER)))
    predictor.save(tmp_path / "model.pt", tmp_path / "roster.json")
    return tmp_path / "model.pt"


def test_load_reconstructs_dimension_from_roster(saved_model):
    predictor = Predictor.load(saved_model)
    assert predictor.roster == ROSTER
    assert predictor.network.model[0].in_features == len(ROSTER)


def test_load_without_roster_is_a_clear_error(tmp_path, saved_model):
    (saved_model.parent / "roster.json").unlink()
    with pytest.raises(FileNotFoundError, match="roster"):
        Predictor.load(saved_model)


def grown_database(fixture_snapshot) -> Database:
    """The fixture database plus a new scored game introducing player Eve."""
    new_game = GameRecord(
        row=99, date="03/01/24", duration="", score="42", anomaly="",
        players=("Eve", "Alice"),
    )
    return Database(
        dataclasses.replace(
            fixture_snapshot, games=fixture_snapshot.games + (new_game,)
        )
    )


def test_known_games_predict_after_roster_growth(fixture_snapshot, saved_model):
    database = grown_database(fixture_snapshot)
    assert "Eve" in database.players  # live roster has grown past the model's

    predictor = Predictor.load(saved_model)
    known_game = Game(
        date=None, duration=None, score=None,
        players=(database.players["Alice"], database.players["Bob"]),
    )
    assert isinstance(predictor.infer(known_game), float)


def test_unknown_player_games_return_none_with_warning(
    fixture_snapshot, saved_model, caplog
):
    database = grown_database(fixture_snapshot)
    predictor = Predictor.load(saved_model)
    eve_game = Game(
        date=None, duration=None, score=None,
        players=(database.players["Eve"], database.players["Alice"]),
    )
    with caplog.at_level("WARNING"):
        assert predictor.infer(eve_game) is None
        assert predictor.infer(eve_game) is None  # second call: no new warning
    assert sum("Eve" in message for message in caplog.messages) == 1


def test_eval_excludes_players_outside_trained_roster(
    fixture_snapshot, saved_model, capsys, caplog
):
    database = grown_database(fixture_snapshot)
    predictor = Predictor.load(saved_model)
    with caplog.at_level("WARNING"):
        evaluate_players(database, predictor, team_size=2)

    output = capsys.readouterr().out
    assert "Eve" not in output
    assert any("Eve" in message for message in caplog.messages)
