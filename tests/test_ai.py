"""AI data preparation: split logic and matrix construction."""

import numpy
import pytest

from variableoptimization.ai import ArtificialIntelligence, validation_mask
from variableoptimization.database import Database


def test_validation_mask_prefers_experienced_games():
    # Games whose least-experienced player has >= threshold scored games
    # go to validation; threshold is as high as train_size allows.
    min_participation = numpy.array([1, 5, 5, 5, 2, 5, 5, 5, 5, 1])
    mask = validation_mask(min_participation, train_size=0.8)

    assert mask.tolist() == [False, True, True, True, False, True, True, True, True, False]


def test_validation_mask_all_zero_participation():
    mask = validation_mask(numpy.array([0, 0, 0]), train_size=0.8)
    assert not mask.any()


def test_matrices_use_scored_games_only(fixture_snapshot):
    database = Database(fixture_snapshot)
    ai = ArtificialIntelligence(database)

    # 3 scored games, 4 players; training + validation partition the whole set.
    assert tuple(ai.complete_x.shape) == (3, 4)
    assert len(ai.train_x) + len(ai.validate_x) == 3
    assert ai.complete_y.max() <= 1.0  # scores are normalised


def test_no_scored_games_is_a_clear_error(fixture_snapshot):
    import dataclasses

    unscored = dataclasses.replace(
        fixture_snapshot,
        games=tuple(
            dataclasses.replace(record, score="") for record in fixture_snapshot.games
        ),
    )
    with pytest.raises(ValueError, match="[Nn]o scored games"):
        ArtificialIntelligence(Database(unscored))
