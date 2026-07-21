"""Pub-quiz score prediction: data pipeline and neural-network training."""

from .database import Database
from .domain import Game, Player
from .loader import DataSettings, load_database
from .snapshot import Snapshot

__all__ = [
    "ArtificialIntelligence",
    "DataSettings",
    "Database",
    "Game",
    "NeuralNetwork",
    "Player",
    "Snapshot",
    "load_database",
]


def __getattr__(name: str):
    # Lazy: importing the ai module pulls in torch (~seconds); commands that
    # never touch a model shouldn't pay for it.
    if name in ("ArtificialIntelligence", "NeuralNetwork"):
        from . import ai

        return getattr(ai, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
