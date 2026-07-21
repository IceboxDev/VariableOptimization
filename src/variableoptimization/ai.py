"""Model training and inference.

Feature vector: binary player participation. The column order is the model's
*trained roster* (sorted player names at training time), persisted next to the
weights as ``roster.json`` — so a model keeps working after the live roster
grows. Training is orchestrated by :class:`ArtificialIntelligence` (pure, no
file I/O); inference goes through :class:`Predictor`, which never depends on
the live database.
"""

import copy
import dataclasses
import json
import logging
import math
import typing
from multiprocessing.pool import ThreadPool
from pathlib import Path

import matplotlib

matplotlib.use("Agg")  # plots are only ever saved, never shown

import matplotlib.pyplot
import numpy
import scipy.stats
import torch
import tqdm

from . import constants, runs
from .database import Database
from .domain import Game, Player

log = logging.getLogger(__name__)


def get_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def validation_mask(
    min_participation: numpy.ndarray, train_size: float
) -> numpy.ndarray:
    """Split games into training/validation by player experience.

    Validation games are those where every player has at least ``threshold``
    scored games — you cannot validate predictions for players the model
    barely saw. The threshold is the largest value keeping the training
    fraction at or below ``train_size``.
    """
    for threshold in range(int(min_participation.max()), 0, -1):
        training_fraction = float((min_participation < threshold).mean())
        if training_fraction <= train_size:
            return min_participation >= threshold
    return numpy.zeros(min_participation.shape, dtype=bool)


class NeuralNetwork:
    def __init__(
        self,
        input_size: int,
        layer_count: int = 3,
        first_layer_function: typing.Callable[[int], int] = lambda x: x * 2,
        next_layer_function: typing.Callable[[int], int] = lambda x: x // 2,
        activation_function: type = torch.nn.ReLU,
        device: torch.device | None = None,
    ) -> None:
        self.device = device or get_device()
        self.model = torch.nn.Sequential()

        node_count = input_size
        for layer_no in range(layer_count):
            next_node_count = (
                first_layer_function(node_count)
                if layer_no == 0
                else next_layer_function(node_count)
            )
            self.model.add_module(
                f"l-{layer_no}", torch.nn.Linear(node_count, next_node_count)
            )
            self.model.add_module(f"f-{layer_no}", activation_function())
            node_count = next_node_count

        self.model.add_module("l-scalar", torch.nn.Linear(node_count, 1))
        self.model.to(self.device)

    def train(
        self,
        train_x: torch.Tensor,
        train_y: torch.Tensor,
        validate_x: torch.Tensor,
        validate_y: torch.Tensor,
        loss_function: typing.Callable = torch.nn.MSELoss(),
        epochs: int = 1000,
        batch_size: int = 10,
    ) -> None:
        best_loss, best_weights = math.inf, None
        batches = range(0, len(train_x), batch_size)
        optimizer = torch.optim.Adam(self.model.parameters())

        for _ in range(epochs):
            self.model.train()
            for start in batches:
                batch_x = train_x[start:start + batch_size]
                batch_y = train_y[start:start + batch_size]

                prediction = self.model(batch_x)
                loss = loss_function(prediction, batch_y)

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

            self.model.eval()
            with torch.inference_mode():
                loss = loss_function(self.model(validate_x), validate_y).item()

            if loss < best_loss:
                best_loss = loss
                best_weights = copy.deepcopy(self.model.state_dict())

        if best_weights is not None:
            self.model.load_state_dict(best_weights)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.inference_mode():
            return torch.flatten(self.model(x.to(self.device)))

    def save(self, path: str | Path) -> None:
        torch.save(self.model.state_dict(), Path(path))

    def load(self, path: str | Path) -> None:
        self.model.load_state_dict(
            torch.load(Path(path), map_location=self.device)
        )


class Predictor:
    """Inference against a model's *trained* roster.

    The roster fixes the feature-column order forever; players unknown to the
    trained roster make a game unpredictable (``infer`` returns None) rather
    than silently wrong.
    """

    def __init__(self, roster: list[str], network: NeuralNetwork) -> None:
        self.roster = list(roster)
        self.network = network
        self._column = {name: index for index, name in enumerate(self.roster)}
        self._warned: set[str] = set()

    @classmethod
    def load(cls, model_path: Path, device: torch.device | None = None) -> "Predictor":
        roster_path = model_path.parent / constants.DEPLOYED_ROSTER_FILENAME
        if not roster_path.is_file():
            raise FileNotFoundError(
                f"No roster next to the model: {roster_path} — a model is only "
                "loadable together with the roster it was trained on."
            )
        with open(roster_path, encoding="utf-8") as handle:
            roster = json.load(handle)

        network = NeuralNetwork(len(roster), device=device)
        network.load(model_path)
        log.info("Loaded model %s (%d-player roster)", model_path, len(roster))
        return cls(roster, network)

    def save(self, model_path: Path, roster_path: Path) -> None:
        self.network.save(model_path)
        with open(roster_path, "w", encoding="utf-8") as handle:
            json.dump(self.roster, handle, indent=2)
            handle.write("\n")

    def known(self, player: Player) -> bool:
        return player.name in self._column

    def column_of(self, player: Player) -> int:
        return self._column[player.name]

    def features_for(self, players: typing.Iterable[Player]) -> numpy.ndarray | None:
        """Participation vector for a team, or None if any player is unknown
        to the trained roster (warned once per name)."""
        names = [player.name for player in players]
        unknown = [name for name in names if name not in self._column]
        if unknown:
            for name in unknown:
                if name not in self._warned:
                    self._warned.add(name)
                    log.warning(
                        "Player %r is not in the trained roster — games with "
                        "them cannot be predicted by this model", name,
                    )
            return None
        features = numpy.zeros(len(self.roster), dtype=numpy.float32)
        features[[self._column[name] for name in names]] = 1.0
        return features

    def infer(self, game: Game) -> float | None:
        features = self.features_for(game.players)
        if features is None:
            return None
        return float(self.infer_features(features.reshape(1, -1))[0])

    def infer_features(self, features: numpy.ndarray) -> numpy.ndarray:
        """Predict scores for a (batch, roster) participation matrix in one
        forward pass. Column order must match ``self.roster``."""
        tensor = torch.tensor(
            features, dtype=torch.float32, device=self.network.device
        )
        predictions = self.network.infer(tensor) * constants.GAME_MAX_SCORE
        return predictions.cpu().numpy()


@dataclasses.dataclass
class TrainingResult:
    predictor: Predictor
    best_loss: float
    loss_history: list[float]


def resolve_model(reference: str | None, output_dir: Path) -> Path:
    """Resolve a model reference to a weights file.

    ``None``/``deployed`` -> the promoted model; ``latest`` -> the newest
    completed run; anything else is an explicit path (with a sibling
    ``roster.json``).
    """
    output_dir = Path(output_dir)
    if reference is None or reference == "deployed":
        path = output_dir / constants.DEPLOYED_MODEL_FILENAME
        if not path.is_file():
            raise FileNotFoundError(
                "No deployed model yet — run 'task train' first."
            )
        return path

    if reference == "latest":
        latest = runs.find_latest_run(runs.runs_root(output_dir))
        if latest is None:
            raise FileNotFoundError(
                "No completed training runs yet — run 'task train' first."
            )
        return latest / "model.pt"

    path = Path(reference)
    if path.is_file():
        return path
    raise FileNotFoundError(f"Model not found: {reference}")


class ArtificialIntelligence:
    def __init__(self, database: Database, device: torch.device | None = None) -> None:
        self.database = database
        self.device = device or get_device()
        self.players = list(database.players.values())

        scored = database.scored_games
        if not scored:
            raise ValueError("No scored games in the database — nothing to learn from.")

        membership = numpy.array(
            [[player in game.players for player in self.players] for game in scored],
            dtype=bool,
        )
        scores = numpy.array(
            [game.score / constants.GAME_MAX_SCORE for game in scored],
            dtype=numpy.float32,
        )

        # Per game: the scored-game count of its least experienced player.
        experience = numpy.array(
            [len(player.scored_games) for player in self.players]
        )
        min_participation = numpy.array([
            experience[row].min() if row.any() else 0 for row in membership
        ])

        # Recent games weigh more when ranking candidate models.
        reference_year = max(game.date.year for game in scored)
        self.recency = torch.tensor(
            [
                math.exp((game.date.year - reference_year) * 0.5)
                for game in scored
            ],
            dtype=torch.float32,
            device=self.device,
        )

        validation = validation_mask(min_participation, constants.TRAINING_SET_SIZE)
        features = torch.tensor(membership, dtype=torch.float32, device=self.device)
        targets = torch.tensor(scores, device=self.device).reshape(-1, 1)

        self.train_x = features[~validation]
        self.train_y = targets[~validation]
        self.validate_x = features[validation]
        self.validate_y = targets[validation]
        self.complete_x = features
        self.complete_y = targets.flatten()

    def _train_once(
        self, algorithm_class: type, epochs: int
    ) -> tuple[NeuralNetwork, float]:
        algorithm = algorithm_class(self.train_x.shape[1], device=self.device)
        algorithm.train(
            self.train_x, self.train_y, self.validate_x, self.validate_y,
            epochs=epochs,
        )

        prediction = algorithm.infer(self.complete_x)
        difference = (self.complete_y - prediction) * constants.GAME_MAX_SCORE
        loss = torch.sum(torch.square(difference * self.recency)).item()
        return algorithm, loss

    def train(
        self,
        algorithm_class: type = NeuralNetwork,
        best_of: int = 100,
        seed: int | None = None,
        workers: int = 1,
        epochs: int = 1000,
    ) -> TrainingResult:
        """Train ``best_of`` candidates and return the best, wrapped with the
        training roster. Pure computation — persisting the result is the
        caller's job. Reproducible for a given seed only with ``workers=1``.
        """
        log.info("Training on %s", self.device)
        if self.device.type == "cuda":
            log.info("GPU: %s", torch.cuda.get_device_name(0))
        if seed is not None:
            torch.manual_seed(seed)
            numpy.random.seed(seed)

        best_loss, best_network, loss_history = math.inf, None, []
        progress = tqdm.tqdm(desc="Training", total=best_of, postfix={"best": math.inf})

        def record(network: NeuralNetwork, loss: float) -> None:
            nonlocal best_loss, best_network
            loss_history.append(loss)
            if loss < best_loss:
                best_loss, best_network = loss, network
                progress.set_postfix({"best": round(best_loss)})
            progress.update(1)

        with progress:
            if workers > 1:
                with ThreadPool(workers) as pool:
                    results = pool.imap_unordered(
                        lambda _: self._train_once(algorithm_class, epochs),
                        range(best_of),
                    )
                    for network, loss in results:
                        record(network, loss)
            else:
                for _ in range(best_of):
                    record(*self._train_once(algorithm_class, epochs))

        roster = [player.name for player in self.players]
        return TrainingResult(
            predictor=Predictor(roster, best_network),
            best_loss=best_loss,
            loss_history=loss_history,
        )


def save_loss_plot(loss_history: list[float], best_loss: float, path: Path) -> None:
    figure, axes = matplotlib.pyplot.subplots()
    losses = sorted(loss_history)
    mean, std = numpy.mean(losses), numpy.std(losses)

    if std > 0:  # a single candidate has no distribution to draw
        density = scipy.stats.norm.pdf(losses, mean, std)
        axes.plot(losses, density)
        label_height = numpy.max(density) / 2
    else:
        label_height = 0.5

    axes.axvline(x=best_loss, color="r", linestyle="--")
    axes.axvline(x=mean, color="y", linestyle="-")
    axes.axvline(x=mean + std, color="y", linestyle="--")
    axes.axvline(x=mean - std, color="y", linestyle="--")
    for value, text in ((best_loss, "Best"), (mean, "Mean"), (mean + std, "STD")):
        axes.text(
            value, label_height, f"{text}: {value:.0f}",
            horizontalalignment="right", verticalalignment="center",
            rotation="vertical",
        )

    axes.set_title("Loss Distribution")
    axes.set_xlabel("Loss")
    axes.set_ylabel("Probability")
    figure.savefig(path)
    matplotlib.pyplot.close(figure)
