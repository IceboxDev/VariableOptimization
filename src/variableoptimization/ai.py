"""Model training and inference.

Feature vector: binary player participation, columns in the fixed sorted-name
order established by Database. Target: game score normalised by
GAME_MAX_SCORE. Games are weighted by recency when ranking trained models.
"""

import copy
import logging
import math
import typing
from multiprocessing.pool import ThreadPool
from pathlib import Path

import matplotlib.pyplot
import numpy
import scipy.stats
import torch
import tqdm

from . import constants
from .database import Database
from .domain import Game

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


def resolve_model_path(reference: str) -> Path:
    """Resolve a model reference to a file.

    ``latest`` picks the newest ``*.pt`` in the models directory (falling back
    to the legacy directory). Anything else is tried as given, then relative
    to the models and legacy directories.
    """
    if reference == "latest":
        candidates = sorted(
            [
                *Path(constants.MODELS_DIR).glob("*.pt"),
                *Path(constants.LEGACY_MODELS_DIR).glob("*.pt"),
            ],
            key=lambda path: path.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No trained models in {constants.MODELS_DIR}/ or "
                f"{constants.LEGACY_MODELS_DIR}/ — run 'vopt train' first."
            )
        return candidates[0]

    tried = [
        Path(reference),
        Path(constants.MODELS_DIR) / reference,
        Path(constants.LEGACY_MODELS_DIR) / reference,
    ]
    for path in tried:
        if path.is_file():
            return path
    raise FileNotFoundError(f"Model not found; tried: {[str(p) for p in tried]}")


class ArtificialIntelligence:
    def __init__(self, database: Database, device: torch.device | None = None) -> None:
        self.database = database
        self.device = device or get_device()
        self.algorithm: NeuralNetwork | None = None
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

    def _train_once(self, algorithm_class: type) -> tuple[typing.Any, float]:
        algorithm = algorithm_class(self.train_x.shape[1], device=self.device)
        algorithm.train(self.train_x, self.train_y, self.validate_x, self.validate_y)

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
    ) -> tuple[Path, float]:
        """Train ``best_of`` candidates, keep the best, save model + loss plot.

        Runs are reproducible for a given seed only with ``workers=1``.
        Returns (model path, best loss).
        """
        log.info("Training on %s", self.device)
        if self.device.type == "cuda":
            log.info("GPU: %s", torch.cuda.get_device_name(0))
        if seed is not None:
            torch.manual_seed(seed)
            numpy.random.seed(seed)

        best_loss, loss_history = math.inf, []
        progress = tqdm.tqdm(desc="Training", total=best_of, postfix={"best": math.inf})

        def record(algorithm: typing.Any, loss: float) -> None:
            nonlocal best_loss
            loss_history.append(loss)
            if loss < best_loss:
                best_loss, self.algorithm = loss, algorithm
                progress.set_postfix({"best": round(best_loss)})
            progress.update(1)

        with progress:
            if workers > 1:
                with ThreadPool(workers) as pool:
                    results = pool.imap_unordered(
                        lambda _: self._train_once(algorithm_class), range(best_of)
                    )
                    for algorithm, loss in results:
                        record(algorithm, loss)
            else:
                for _ in range(best_of):
                    record(*self._train_once(algorithm_class))

        models_dir = Path(constants.MODELS_DIR)
        models_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{algorithm_class.__name__.lower()}-{best_loss:04.0f}"
        model_path = models_dir / f"{stem}.pt"
        self.algorithm.save(model_path)
        self._save_loss_plot(loss_history, best_loss, models_dir / f"{stem}.png")
        return model_path, best_loss

    @staticmethod
    def _save_loss_plot(
        loss_history: list[float], best_loss: float, path: Path
    ) -> None:
        figure, axes = matplotlib.pyplot.subplots()
        losses = sorted(loss_history)
        mean, std = numpy.mean(losses), numpy.std(losses)
        density = scipy.stats.norm.pdf(losses, mean, std)

        axes.plot(losses, density)
        axes.axvline(x=best_loss, color="r", linestyle="--")
        axes.axvline(x=mean, color="y", linestyle="-")
        axes.axvline(x=mean + std, color="y", linestyle="--")
        axes.axvline(x=mean - std, color="y", linestyle="--")

        label_height = numpy.max(density) / 2
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

    def load(self, model_reference: str, algorithm_class: type = NeuralNetwork) -> Path:
        path = resolve_model_path(model_reference)
        self.algorithm = algorithm_class(self.train_x.shape[1], device=self.device)
        try:
            self.algorithm.load(path)
        except RuntimeError as error:
            raise RuntimeError(
                f"Model {path} does not fit the current roster of "
                f"{self.train_x.shape[1]} players — it was trained on older "
                "data. Retrain with 'vopt train'."
            ) from error
        log.info("Loaded model %s", path)
        return path

    def infer(self, game: Game) -> float:
        features = numpy.array(
            [[player in game.players for player in self.players]],
            dtype=numpy.float32,
        )
        return float(self.infer_features(features)[0])

    def infer_features(self, features: numpy.ndarray) -> numpy.ndarray:
        """Predict scores for a (batch, players) participation matrix in one
        forward pass. Column order must match ``self.players``."""
        if self.algorithm is None:
            raise RuntimeError("No model loaded — call train() or load() first.")
        tensor = torch.tensor(features, dtype=torch.float32, device=self.device)
        predictions = self.algorithm.infer(tensor) * constants.GAME_MAX_SCORE
        return predictions.cpu().numpy()
