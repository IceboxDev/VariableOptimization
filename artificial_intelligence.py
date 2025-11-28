# -*- coding: utf-8 -*-
from multiprocessing.pool import ThreadPool
import matplotlib.pyplot

import constants
import itertools
import typing
import numpy
import scipy
import torch
import copy
import math
import tqdm
import os

# Select device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ArtificialIntelligence:

    # noinspection PyUnresolvedReferences
    def __init__(self, database: 'Database') -> None:
        self.database = database
        self.algorithm = None
        self.device = device

        # Player matrix is a 2D matrix of games x players
        player_matrix = numpy.ma.masked_equal([[
            sum([g.has_score() for g in pl.games]) * int(pl in game.players)
            for pl in self.database.players.values()
        ] for game in self.database.games if game.has_score()], 0)

        # Result matrix is a 2D matrix of games x 1
        result_matrix = numpy.array([
            game.score / constants.GAME_MAX_SCORE
            for game in self.database.games if game.has_score()
        ]).T

        # Calculate the recency coefficient for each game
        last_game_year = self.database.games[-1].date.year
        self.recency_matrix = torch.from_numpy(numpy.array([
            math.exp((game.date.year - last_game_year) * 0.5)
            for game in self.database.games if game.has_score()
        ])).to(self.device)

        # Calculate how many participations qualify you to be in validation set
        minimum_participations = numpy.min(player_matrix, axis=1)
        training_set_percentage, split = 1, len(database.games) - 1
        while training_set_percentage > constants.TRAINING_SET_SIZE:
            training_set_size = (minimum_participations < split).sum()
            training_set_percentage = training_set_size / len(database.games)
            split -= 1

        # Convert to 2D PyTorch tensors on chosen device
        player_matrix[~player_matrix.mask] = 1
        indexes = numpy.where(minimum_participations >= split)[0]
        self.train_x = torch.tensor(
            numpy.delete(player_matrix[:, :], indexes, axis=0),
            dtype=torch.float32,
            device=self.device
        )
        self.train_y = torch.tensor(
            numpy.delete(result_matrix, indexes, axis=0),
            dtype=torch.float32,
            device=self.device
        ).reshape(-1, 1)

        self.validate_x = torch.tensor(
            player_matrix[indexes, :],
            dtype=torch.float32,
            device=self.device
        )
        self.validate_y = torch.tensor(
            result_matrix[indexes],
            dtype=torch.float32,
            device=self.device
        ).reshape(-1, 1)

        self.complete_x = torch.tensor(player_matrix, dtype=torch.float32, device=self.device)
        self.complete_y = torch.tensor(result_matrix, dtype=torch.float32, device=self.device)

    def _train(self, algorithm_type: type) -> tuple[typing.Any, float]:
        algorithm = algorithm_type(self.train_x.shape[1])
        algorithm.model.to(self.device)
        algorithm.train(
            self.train_x, self.train_y,
            self.validate_x, self.validate_y
        )

        prediction = algorithm.infer(self.complete_x)
        diff = (self.complete_y - prediction) * constants.GAME_MAX_SCORE
        loss = torch.sum(torch.square(diff * self.recency_matrix)).item()
        return algorithm, loss

    def train(
            self, algorithm_class: type,
            best_of: int = 100,
            with_multiprocessing: bool = False,
    ) -> None:

        print(f"Training on device: {self.device}")
        if self.device.type == 'cuda':
            print(f"GPU: {torch.cuda.get_device_name(0)}")

        algorithm_name = algorithm_class.__name__
        if not os.path.exists(algorithm_name):
            os.makedirs(algorithm_name)

        best_loss, best_model, loss_history = numpy.inf, None, []
        if with_multiprocessing:
            with ThreadPool() as pool:
                results = pool.imap_unordered(
                    self._train, itertools.repeat(algorithm_class, best_of))

                progress_bar = tqdm.tqdm(
                    results, desc='Training AI',
                    total=best_of, postfix={'best': numpy.inf}
                )

                for algorithm, loss in progress_bar:
                    loss_history.append(loss)
                    if loss < best_loss:
                        best_loss, self.algorithm = loss, algorithm
                        progress_bar.set_postfix({'best': best_loss})

        else:
            progress_bar = tqdm.tqdm(
                range(best_of), desc='Training AI',
                total=best_of, postfix={'best': numpy.inf}
            )

            for _ in progress_bar:
                algorithm, loss = self._train(algorithm_class)
                loss_history.append(loss)
                if loss < best_loss:
                    best_loss, self.algorithm = loss, algorithm
                    progress_bar.set_postfix({'best': best_loss})

        # Probability density function
        loss_history.sort()
        mean, std = numpy.mean(loss_history), numpy.std(loss_history)
        pdf = scipy.stats.norm.pdf(loss_history, mean, std)
        matplotlib.pyplot.plot(loss_history, pdf)

        # Vertical lines
        matplotlib.pyplot.axvline(x=best_loss, color='r', linestyle='--')
        matplotlib.pyplot.axvline(x=mean, color='y', linestyle='-')
        matplotlib.pyplot.axvline(x=mean + std, color='y', linestyle='--')
        matplotlib.pyplot.axvline(x=mean - std, color='y', linestyle='--')

        # Text
        matplotlib.pyplot.text(
            best_loss, numpy.max(pdf) / 2, f'Best: {best_loss:.0f}',
            horizontalalignment='right',
            verticalalignment='center',
            rotation='vertical'
        )
        matplotlib.pyplot.text(
            mean, numpy.max(pdf) / 2, f'Mean: {mean:.0f}',
            horizontalalignment='right',
            verticalalignment='center',
            rotation='vertical'
        )
        matplotlib.pyplot.text(
            mean + std, numpy.max(pdf) / 2, f'STD: {std:.0f}',
            horizontalalignment='right',
            verticalalignment='center',
            rotation='vertical'
        )

        # Labeling
        matplotlib.pyplot.title('Loss Distribution')
        matplotlib.pyplot.xlabel('Loss')
        matplotlib.pyplot.ylabel('Probability')

        # Saving
        name = f'{algorithm_name}/[{best_loss:4.0f}] Loss.png'
        matplotlib.pyplot.savefig(name)
        self.algorithm.save(f'[{best_loss:4.0f}] Model.pt')

    def load(self, algorithm_type: type, model_file) -> None:
        self.algorithm = algorithm_type(self.train_x.shape[1])
        self.algorithm.model.to(self.device)
        self.algorithm.load(model_file)

    # noinspection PyUnresolvedReferences
    def infer(self, game: 'Game') -> float:
        game_tensor = torch.tensor([
            int(player in game.players)
            for player in self.database.players.values()
        ], dtype=torch.float32, device=self.device).reshape(1, -1)

        prediction = self.algorithm.infer(game_tensor)
        return prediction.item() * constants.GAME_MAX_SCORE


class NeuralNetwork:
    def __init__(
            self, game_count: int, layer_count: int = 3,
            first_layer_function: typing.Callable = lambda x: x * 2,
            next_layer_function: typing.Callable = lambda x: x // 2,
            activation_function: typing.Callable = torch.nn.ReLU,
    ) -> None:
        self.device = device
        self.model = torch.nn.Sequential()

        node_count = game_count
        for layer_no in range(layer_count):
            next_node_count = first_layer_function(node_count) \
                if not layer_no else next_layer_function(node_count)

            self.model.add_module(
                f'l-{layer_no}', torch.nn.Linear(node_count, next_node_count))
            self.model.add_module(
                f'f-{layer_no}', activation_function())

            node_count = next_node_count

        self.model.add_module('l-scalar', torch.nn.Linear(node_count, 1))
        self.model.to(self.device)

    def train(
            self, train_x: torch.Tensor, train_y: torch.Tensor,
            validate_x: torch.Tensor, validate_y: torch.Tensor,
            loss_function: typing.Callable = torch.nn.MSELoss(),
            epochs: int = 1000, batch_size: int = 10,
    ) -> None:

        best_loss, best_weights = numpy.inf, None
        batches = torch.arange(0, len(train_x), batch_size)
        optimizer = torch.optim.Adam(self.model.parameters())

        for _ in range(epochs):
            self.model.train()

            for start in batches:

                # Take a batch
                batch_x = train_x[start:start + batch_size]
                batch_y = train_y[start:start + batch_size]

                # Forward pass
                prediction = self.model(batch_x)
                loss = loss_function(prediction, batch_y)

                # Backward pass
                optimizer.zero_grad()
                loss.backward()

                # Weight update
                optimizer.step()

            # Validate
            self.model.eval()
            prediction = self.model(validate_x)
            loss = loss_function(prediction, validate_y)

            # Save best weights
            if loss < best_loss:
                best_loss = loss
                best_weights = copy.deepcopy(self.model.state_dict())

        # Restore model and return best accuracy
        self.model.load_state_dict(best_weights)

    def infer(self, x: torch.Tensor) -> torch.Tensor:
        self.model.eval()
        with torch.inference_mode():
            x = x.to(self.device)
            return torch.flatten(self.model(x))

    def save(self, filename) -> None:
        torch.save(self.model.state_dict(), f'NeuralNetwork/{filename}')

    def load(self, filename) -> None:
        self.model.load_state_dict(torch.load(f'NeuralNetwork/{filename}', map_location=self.device))
