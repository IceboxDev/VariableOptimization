# -*- coding: utf-8 -*-

from multiprocessing.pool import ThreadPool

import constants
import itertools
import typing
import numpy
import torch
import copy
import tqdm
import os


class ArtificialIntelligence:

    # noinspection PyUnresolvedReferences
    def __init__(self, database: 'Database') -> None:
        self.database = database
        self.algorithm = None

        # Player matrix is a 2D matrix of games x players
        player_matrix = numpy.ma.masked_equal([[
            len(player.games) * int(player in game.players)
            for player in self.database.players.values()
        ] for game in self.database.games], 0)

        # Result matrix is a 2D matrix of games x 1
        result_matrix = numpy.array([
            game.score / constants.GAME_MAX_SCORE
            for game in self.database.games
        ]).T

        # Calculate how many participations qualify you to be in validation set
        minimum_participations = numpy.min(player_matrix, axis=1)
        training_set_percentage, split = 1, len(database.games) - 1
        while training_set_percentage > constants.TRAINING_SET_SIZE:
            training_set_size = (minimum_participations < split).sum()
            training_set_percentage = training_set_size / len(database.games)
            split -= 1

        # Convert to 2D PyTorch tensors
        player_matrix[~player_matrix.mask] = 1
        indexes = numpy.where(minimum_participations >= split)[0]
        self.train_x = torch.tensor(
            numpy.delete(player_matrix[:, :], indexes, axis=0),
            dtype=torch.float32)
        self.train_y = torch.tensor(
            numpy.delete(result_matrix, indexes, axis=0),
            dtype=torch.float32).reshape(-1, 1)

        self.validate_x = torch.tensor(
            player_matrix[indexes, :], dtype=torch.float32)
        self.validate_y = torch.tensor(
            result_matrix[indexes], dtype=torch.float32).reshape(-1, 1)

        self.complete_x = torch.tensor(player_matrix, dtype=torch.float32)
        self.complete_y = torch.tensor(result_matrix, dtype=torch.float32)

    def _train(self, algorithm_type: type) -> tuple[typing.Any, float]:
        algorithm = algorithm_type(self.train_x.shape[1])
        algorithm.train(
            self.train_x, self.train_y,
            self.validate_x, self.validate_y
        )

        prediction = algorithm.infer(self.complete_x)
        diff = (self.complete_y - prediction) * constants.GAME_MAX_SCORE
        loss = torch.sum(torch.square(diff)).item()
        return algorithm, loss

    def train(self, algorithm_type: type, best_of: int = 100) -> None:
        algorithm_name = algorithm_type.__name__
        if not os.path.exists(algorithm_name):
            os.makedirs(algorithm_name)

        best_loss, best_model = numpy.inf, None
        with ThreadPool() as pool:
            results = pool.imap_unordered(
                self._train, itertools.repeat(algorithm_type, best_of))

            progress_bar = tqdm.tqdm(
                results, desc='Training AI',
                total=best_of, postfix={'best': numpy.inf}
            )

            for algorithm, loss in progress_bar:
                if loss < best_loss:
                    best_loss, self.algorithm = loss, algorithm
                    progress_bar.set_postfix({'best': best_loss})

        self.algorithm.save('model.pt')

    def load(self, algorithm_type: type, model_file) -> None:
        self.algorithm = algorithm_type(self.train_x.shape[1])
        self.algorithm.load(model_file)

    # noinspection PyUnresolvedReferences
    def infer(self, game: 'Game') -> float:
        game_tensor = torch.tensor([
            int(player in game.players)
            for player in self.database.players.values()
        ], dtype=torch.float32).reshape(1, -1)

        return self.algorithm.infer(game_tensor).item() * constants.GAME_MAX_SCORE


class NeuralNetwork:
    def __init__(
            self, game_count: int, layer_count: int = 3,
            first_layer_function: typing.Callable = lambda x: x * 2,
            next_layer_function: typing.Callable = lambda x: x // 2,
            activation_function: typing.Callable = torch.nn.ReLU,
    ) -> None:
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
            return torch.flatten(self.model(x))

    def save(self, filename) -> None:
        torch.save(self.model.state_dict(), f'NeuralNetwork/{filename}')

    def load(self, filename) -> None:
        self.model.load_state_dict(torch.load(f'NeuralNetwork/{filename}'))
