# -*- coding: utf-8 -*-

from oauth2client.service_account import ServiceAccountCredentials
from sklearn.ensemble import RandomForestRegressor
from typing import List, Tuple, Set, Dict, Generator, DefaultDict
from typing import NoReturn, Optional, Union

import artificial_intelligence
import constants

import collections
import matplotlib
import statistics
import functools
import datetime
import gspread
import pandas
import numpy
import scipy
import math


# Class for the database of pub-quiz
class Database:

    # Initialize the database with the credentials
    def __init__(self, credentials_path: str) -> None:
        credentials = ServiceAccountCredentials \
            .from_json_keyfile_name(credentials_path)

        self.client = gspread.authorize(credentials)
        self.sheet = self.client.open(constants.SPREADSHEET_NAME)
        self.sheet = self.sheet.worksheet(constants.WORKSHEET_NAME)

        response = self.sheet.batch_get([
            f'{Database.index_to_cell(*constants.CELL_PLAYER_GRID_L)}:'
            f'{Database.index_to_cell(*constants.CELL_PLAYER_GRID_R)}',
            Database.index_to_cell(*constants.CELL_GAME_DATES),
            Database.index_to_cell(*constants.CELL_GAME_DURATIONS),
            Database.index_to_cell(*constants.CELL_GAME_SCORES),
            Database.index_to_cell(*constants.CELL_WEIGHTS_NAME),
            Database.index_to_cell(*constants.CELL_WEIGHTS_WEIGHT),
        ])

        self.game_list, self.game_dates, self.game_durations, self.game_scores,\
            self.weight_players, self.weight_weights = [response[0]] + [
                tuple(x for y in out_lst for x in y) for out_lst in response[1:]
            ]

        self.overlap = float("nan")
        self.players = {}
        self.games = []

        self.generate_players()
        self.generate_games()

    @staticmethod
    # Convert the indices of row and column to the corresponding cell
    def index_to_cell(
            row: Optional[int],
            column: Optional[int],
            row_offset: Optional[int],
            column_offset: Optional[int],
    ) -> str:

        if column is not None:
            div, mod = divmod(column, 26)
            prefix_letter = chr(div + ord("A") - 1) if div else ""
            column_cell = prefix_letter + chr(mod + ord("A") - 1)

        else:
            return f"{row}{row_offset}:{row}"

        if row is None:
            return f"{column_cell}{column_offset}:{column_cell}"

        else:
            return f"{column_cell}{row}"

    @property
    def overlap(self) -> float:
        return self._overlap

    @overlap.setter
    def overlap(self, overlap: float) -> None:
        self._overlap = overlap

    @property
    def players(self) -> Dict[str, "Player"]:
        return self._players

    @players.setter
    def players(self, players: Dict[str, "Player"]) -> None:
        self._players = players

    @property
    def games(self) -> List["Game"]:
        return self._games

    @games.setter
    def games(self, games: List["Game"]) -> None:
        self._games = games

    # Generate the players and their weights
    def generate_players(self) -> None:
        flat = sorted((set(player for pl in self.game_list for player in pl)))
        flat.remove('N/A')

        for name in flat:
            if name in self.weight_players:
                player_index = self.weight_players.index(name)
                weight = float(self.weight_weights[player_index])
                self.players[name] = Player(name, weight)

            else:
                self.players[name] = Player(name, 0.0)

        overlap_index = self.weight_players.index('Overlap')
        self.overlap = float(self.weight_weights[overlap_index])

    # Generate the games and their scores
    def generate_games(self) -> None:
        for g_idx, team in enumerate(self.game_list):
            date = datetime.datetime.strptime(
                self.game_dates[g_idx], '%m/%d/%y').date()

            if len(self.game_durations) > g_idx and self.game_durations[g_idx]:
                start, end = self.game_durations[g_idx].split(' - ')
                start = datetime.datetime.strptime(start, '%H:%M').time()
                end = datetime.datetime.strptime(end, '%H:%M').time()
                duration = datetime.datetime.combine(datetime.date.min, end) - \
                    datetime.datetime.combine(datetime.date.min, start)

                if duration.days == -1:
                    duration += datetime.timedelta(days=1)

            else:
                duration = None

            if len(self.game_scores) <= g_idx:
                continue

            score = int(self.game_scores[g_idx])
            players = [self.players[name] for name in team if name != 'N/A']
            self.games.append(Game(date, duration, score, *players))

    # Save the weights of the players into the database
    def save_weights(self) -> None:
        range_name_l = Database.index_to_cell(
            constants.COLUMN_PADDING,
            constants.WEIGHTS_NAMES_COLUMN, None, None
        )
        range_name_r = Database.index_to_cell(
            constants.COLUMN_PADDING + len(self.players),
            constants.WEIGHTS_WEIGHTS_COLUMN, None, None
        )

        self.sheet.update(
            f'{range_name_l}:{range_name_r}',
            [[name, player.weight] for name, player in self.players.items()] +
            [['Overlap', self.overlap]]
        )

    # Recalculate the weights of the players
    def recalculate_weights(self, reset=False) -> scipy.optimize.OptimizeResult:
        if reset:
            loss = lambda weights: numpy.sum(numpy.square(numpy.multiply(numpy.sum(self.player_matrix, axis=1) * weights[0], 1 - numpy.log(numpy.sum(self.player_matrix, axis=1)) * weights[1]) - self.result_matrix)) + (weights[1] * constants.GAME_MAX_SCORE)
            result = scipy.optimize.minimize(
                loss, numpy.array([0, 0]),
                method='Nelder-Mead',
                options={'xatol': 1e-18, 'disp': False},
                bounds=scipy.optimize.Bounds(0, 1)
            )
            for player in self.players.values():
                player.weight = result.x[0]

            self.overlap = result.x[-1]
            # self.save_weights()
            return result


        else:
            loss = lambda weights, *args: numpy.sum(numpy.square((numpy.multiply(numpy.dot(
                self.player_matrix, weights.T
            ), 1 - numpy.log(numpy.sum(self.player_matrix, axis=1)) * weights[-1]) - self.result_matrix) * 66)) + (weights[-1] * constants.GAME_MAX_SCORE)

            result = scipy.optimize.differential_evolution(
                loss,
                [(0, 1)] * (len(self.players) + 1),
            )

            for idx, player in enumerate(self.players.values()):
                player.weight = result.x[idx]

            self.overlap = result.x[-1]
            # self.save_weights()
            return result

    def random_forest(self) -> None:
        R = 1000
        Q = []

        while R > 200:
            self.regressor = RandomForestRegressor()
            # X_train, X_test, y_train, y_test = train_test_split(
            #     self.player_matrix[:,:-1], self.result_matrix, test_size=0.333
            # )

            # self.bonus_column = numpy.array([numpy.sum(self.player_matrix, axis=1)]).T
            # self.input = numpy.concatenate((self.player_matrix[:,:-1], self.bonus_column), axis=1)
            self.input = self.player_matrix[:,:-1]
            self.regressor.fit(self.input, self.result_matrix)
            self.prediction = self.regressor.predict(self.input)
            self.a = (self.result_matrix - self.prediction) * 66
            R = numpy.sum(numpy.square(self.a))
            Q.append(R)
            print(f'{R}\t{statistics.mean(Q)}')


# Class for the participants of the pub-quiz
@functools.total_ordering
class Player:

    # Initialize the player with a name and a strength
    def __init__(self, player_name: str, player_weight: float) -> None:
        self.name = player_name
        self.weight = player_weight
        self.games = set()

    def __hash__(self) -> int:
        return hash((self.name, self.weight))

    def __eq__(self, other: 'Player') -> bool:
        return self.__hash__() == other.__hash__()

    def __lt__(self, other: 'Player') -> bool:
        return self.weight < other.weight

    def __repr__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> NoReturn:
        self._name = value

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> NoReturn:
        self._weight = value

    @property
    def games(self) -> Set['Game']:
        return self._games

    @games.setter
    def games(self, value: Set) -> NoReturn:
        self._games = value

    def add_game(self, value: 'Game') -> NoReturn:
        self.games.add(value)

    def get_score(self) -> float:
        return self.weight * Game.MAX_POINTS

    def first_timer(self) -> bool:
        return len(self.games) == 1

    def has_always_as_teammate(self, player: 'Player') -> bool:
        return all(player in game.players for game in self.games)


# Class for one quiz of pub-quiz
class Game:
    def __init__(
            self,
            game_date: Optional[datetime.date],
            game_duration: Optional[datetime.timedelta],
            game_score: Optional[int],
            *players: Player
    ) -> None:

        self.date: Optional[datetime.date] = game_date
        self.duration: Optional[datetime.timedelta] = game_duration
        self.score: Optional[int] = game_score
        self.players = players

        for player in self.players:
            player.add_game(self)

    def __hash__(self) -> int:
        return hash((self.date, self.duration, self.score, self.players))

    @property
    def date(self) -> datetime.date:
        return self._date

    @date.setter
    def date(self, value: date) -> None:
        self._date = value

    @property
    def duration(self) -> datetime.timedelta:
        return self._duration

    @duration.setter
    def duration(self, value: datetime.timedelta) -> None:
        self._duration = value

    @property
    def score(self) -> int:
        return self._score

    @score.setter
    def score(self, value: int) -> None:
        self._score = value

    @property
    def players(self) -> Tuple[Player]:
        return self._players

    @players.setter
    def players(self, value: Tuple[Player]) -> None:
        self._players = value

    def player_count(self) -> int:
        return len(self.players)

    def has_score(self) -> bool:
        return self.score != -1

    def has_first_timers(self) -> bool:
        return any(player.first_timer() for player in self.players)

    def other_players(self, player: Player) -> Generator[Player, None, None]:
        for other_player in self.players:
            if other_player != player:
                yield other_player

    def predict_score(self, db: Database) -> int:
        individual = sum(player.get_score() for player in self.players)
        overlap = 1 - db.overlap * math.log(self.player_count())

        return round(individual * overlap)

    def calculate_score(self, weight: numpy.ndarray) -> numpy.ndarray:
        particips = numpy.array([
            int(i in self.get_player_indices()) for i in range(len(weight) - 1)
        ] + [0])
        overlap = 1 - weight[-1] * math.log(self.player_count())

        return numpy.dot(particips, weight) * overlap * constants.GAME_MAX_SCORE

if __name__ == "__main__":
    database = Database('.config/personal-433622-62e046c7be64.json')
    ai = artificial_intelligence.ArtificialIntelligence(database)
    ai.train(artificial_intelligence.NeuralNetwork, best_of=100)

    # ai.load(artificial_intelligence.NeuralNetwork, '[ 457] Model.pt')
    # for game in database.games:
    #     print(game.score, round(ai.infer(game)), round(ai.infer(game) - game.score))

    # ai.save()
