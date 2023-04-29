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
        print(flat)
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

        self.date = game_date
        self.duration = game_duration
        self.score = game_score
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
        participations = numpy.array([
            int(i in self.get_player_indices()) for i in range(len(weight) - 1)
        ] + [0])
        overlap = 1 - weight[-1] * math.log(self.player_count())

        return numpy.dot(participations, weight) * overlap * Game.MAX_POINTS


# Class for the complete history of all quizzes
class History:

    def __init__(self, *games: Game) -> None:
        self.games: Tuple[Game] = games
        self.ucrts: DefaultDict[Player, List[List[Player]]] = collections.defaultdict(list)

        for player in self.get_all_players():
            o = [
                oth for oth in self.get_all_players().difference({player}) if
                oth.games.issubset(player.games)
            ]

            for i in History.find_uncertainties(player.games, o):
                self.ucrts[player].append(i)

    @staticmethod
    def find_uncertainties(
            goal: Set[Game],
            players: List[Player],
            path: Optional[List[Player]] = None
    ) -> Generator[List[Player], None, None]:

        if path is None:
            path = []

        while players:
            p = players.pop()
            if not p.games.issubset(goal):
                continue

            if p.games == goal:
                yield [p]

            n_goal = goal.difference(p.games)
            for n_path in History.find_uncertainties(n_goal, players, path+[p]):
                yield n_path + [p]

    def is_uncertain(self, player: Player) -> Tuple[bool, bool]:
        induced = len(self.ucrts[player]) != 0
        inducer = any(
            any(
                player in player_combination for player_combination in pc_list
            ) for pc_list in self.ucrts.values()
        )

        return inducer, induced

    def is_certain(self, player: Player) -> bool:
        return all(ucrt is False for ucrt in self.is_uncertain(player))

    # @staticmethod
    # def get_common_knowledge(player_number: int) -> float:
    #     common_knowledge = Game.WEIGHT_A * log(player_number) + Game.WEIGHT_B
    #     return common_knowledge * Game.MAX_POINTS

    def get_min_max(self, player: Player) -> Tuple[float, float]:
        try:
            minimum = player.weight - max((
                min(comb, key=lambda p: p.weight)
                for comb in [a for b in self.ucrts.values() for a in b]
                if player in comb), key=lambda p: p.weight
            ).weight
        except ValueError:
            minimum = player.weight

        try:
            maximum = player.weight + max((
                min(comb, key=lambda p: p.weight)
                for comb in self.ucrts[player]),
                key=lambda p: p.weight
            ).weight
        except ValueError:
            maximum = player.weight

        return minimum, maximum

    def _objective_function(

            self, w: Optional[Union[numpy.ndarray, float]], reset=False) -> float:
        print('test')
        if reset:
            players = sorted(self.get_all_players(), key=lambda p: p.index)
            strengths = numpy.array([player.strength.value for player in players])
            w = numpy.append(strengths * w, [Game.WEIGHT])

        if w is None:
            return sum(
                (g.score - g.predict_score()) ** 2 for g in self.games
            )

        return sum(
            (g.score - g.calculate_score(w)) ** 2 for g in self.games
        ) + (w[-1] * (Game.MAX_POINTS / 2))

    def minimize_weight(self, player: Player, set_to=None) -> NoReturn:
        global WEIGHTS

        file_name = f'data_{player.name.lower()}_minimized.csv'
        p_to_max, p_to_min, by = max((
            (p, comb, min(comb, key=lambda x: x.weight).weight)
            for p, combinations in self.ucrts.items()
            for comb in combinations if player in comb
        ), key=lambda x: x[2])

        if set_to is not None:
            change = player.weight - set_to / Game.MAX_POINTS
            assert(change <= by)
            by = change
            file_name = f'data_{player.name.lower()}_set_to_{set_to}.csv'

        WEIGHTS.loc[p_to_max.index, 'weight'] = p_to_max.weight + by
        for p in p_to_min:
            WEIGHTS.loc[p.index, 'weight'] = p.weight - by
        WEIGHTS.to_csv(file_name, index=False)

    def maximize_weight(self, player: Player, set_to=None) -> NoReturn:
        global WEIGHTS

        file_name = f'data_{player.name.lower()}_maximized.csv'
        p_to_max, p_to_min, by = max((
            (player, comb, min(comb, key=lambda x: x.weight).weight)
            for comb in self.ucrts[player]
        ), key=lambda x: x[2])

        if set_to is not None:
            change = set_to / Game.MAX_POINTS - player.weight
            assert (change <= by)
            by = change
            file_name = f'data_{player.name.lower()}_set_to_{set_to}.csv'

        WEIGHTS.loc[p_to_max.index, 'weight'] = p_to_max.weight + by
        for p in p_to_min:
            WEIGHTS.loc[p.index, 'weight'] = p.weight - by
        WEIGHTS.to_csv(file_name, index=False)

    def recalculate_weights(self, r: bool = False) -> scipy.optimize.OptimizeResult:
        global WEIGHTS

        WEIGHTS = WEIGHTS.sort_values('name', ignore_index=True)
        for player in self.get_all_players():
            db_row = WEIGHTS.loc[WEIGHTS['name'] == player.name]
            player.index = db_row.index

        weights = WEIGHTS.loc[:, 'weight'].to_numpy()

        if r:
            objective_function = functools.partial(self._objective_function, reset=True)
            result = scipy.optimize.minimize_scalar(objective_function)
            players = sorted(self.get_all_players(), key=lambda p: p.index)
            arr = numpy.array([player.strength.value for player in players])
            WEIGHTS['weight'] = pandas.Series(numpy.append(arr * result.x, [Game.WEIGHT]))

        else:
            result = scipy.optimize.minimize(
                self._objective_function, weights,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'disp': False},
                bounds=scipy.optimize.Bounds(0)
            )

            WEIGHTS['weight'] = pandas.Series(result.x)

        WEIGHTS.to_csv('data.csv', index=False)
        for player in self.get_all_players():
            db_row = WEIGHTS.loc[WEIGHTS['name'] == player.name]
            player.weight = db_row.loc[player.index, 'weight'].item()

        weight_a = WEIGHTS.loc[WEIGHTS['name'] == '~WEIGHT_A']
        Game.WEIGHT = weight_a.loc[weight_a.index, 'weight'].item()

        return result

    def plot_history(self) -> NoReturn:
        x, y = list(range(len(self.games))), [i.score for i in self.games]
        matplotlib.pyplot.plot(x, y)

        spline = scipy.interpolate.make_interp_spline(x, y)

        X_ = numpy.linspace(0, len(self.games) - 1, 500)
        Y_ = spline(X_)
        matplotlib.pyplot.plot(X_, Y_)

        ius = scipy.interpolate.Rbf(x, y)
        Y_ = ius(X_)
        matplotlib.pyplot.plot(X_, Y_)



        #pyplot.gcf().autofmt_xdate()
        matplotlib.pyplot.show()

    def get_uncertainties(self, player: Player) -> NoReturn:
        flat = [(p, a) for p, l in self.ucrts.items() for a in l if player in a]
        unique_players = set([a.name for b in flat for a in b[1]])
        if player.name in unique_players:
            unique_players.remove(player.name)
            print(f'People he/she should play with: '
                  f'{", ".join(unique_players)}')

        for underrated_player, comb in flat:
            worst_player = min(comb, key=lambda x: x.weight)
            print(
                f'-{worst_player.weight * Game.MAX_POINTS:>5,.2f} | '
                f' {underrated_player.name:<8} |'
                f' {*comb,}'
            )

        flat = set([p.name for comb in self.ucrts[player] for p in comb])
        if flat:
            print(f'People that should play with each other: {", ".join(flat)}')

        for comb in self.ucrts[player]:
            worst_player = min(comb, key=lambda x: x.weight)
            print(
                f'+{worst_player.weight * Game.MAX_POINTS:>5,.2f} |'
                f' {worst_player.name:<8} |'
                f' {*comb,}'
            )

    def get_scoreboard(self) -> NoReturn:
        players = sorted(list(self.get_all_players()), reverse=True)
        for idx, player in enumerate(players, start=1):
            minimum, maximum = self.get_min_max(player)
            print(
                f'#{idx:<2} |'
                f' {player.get_score():>5,.2f} |'
                f' {minimum * Game.MAX_POINTS:>5,.2f} -'
                f' {maximum * Game.MAX_POINTS:>5,.2f} |'
                f' {str(self.is_certain(player)):<5} | '
                f'({len(player.get_match_history()):02}) |'
                f' {player.name}'
            )
        print()


if __name__ == "__main__":
    database = Database('optimal-timer-234608-a9f776f9605a.json')
    ai = artificial_intelligence.ArtificialIntelligence(database)
    ai.train(artificial_intelligence.NeuralNetwork, 12)

    # ai.load(artificial_intelligence.NeuralNetwork, 'model.pt')
    # for game in database.games:
    #     print(game.score, round(ai.infer(game)), round(ai.infer(game) - game.score))
