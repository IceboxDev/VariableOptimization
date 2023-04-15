# -*- coding: utf-8 -*-

from collections import defaultdict
from matplotlib import pyplot
from functools import total_ordering, partial
from datetime import date
from typing import List, Tuple, Set, Generator, DefaultDict
from typing import NoReturn, Optional, Union
from pandas import DataFrame, Series, read_csv, concat
from numpy import array, ndarray, dot, linspace, append
from scipy import optimize, interpolate
from enum import Enum
from math import log

WEIGHTS = read_csv('data.csv')
print("Finished loading dependencies")


class DefaultStrength(Enum):
    WEAK = 0
    AVERAGE = 1
    STRONG = 2


# Class for the participants of the pubquiz
@total_ordering
class Player:
    def __init__(self, name: str, strength: DefaultStrength) -> NoReturn:
        global WEIGHTS

        self.name = name
        self.strength = strength

        db_row = WEIGHTS.loc[WEIGHTS['name'] == self.name]
        if db_row.empty:
            db_row = DataFrame([[self.name, 0.1]], columns=['name', 'weight'])
            WEIGHTS = concat([WEIGHTS, db_row], axis=0, ignore_index=True)
            db_row = WEIGHTS.loc[WEIGHTS['name'] == self.name]

        self.index = db_row.index
        self.weight = db_row.loc[self.index, 'weight'].item()
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
    def strength(self):
        return self._strength

    @strength.setter
    def strength(self, value):
        self._strength = value

    @property
    def index(self) -> int:
        return self._index

    @index.setter
    def index(self, value: int) -> NoReturn:
        self._index = value

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
    def games(self, value) -> NoReturn:
        self._games = value

    def rename(self, new_name) -> NoReturn:
        global WEIGHTS

        WEIGHTS.loc[self.index, 'name'] = new_name
        self.name = new_name

        WEIGHTS = WEIGHTS.sort_values('name', ignore_index=True)
        WEIGHTS.to_csv('data.csv', index=False)
        exit()

    def add_game(self, value: 'Game') -> NoReturn:
        self.games.add(value)

    def get_match_history(self) -> Set['Game']:
        return self.games

    def get_score(self) -> float:
        return self.weight * Game.MAX_POINTS

    def first_timer(self) -> bool:
        return len(self.games) == 1

    def has_always_as_teammate(self, player: 'Player') -> bool:
        return all(player in g.players for g in self.games)


# Class for one quiz of pubquiz
class Game:

    MAX_POINTS = 66
    WEIGHT = WEIGHTS.loc[WEIGHTS['name'] == "~WEIGHT_A"]["weight"].item()

    def __init__(
            self,
            game_date: Optional[date],
            game_score: Optional[int],
            *players: Player
    ) -> NoReturn:

        self.date = game_date
        self.score = game_score
        self.players = players

        for player in self.players:
            player.add_game(self)

    def __hash__(self) -> int:
        return hash((self.date, self.score, self.players))

    @property
    def date(self):
        return self._date

    @date.setter
    def date(self, value: date):
        self._date = value

    @property
    def score(self) -> int:
        return self._score

    @score.setter
    def score(self, value: int) -> NoReturn:
        self._score = value

    @property
    def players(self) -> Tuple[Player]:
        return self._players

    @players.setter
    def players(self, value: Tuple[Player]) -> NoReturn:
        self._players = value

    def player_count(self) -> int:
        return len(self.players)

    def has_first_timers(self) -> bool:
        return any(player.first_timer() for player in self.players)

    def other_players(self, player: Player) -> Generator[Player, None, None]:
        for other_player in self.players:
            if other_player != player:
                yield other_player

    def get_player_indices(self) -> Generator[int, None, None]:
        for player in self.players:
            yield player.index

    def predict_score(self) -> int:
        individual = sum(player.get_score() for player in self.players)
        overlap = 1 - Game.WEIGHT * log(self.player_count())

        return round(individual * overlap)

    def calculate_score(self, weight: ndarray) -> ndarray:
        participations = array([
            int(i in self.get_player_indices()) for i in range(len(weight) - 1)
        ] + [0])
        overlap = 1 - weight[-1] * log(self.player_count())

        return dot(participations, weight) * overlap * Game.MAX_POINTS


# Class for the complete history of all quizzes
class History:

    def __init__(self, *games: Game) -> NoReturn:
        self.games: Tuple[Game] = games
        self.ucrts: DefaultDict[Player, List[List[Player]]] = defaultdict(list)

        for player in self.get_all_players():
            o = [
                oth for oth in self.get_all_players().difference({player}) if
                oth.get_match_history().issubset(player.get_match_history())
            ]

            for i in History.find_uncertainties(player.get_match_history(), o):
                self.ucrts[player].append(i)

    @staticmethod
    def find_uncertainties(
            goal: Set[Game],
            players: List[Player],
            path: Optional[List[Player]] = None
    ) -> Generator[List[Player], None, None]:

        if path is None:
            path = []

        players = [s for s in players if s.get_match_history().issubset(goal)]
        # players.sort(key=lambda p: len(p.get_match_history()))

        while players:
            p = players.pop()
            if p.get_match_history() == goal:
                yield [p]

            n_goal = goal.difference(p.get_match_history())
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

    def get_all_players(self) -> Set[Player]:
        return set(player for game in self.games for player in game.players)

    def _objective_function(

            self, w: Optional[Union[ndarray, float]], reset=False) -> float:
        print('test')
        if reset:
            players = sorted(self.get_all_players(), key=lambda p: p.index)
            strengths = array([player.strength.value for player in players])
            w = append(strengths * w, [Game.WEIGHT])

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

    def recalculate_weights(self, r: bool = False) -> optimize.OptimizeResult:
        global WEIGHTS

        WEIGHTS = WEIGHTS.sort_values('name', ignore_index=True)
        for player in self.get_all_players():
            db_row = WEIGHTS.loc[WEIGHTS['name'] == player.name]
            player.index = db_row.index

        weights = WEIGHTS.loc[:, 'weight'].to_numpy()

        if r:
            objective_function = partial(self._objective_function, reset=True)
            result = optimize.minimize_scalar(objective_function)
            players = sorted(self.get_all_players(), key=lambda p: p.index)
            arr = array([player.strength.value for player in players])
            WEIGHTS['weight'] = Series(append(arr * result.x, [Game.WEIGHT]))

        else:
            result = optimize.minimize(
                self._objective_function, weights,
                method='Nelder-Mead',
                options={'xatol': 1e-8, 'disp': False},
                bounds=optimize.Bounds(0)
            )

            WEIGHTS['weight'] = Series(result.x)

        WEIGHTS.to_csv('data.csv', index=False)
        for player in self.get_all_players():
            db_row = WEIGHTS.loc[WEIGHTS['name'] == player.name]
            player.weight = db_row.loc[player.index, 'weight'].item()

        weight_a = WEIGHTS.loc[WEIGHTS['name'] == '~WEIGHT_A']
        Game.WEIGHT = weight_a.loc[weight_a.index, 'weight'].item()

        return result

    def plot_history(self) -> NoReturn:
        x, y = list(range(len(self.games))), [i.score for i in self.games]
        pyplot.plot(x, y)

        spline = interpolate.make_interp_spline(x, y)

        X_ = linspace(0, len(self.games) - 1, 500)
        Y_ = spline(X_)
        pyplot.plot(X_, Y_)

        ius = interpolate.Rbf(x, y)
        Y_ = ius(X_)
        pyplot.plot(X_, Y_)



        #pyplot.gcf().autofmt_xdate()
        pyplot.show()

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


# Placeholder average player
# PAP1 = Player("Average Player 1", DefaultStrength.AVERAGE)
# PAP2 = Player("Average Player 2", DefaultStrength.AVERAGE)
# PAP3 = Player("Average Player 3", DefaultStrength.AVERAGE)
# PAP4 = Player("Average Player 4", DefaultStrength.AVERAGE)

print('off to a good start')
adolfo = Player("Adolfo", DefaultStrength.WEAK)
print('off to a good start 2')
adrian = Player("Adrian", DefaultStrength.AVERAGE)
agne = Player("Agnė", DefaultStrength.AVERAGE)
andi = Player("Andi", DefaultStrength.WEAK)
anni = Player("Anni", DefaultStrength.WEAK)
benedikt = Player("Benedikt", DefaultStrength.AVERAGE)
charles = Player("Charles", DefaultStrength.AVERAGE)
daniela = Player("Daniela", DefaultStrength.WEAK)
ella = Player("Ella", DefaultStrength.WEAK)
fabian_moe = Player("Fabian (Moe)", DefaultStrength.AVERAGE)
fabian_muc = Player("Fabian (Muc)", DefaultStrength.WEAK)
felix = Player("Felix", DefaultStrength.WEAK)
georg = Player("Georg", DefaultStrength.WEAK)
hendrik = Player("Hendrik", DefaultStrength.WEAK)
jana = Player("Jana", DefaultStrength.WEAK)
johannes = Player("Johannes", DefaultStrength.STRONG)
joseph = Player("Joseph", DefaultStrength.AVERAGE)
julian_moe = Player("Julian (Moe)", DefaultStrength.AVERAGE)
julian_muc = Player("Julian (Muc)", DefaultStrength.WEAK)
juergen = Player("Jürgen", DefaultStrength.AVERAGE)
laurin = Player("Laurin", DefaultStrength.AVERAGE)
leander = Player("Leander", DefaultStrength.AVERAGE)
leo = Player("Leo", DefaultStrength.WEAK)
linda = Player("Linda", DefaultStrength.WEAK)
luis = Player("Luis", DefaultStrength.STRONG)
mantas = Player("Mantas", DefaultStrength.WEAK)
milena = Player("Milena", DefaultStrength.AVERAGE)
mutlu = Player("Mutlu", DefaultStrength.WEAK)
nicolo = Player("Nicolo", DefaultStrength.STRONG)
patrick = Player("Patrick", DefaultStrength.AVERAGE)
paul = Player("Paul", DefaultStrength.STRONG)
philip = Player("Philip", DefaultStrength.AVERAGE)
riccardo = Player("Riccardo", DefaultStrength.WEAK)
richard_alt = Player("Richard (Alt)", DefaultStrength.WEAK)
richard_neu = Player("Richard (Neu)", DefaultStrength.WEAK)
roman = Player("Roman", DefaultStrength.AVERAGE)
sebastian_moe = Player("Sebastian (Moe)", DefaultStrength.STRONG)
sebastian_neu = Player("Sebastian (Neu)", DefaultStrength.WEAK)
simon_g = Player('Simon (G)', DefaultStrength.STRONG)
simon_moe = Player("Simon (Moe)", DefaultStrength.STRONG)
stephan = Player("Stephan", DefaultStrength.WEAK)
thomas = Player("Thomas", DefaultStrength.WEAK)
valeria = Player("Valeria", DefaultStrength.WEAK)
victor = Player("Victor", DefaultStrength.WEAK)
vinicius = Player("Vinicius", DefaultStrength.STRONG)

print('off to a good start 2')

history = History(
    Game(date(2022, 7 , 18), 23, mantas, paul, linda, agne),
    Game(date(2022, 7 , 25), 42, mantas, paul, nicolo, simon_moe, fabian_moe, richard_alt),
    Game(date(2022, 8 , 1 ), 34, mantas, luis, linda, roman, felix),
    Game(date(2022, 8 , 8 ), 29, mantas, luis, nicolo, vinicius, laurin),
    Game(date(2022, 8 , 15), 38, mantas, luis, juergen, johannes, julian_muc),
    Game(date(2022, 8 , 22), 40, mantas, benedikt, nicolo, philip, julian_moe),
    Game(date(2022, 8 , 29), 42, mantas, luis, nicolo, philip, patrick, joseph),
    Game(date(2022, 9 , 5 ), 38, mantas, luis, nicolo, linda, patrick, milena),
    Game(date(2022, 9 , 12), 44, mantas, paul, nicolo, linda, vinicius, milena, jana),
    Game(date(2022, 9 , 19), 31, mantas, simon_g, juergen, linda, milena),
    Game(date(2022, 9 , 26), 38, mantas, simon_g, luis, nicolo, laurin, simon_moe),
    Game(date(2022, 10, 3 ), 40, mantas, simon_g, milena, leo, riccardo, patrick, joseph, paul),
    Game(date(2022, 10, 10), 31, mantas, simon_moe, milena, joseph, paul, luis),
    Game(date(2022, 10, 17), 48, mantas, paul, nicolo, simon_moe),
    Game(date(2022, 10, 24), 41, mantas, paul, riccardo, linda, vinicius, richard_neu, adrian),
    Game(date(2022, 10, 31), 43, mantas, paul, adrian, ella, stephan, sebastian_moe, charles, simon_g),
    Game(date(2022, 11, 7 ), 36, mantas, luis, andi, anni, vinicius, nicolo),
    Game(date(2022, 11, 14), 36, mantas, laurin, milena, nicolo, patrick, thomas),
    Game(date(2022, 11, 21), 37, mantas, nicolo, vinicius, paul, georg, simon_g),
    Game(date(2022, 11, 28), 40, mantas, nicolo, paul, linda, luis, sebastian_neu),
    Game(date(2022, 12,  5), 36, mantas, nicolo, paul, linda, patrick),
    Game(date(2022, 12, 12), 34, mantas, paul, linda, patrick, adrian, juergen),
    Game(date(2022, 12, 19), 46, mantas, nicolo, paul, luis, thomas, riccardo, simon_moe, julian_moe, mutlu),
    Game(date(2023, 1 ,  2), 33, mantas, johannes, adrian, felix, hendrik, fabian_muc, leander),
    Game(date(2023, 1 ,  9), 46, mantas, nicolo, paul, luis, linda, vinicius, simon_g, simon_moe),
    Game(date(2023, 1 , 16), 40, mantas, nicolo, paul, milena, patrick),
    Game(date(2023, 1 , 23), 45, mantas, nicolo, paul, philip, adolfo, victor, riccardo),
    Game(date(2023, 1 , 30), 42, mantas, paul, adrian, riccardo, daniela, valeria),
    Game(date(2023, 2 ,  6), 50, mantas, thomas, luis, paul, patrick, philip, daniela),
    Game(date(2023, 2 , 13), 33, mantas, valeria, riccardo, adrian, simon_moe, laurin),
    Game(date(2023, 2 , 20), 28, patrick, nicolo, luis, milena),
)

if __name__ == "__main__":
    print('hello')
    print(history._objective_function(None))
    history.get_scoreboard()

    for _game in history.games:
        print(_game.score, _game.predict_score())


    # history.plot_history()
    # history.recalculate_weights()
    # strong = [nicolo.index, luis.index, johannes.index, vinicius.index,
    #           sebastian_moe.index, simon_moe.index, simon_g.index, paul.index]

    # while True:
    #     print(history.recalculate_weights().fun)
