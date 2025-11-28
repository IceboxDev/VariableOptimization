# -*- coding: utf-8 -*-
# modules.py

from oauth2client.service_account import ServiceAccountCredentials
from typing import List, Tuple, Set, Dict, Optional

import datetime
import gspread
import pickle
import time
import os

import constants


class Player:
    """Participant of the pub quiz."""

    def __init__(self, player_name: str, player_weight: float) -> None:
        self.name = player_name
        self.weight = player_weight
        self.games: Set["Game"] = set()

    def __hash__(self) -> int:
        return hash((self.name, self.weight))

    def __eq__(self, other: "Player") -> bool:
        return self.__hash__() == other.__hash__()

    def __lt__(self, other: "Player") -> bool:
        return self.weight < other.weight

    def __repr__(self) -> str:
        return self.name

    @property
    def name(self) -> str:
        return self._name

    @name.setter
    def name(self, value: str) -> None:
        self._name = value

    @property
    def weight(self) -> float:
        return self._weight

    @weight.setter
    def weight(self, value: float) -> None:
        self._weight = value

    @property
    def games(self) -> Set["Game"]:
        return self._games

    @games.setter
    def games(self, value: Set["Game"]) -> None:
        self._games = value

    def add_game(self, value: "Game") -> None:
        self.games.add(value)

    def get_score(self) -> float:
        # Uses Game.MAX_POINTS (see Game below)
        return self.weight * Game.MAX_POINTS  # type: ignore[attr-defined]

    def first_timer(self) -> bool:
        return len(self.games) == 1


class Game:
    """Represents one pub-quiz game."""

    # This constant is used by Player.get_score()
    MAX_POINTS = constants.GAME_MAX_SCORE

    def __init__(
            self,
            game_date: Optional[datetime.date],
            game_duration: Optional[datetime.timedelta],
            game_score: Optional[int],
            *players: Player,
            is_anomaly: bool = False,
    ) -> None:
        self.date: Optional[datetime.date] = game_date
        self.duration: Optional[datetime.timedelta] = game_duration
        self.score: Optional[int] = game_score
        self.players: Tuple[Player, ...] = players
        self.is_anomaly: bool = is_anomaly

        for player in self.players:
            player.add_game(self)

    def __hash__(self) -> int:
        return hash((self.date, self.duration, self.score, self.players, self.is_anomaly))

    @property
    def date(self) -> Optional[datetime.date]:
        return self._date

    @date.setter
    def date(self, value: Optional[datetime.date]) -> None:
        self._date = value

    @property
    def duration(self) -> Optional[datetime.timedelta]:
        return self._duration

    @duration.setter
    def duration(self, value: Optional[datetime.timedelta]) -> None:
        self._duration = value

    @property
    def score(self) -> Optional[int]:
        return self._score

    @score.setter
    def score(self, value: Optional[int]) -> None:
        self._score = value

    @property
    def players(self) -> Tuple[Player, ...]:
        return self._players

    @players.setter
    def players(self, value: Tuple[Player, ...]) -> None:
        self._players = value

    @property
    def is_anomaly(self) -> bool:
        return self._is_anomaly

    @is_anomaly.setter
    def is_anomaly(self, value: bool) -> None:
        self._is_anomaly = value

    def player_count(self) -> int:
        return len(self.players)

    def has_score(self) -> bool:
        return self.score != -1


class Database:
    """
    Database abstraction over the Google Sheet, with a 3-hour local cache.

    - On init:
        * if use_cache and cache is younger than CACHE_TTL → use cached data
        * else → pull fresh data from Google Sheets and write cache
    - You can force bypassing the cache with force_refresh=True
    - You can later force a re-pull via refresh_from_google()
    """

    CACHE_TTL = datetime.timedelta(hours=3)

    def __init__(
            self,
            credentials_path: str,
            use_cache: bool = True,
            force_refresh: bool = False,
            cache_path: str = ".pubquiz_database.cache",
    ) -> None:
        self._cache_path = cache_path
        self._credentials_path = credentials_path

        # Always authorise so write-backs (save_weights) still work
        credentials = ServiceAccountCredentials.from_json_keyfile_name(
            credentials_path
        )
        self.client = gspread.authorize(credentials)
        self.sheet = (
            self.client.open(constants.SPREADSHEET_NAME)
            .worksheet(constants.WORKSHEET_NAME)
        )

        # Try cache first
        snapshot: Optional[Dict[str, object]] = None
        if use_cache and not force_refresh:
            snapshot = self._load_cache_if_valid()

        if snapshot is None:
            # Pull fresh data from Google Sheets
            self._load_from_google()
            self._save_cache()
        else:
            # Use cached snapshot
            self._load_from_snapshot(snapshot)

        # Derived objects
        self.overlap = float("nan")
        self.players: Dict[str, Player] = {}
        self.games: List[Game] = []

        self.generate_players()
        self.generate_games()

    # --- Cache helpers ------------------------------------------------------

    def _cache_file_exists(self) -> bool:
        return os.path.exists(self._cache_path)

    def _load_cache_if_valid(self) -> Optional[Dict[str, object]]:
        if not self._cache_file_exists():
            return None

        try:
            with open(self._cache_path, "rb") as fh:
                data = pickle.load(fh)
        except Exception:
            # Corrupt / incompatible cache → ignore
            return None

        ts = data.get("timestamp")
        snapshot = data.get("snapshot")
        if ts is None or snapshot is None:
            return None

        age_seconds = time.time() - float(ts)
        if age_seconds > self.CACHE_TTL.total_seconds():
            return None

        return snapshot  # type: ignore[return-value]

    def _build_snapshot(self) -> Dict[str, object]:
        return {
            "game_list": self.game_list,
            "game_dates": self.game_dates,
            "game_durations": self.game_durations,
            "game_scores": self.game_scores,
            "game_anomalies": self.game_anomalies,
            "weight_players": self.weight_players,
            "weight_weights": self.weight_weights,
        }

    def _save_cache(self) -> None:
        try:
            payload = {
                "timestamp": time.time(),
                "snapshot": self._build_snapshot(),
            }
            with open(self._cache_path, "wb") as fh:
                # https://youtrack.jetbrains.com/projects/PY/issues/PY-81830/
                # noinspection PyTypeChecker
                pickle.dump(payload, fh)

        except Exception:
            pass

    def _load_from_snapshot(self, snapshot: Dict[str, object]) -> None:
        self.game_list = snapshot["game_list"]
        self.game_dates = snapshot["game_dates"]
        self.game_durations = snapshot["game_durations"]
        self.game_scores = snapshot["game_scores"]
        self.weight_players = snapshot["weight_players"]
        self.weight_weights = snapshot["weight_weights"]

        # Handle older cache files that didn't yet contain anomalies
        anomalies = snapshot.get("game_anomalies")
        if anomalies is None:
            # default: no anomalies
            self.game_anomalies = tuple("" for _ in range(len(self.game_dates)))
        else:
            self.game_anomalies = anomalies

    def _load_from_google(self) -> None:
        """Pull raw values from Google Sheets into the same fields as before."""
        response = self.sheet.batch_get(
            [
                f"{Database.index_to_cell(*constants.CELL_PLAYER_GRID_L)}:"
                f"{Database.index_to_cell(*constants.CELL_PLAYER_GRID_R)}",
                Database.index_to_cell(*constants.CELL_GAME_DATES),
                Database.index_to_cell(*constants.CELL_GAME_DURATIONS),
                Database.index_to_cell(*constants.CELL_GAME_SCORES),
                Database.index_to_cell(*constants.CELL_GAME_ANOMALIES),
                Database.index_to_cell(*constants.CELL_WEIGHTS_NAME),
                Database.index_to_cell(*constants.CELL_WEIGHTS_WEIGHT),
            ]
        )

        (
            self.game_list,
            self.game_dates,
            self.game_durations,
            self.game_scores,
            self.game_anomalies,
            self.weight_players,
            self.weight_weights,
        ) = [response[0]] + [
            tuple(x for y in out_lst for x in y) for out_lst in response[1:]
        ]

    def refresh_from_google(self, save_to_cache: bool = True) -> None:
        """
        Force a re-read from Google Sheets, ignoring the cache.

        Call this when you *know* the sheet changed and you want up-to-date
        data immediately, without waiting for the 3-hour TTL.
        """
        self._load_from_google()
        if save_to_cache:
            self._save_cache()

        # Rebuild derived objects
        self.overlap = float("nan")
        self.players = {}
        self.games = []
        self.generate_players()
        self.generate_games()

    # --- Existing API (unchanged behaviour) ---------------------------------

    @staticmethod
    def index_to_cell(
            row: Optional[int],
            column: Optional[int],
            row_offset: Optional[int],
            column_offset: Optional[int],
    ) -> str:
        # Same logic as in your original code
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
    def players(self) -> Dict[str, Player]:
        return self._players

    @players.setter
    def players(self, players: Dict[str, Player]) -> None:
        self._players = players

    @property
    def games(self) -> List[Game]:
        return self._games

    @games.setter
    def games(self, games: List[Game]) -> None:
        self._games = games

    # Generate the players and their weights
    def generate_players(self) -> None:
        flat = sorted({player for pl in self.game_list for player in pl})
        if "N/A" in flat:
            flat.remove("N/A")

        for name in flat:
            if name in self.weight_players:
                player_index = self.weight_players.index(name)
                weight = float(self.weight_weights[player_index])
                self.players[name] = Player(name, weight)
            else:
                self.players[name] = Player(name, 0.0)

        overlap_index = self.weight_players.index("Overlap")
        self.overlap = float(self.weight_weights[overlap_index])

    # Generate the games and their scores
    def generate_games(self) -> None:
        for g_idx, team in enumerate(self.game_list):
            date = datetime.datetime.strptime(self.game_dates[g_idx], "%m/%d/%y").date()

            if len(self.game_durations) > g_idx and self.game_durations[g_idx]:
                start, end = self.game_durations[g_idx].split(" - ")
                start = datetime.datetime.strptime(start, "%H:%M").time()
                end = datetime.datetime.strptime(end, "%H:%M").time()
                duration = datetime.datetime.combine(
                    datetime.date.min, end
                ) - datetime.datetime.combine(datetime.date.min, start)

                if duration.days == -1:
                    duration += datetime.timedelta(days=1)
            else:
                duration = None

            if len(self.game_scores) <= g_idx:
                continue

            score = int(self.game_scores[g_idx])

            # Parse anomaly flag from the new TRUE/FALSE column
            anomaly_raw = ""
            if len(self.game_anomalies) > g_idx:
                anomaly_raw = str(self.game_anomalies[g_idx]).strip().upper()
            is_anomaly = anomaly_raw in ("TRUE", "1", "YES", "Y")

            players = [self.players[name] for name in team if name != "N/A"]
            self.games.append(Game(date, duration, score, *players, is_anomaly=is_anomaly))

    # Save the weights of the players into the database
    def save_weights(self) -> None:
        range_name_l = Database.index_to_cell(
            constants.COLUMN_PADDING,
            constants.WEIGHTS_NAMES_COLUMN,
            None,
            None,
        )
        range_name_r = Database.index_to_cell(
            constants.COLUMN_PADDING + len(self.players),
            constants.WEIGHTS_WEIGHTS_COLUMN,
            None,
            None,
            )

        self.sheet.update(
            f"{range_name_l}:{range_name_r}",
            [[name, player.weight] for name, player in self.players.items()]
            + [["Overlap", self.overlap]],
            )
