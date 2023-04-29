# constants.py

"""This module defines project-level constants."""

from typing import Final
import typing

# Typing constants
Index = typing.Tuple[
    typing.Optional[int],
    typing.Optional[int],
    typing.Optional[int],
    typing.Optional[int]
]

# Spreadsheet constants
SPREADSHEET_NAME: Final[str] = 'Inventory - Board Games'
WORKSHEET_NAME: Final[str] = 'Quiz Match History'

GAME_DATES_COLUMN: Final[int] = 1
GAME_DURATIONS_COLUMN: Final[int] = 2
GAME_SCORES_COLUMN: Final[int] = 3
GAME_PLAYERS_COLUMN: Final[int] = 7
WEIGHTS_NAMES_COLUMN: Final[int] = 32
WEIGHTS_WEIGHTS_COLUMN: Final[int] = 33

PLAYER_PADDING: Final[int] = 2
COLUMN_PADDING: Final[int] = 4

SPREADSHEET_MAX_ROWS: Final[int] = 1000

GAME_MAX_PLAYERS: Final[int] = 9

CELL_PLAYER_GRID_L: Final[Index] = (
    PLAYER_PADDING, GAME_PLAYERS_COLUMN, None, None)
CELL_PLAYER_GRID_R: Final[Index] = (
    SPREADSHEET_MAX_ROWS, GAME_PLAYERS_COLUMN + GAME_MAX_PLAYERS, None, None)
CELL_GAME_DATES: Final[Index] = (
    None, GAME_DATES_COLUMN, None, PLAYER_PADDING)
CELL_GAME_DURATIONS: Final[Index] = (
    None, GAME_DURATIONS_COLUMN, None, PLAYER_PADDING)
CELL_GAME_SCORES: Final[Index] = (
    None, GAME_SCORES_COLUMN, None, PLAYER_PADDING)
CELL_WEIGHTS_NAME: Final[Index] = (
    None, WEIGHTS_NAMES_COLUMN, None, COLUMN_PADDING)
CELL_WEIGHTS_WEIGHT: Final[Index] = (
    None, WEIGHTS_WEIGHTS_COLUMN, None, COLUMN_PADDING)

# Game constants
GAME_MAX_SCORE: Final[int] = 66

# AI constants
TRAINING_SET_SIZE: Final[float] = 0.8
