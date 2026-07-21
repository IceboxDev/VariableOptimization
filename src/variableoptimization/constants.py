"""Project-wide constants: data-source locations, worksheet layout, game rules,
and training configuration.

When the spreadsheet structure changes, update the layout section here — it is
the single source of truth for both the Google Sheets and the xlsx reader.
"""

import datetime
from typing import Final

# --- Data sources -----------------------------------------------------------
SPREADSHEET_NAME: Final[str] = "Inventory - Board Games"
WORKSHEET_NAME: Final[str] = "Quiz Match History"
XLSX_DEFAULT_PATH: Final[str] = "Inventory - Board Games.xlsx"

CREDENTIALS_ENV_VAR: Final[str] = "GOOGLE_APPLICATION_CREDENTIALS"
CREDENTIALS_GLOB: Final[str] = ".config/*.json"

CACHE_DEFAULT_PATH: Final[str] = ".cache/snapshot.json"
CACHE_TTL: Final[datetime.timedelta] = datetime.timedelta(hours=3)

# --- Worksheet layout (1-based rows and columns) ----------------------------
GAMES_FIRST_ROW: Final[int] = 2

COL_DATE: Final[int] = 1            # A
COL_DURATION: Final[int] = 2        # B
COL_SCORE: Final[int] = 3           # C
COL_ANOMALY: Final[int] = 5         # E
COL_PLAYERS_FIRST: Final[int] = 9   # I
COL_PLAYERS_LAST: Final[int] = 22   # V

WEIGHTS_FIRST_ROW: Final[int] = 3
COL_WEIGHT_NAME: Final[int] = 57    # BE
COL_WEIGHT_VALUE: Final[int] = 58   # BF

SHEET_MAX_ROW: Final[int] = 1000

# --- Cell value conventions -------------------------------------------------
DATE_FORMAT: Final[str] = "%m/%d/%y"
TIME_FORMAT: Final[str] = "%H:%M"
DURATION_SEPARATOR: Final[str] = " - "
NO_PLAYER: Final[str] = "N/A"
OVERLAP_KEY: Final[str] = "Overlap"
TRUE_VALUES: Final[frozenset[str]] = frozenset({"TRUE", "1", "YES", "Y"})

# --- Game rules -------------------------------------------------------------
GAME_MAX_SCORE: Final[int] = 66
TEAM_SIZE: Final[int] = 5

# Refuse player evaluations that would enumerate more teams than this —
# C(unfiltered roster, TEAM_SIZE) runs into the billions.
EVAL_MAX_TEAMS: Final[int] = 10_000_000
EVAL_BATCH_SIZE: Final[int] = 16_384

# --- Training ---------------------------------------------------------------
TRAINING_SET_SIZE: Final[float] = 0.8
MODELS_DIR: Final[str] = "models"
LEGACY_MODELS_DIR: Final[str] = "NeuralNetwork"
