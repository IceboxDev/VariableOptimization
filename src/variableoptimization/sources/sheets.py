"""Google Sheets data source.

Fetches the match history as whole rows and indexes columns *within each
row*, so blank cells can never shift values between rows — the Sheets API
only drops trailing cells of a row, never interior ones.
"""

import logging
from pathlib import Path

import gspread
from gspread.utils import rowcol_to_a1

from .. import constants
from ..snapshot import GameRecord, Snapshot

log = logging.getLogger(__name__)

GAMES_RANGE = (
    f"{rowcol_to_a1(constants.GAMES_FIRST_ROW, constants.COL_DATE)}"
    f":{rowcol_to_a1(constants.SHEET_MAX_ROW, constants.COL_PLAYERS_LAST)}"
)


def _cell(row: list, index: int) -> str:
    """Value of a 0-based cell within a fetched row; '' beyond the row's end."""
    return str(row[index]).strip() if index < len(row) else ""


class SheetsSource:
    """Reads and writes the live Google Sheet. The client is built lazily, so
    constructing the source (e.g. for a later write-back) costs nothing."""

    def __init__(
        self,
        credentials_path: str | Path,
        *,
        worksheet: gspread.Worksheet | None = None,
    ) -> None:
        self._credentials_path = Path(credentials_path)
        self._worksheet = worksheet
        if worksheet is None and not self._credentials_path.is_file():
            raise FileNotFoundError(
                f"Google service-account key not found: {self._credentials_path}"
            )

    @property
    def worksheet(self) -> gspread.Worksheet:
        if self._worksheet is None:
            client = gspread.service_account(filename=self._credentials_path)
            self._worksheet = client.open(constants.SPREADSHEET_NAME).worksheet(
                constants.WORKSHEET_NAME
            )
        return self._worksheet

    def fetch(self) -> Snapshot:
        [game_rows] = self.worksheet.batch_get([GAMES_RANGE])

        games = []
        for offset, row in enumerate(game_rows):
            # The games range starts at column A, so 0-based index == column - 1.
            date = _cell(row, constants.COL_DATE - 1)
            players = tuple(
                name
                for column in range(
                    constants.COL_PLAYERS_FIRST, constants.COL_PLAYERS_LAST + 1
                )
                if (name := _cell(row, column - 1))
            )
            if not date and not players:
                continue
            games.append(
                GameRecord(
                    row=constants.GAMES_FIRST_ROW + offset,
                    date=date,
                    duration=_cell(row, constants.COL_DURATION - 1),
                    score=_cell(row, constants.COL_SCORE - 1),
                    anomaly=_cell(row, constants.COL_ANOMALY - 1),
                    players=players,
                )
            )

        log.debug("Fetched %d game rows", len(games))
        return Snapshot(games=tuple(games))

    def save_anomalies(self, flags: dict[int, bool]) -> None:
        """Write the anomaly column for the given worksheet rows.

        ``flags`` maps 1-based worksheet rows to their anomaly state. Rows
        inside the written span that are missing from ``flags`` are set FALSE.
        """
        if not flags:
            return

        first_row, last_row = min(flags), max(flags)
        values = [
            ["TRUE" if flags.get(row, False) else "FALSE"]
            for row in range(first_row, last_row + 1)
        ]
        cell_range = (
            f"{rowcol_to_a1(first_row, constants.COL_ANOMALY)}"
            f":{rowcol_to_a1(last_row, constants.COL_ANOMALY)}"
        )
        self.worksheet.update(values=values, range_name=cell_range)
        log.info("Wrote anomaly flags to %s (%d rows)", cell_range, len(values))

    def save_rankings(self, rankings: list[tuple[str, float]]) -> None:
        """Write (player, strength) pairs into the ranking block.

        The full block is overwritten down to SHEET_MAX_ROW so stale entries
        from a previously larger roster are cleared.
        """
        values: list[list[str | float]] = [
            [name, round(value, 2)] for name, value in rankings
        ]
        block_rows = constants.SHEET_MAX_ROW - constants.RANKINGS_FIRST_ROW + 1
        values += [["", ""]] * (block_rows - len(values))

        cell_range = (
            f"{rowcol_to_a1(constants.RANKINGS_FIRST_ROW, constants.COL_RANKING_NAME)}"
            f":{rowcol_to_a1(constants.SHEET_MAX_ROW, constants.COL_RANKING_VALUE)}"
        )
        self.worksheet.update(values=values, range_name=cell_range)
        log.info("Wrote %d player rankings to %s", len(rankings), cell_range)
