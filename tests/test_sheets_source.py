"""Sheets source: row-aligned parsing of API responses and write-backs.

The fake worksheet emulates the Sheets API's exact quirk: trailing empty
cells of each row are dropped, interior blanks come back as ''.
"""

from variableoptimization import constants
from variableoptimization.sources import SheetsSource
from variableoptimization.sources.sheets import GAMES_RANGE, WEIGHTS_RANGE


def make_game_row(date="", duration="", score="", anomaly="", players=()):
    row = [""] * constants.COL_PLAYERS_LAST
    row[constants.COL_DATE - 1] = date
    row[constants.COL_DURATION - 1] = duration
    row[constants.COL_SCORE - 1] = score
    row[constants.COL_ANOMALY - 1] = anomaly
    for offset, player in enumerate(players):
        row[constants.COL_PLAYERS_FIRST - 1 + offset] = player
    while row and row[-1] == "":  # the API drops trailing empty cells
        row.pop()
    return row


class FakeWorksheet:
    def __init__(self, game_rows, weight_rows):
        self._payload = {GAMES_RANGE: game_rows, WEIGHTS_RANGE: weight_rows}
        self.updates = []

    def batch_get(self, ranges):
        return [self._payload[cell_range] for cell_range in ranges]

    def update(self, values, range_name):
        self.updates.append((range_name, values))


def make_source(game_rows, weight_rows=()):
    return SheetsSource(
        "unused.json", worksheet=FakeWorksheet(list(game_rows), list(weight_rows))
    )


def test_interior_blanks_stay_row_aligned():
    source = make_source(
        [
            make_game_row(date="01/05/24", score="40", players=("Alice", "Bob")),
            make_game_row(date="01/12/24", players=("Alice",)),  # no score
            make_game_row(date="01/19/24", score="33", players=("Bob",)),
        ]
    )
    snapshot = source.fetch()

    assert [record.score for record in snapshot.games] == ["40", "", "33"]
    assert [record.row for record in snapshot.games] == [2, 3, 4]


def test_ragged_weight_rows():
    source = make_source(
        [make_game_row(date="01/05/24", players=("Alice",))],
        weight_rows=[["Alice", "0.5"], ["Overlap"]],  # Overlap weight cell empty
    )
    snapshot = source.fetch()
    assert snapshot.weights[-1].name == "Overlap"
    assert snapshot.weights[-1].weight == ""


def test_fully_empty_rows_are_skipped():
    source = make_source(
        [
            make_game_row(date="01/05/24", players=("Alice",)),
            [],
            make_game_row(date="01/19/24", players=("Bob",)),
        ]
    )
    snapshot = source.fetch()
    # The blank row is skipped but row NUMBERS stay correct for write-backs.
    assert [record.row for record in snapshot.games] == [2, 4]


def test_save_anomalies_writes_contiguous_column():
    source = make_source([make_game_row(date="01/05/24", players=("Alice",))])
    worksheet = source.worksheet

    source.save_anomalies({2: True, 4: False})

    [(range_name, values)] = worksheet.updates
    assert range_name == "E2:E4"
    # Row 3 is inside the span but unknown -> explicitly FALSE, never blank.
    assert values == [["TRUE"], ["FALSE"], ["FALSE"]]


def test_save_anomalies_with_no_flags_is_a_no_op():
    source = make_source([make_game_row(date="01/05/24", players=("Alice",))])
    source.save_anomalies({})
    assert source.worksheet.updates == []
