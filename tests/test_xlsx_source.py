"""The xlsx source must produce sheet-convention strings, row-aligned."""

import pytest

from variableoptimization.sources import XlsxSource


def test_missing_file_raises():
    with pytest.raises(FileNotFoundError):
        XlsxSource("does-not-exist.xlsx")


def test_missing_worksheet_raises(fixture_xlsx):
    with pytest.raises(ValueError, match="Bogus"):
        XlsxSource(fixture_xlsx, worksheet_name="Bogus").fetch()


def test_games_are_row_aligned(fixture_snapshot):
    games = fixture_snapshot.games
    assert len(games) == 6
    assert [record.row for record in games] == [2, 3, 4, 5, 6, 7]

    # Dates are canonicalised to the sheet's MM/DD/YY convention.
    assert games[0].date == "01/05/24"

    # The mid-column blank score stays attached to ITS row — rows below keep
    # their own values (this is the misalignment bug class from the old code).
    assert games[1].score == ""
    assert games[2].score == "-1"
    assert games[3].score == "55"
    assert games[4].score == "33"


def test_values_are_canonical_strings(fixture_snapshot):
    games = fixture_snapshot.games
    assert games[0].duration == "18:30 - 20:15"
    assert games[3].anomaly == "TRUE"
    assert games[0].anomaly == "FALSE"
    assert games[5].anomaly == ""
    assert games[3].players == ("Alice", "Bob", "N/A")  # raw, unfiltered


def test_weights_include_empty_overlap(fixture_snapshot):
    weights = {record.name: record.weight for record in fixture_snapshot.weights}
    assert weights == {"Alice": "0.5", "Bob": "0.25", "Overlap": ""}
