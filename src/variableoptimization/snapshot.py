"""Snapshot: the transport-level contract every data source produces.

A Snapshot holds raw *strings* in worksheet conventions (dates as
``MM/DD/YY``, anomalies as ``TRUE``/``FALSE``), row-aligned per record so a
blank cell can never shift values between rows. Parsing into domain objects
happens once, in :mod:`variableoptimization.database`.
"""

import dataclasses
import hashlib
import json
from typing import Any, Self

from . import constants

SCHEMA_VERSION = 2


@dataclasses.dataclass(frozen=True, slots=True)
class GameRecord:
    """One worksheet row of the match history, unparsed."""

    row: int  # 1-based worksheet row, kept for write-backs
    date: str
    duration: str
    score: str
    anomaly: str
    players: tuple[str, ...]


@dataclasses.dataclass(frozen=True, slots=True)
class WeightRecord:
    """One (name, weight) row of the weights block, unparsed."""

    name: str
    weight: str


@dataclasses.dataclass(frozen=True, slots=True)
class Snapshot:
    games: tuple[GameRecord, ...]
    weights: tuple[WeightRecord, ...]

    def to_dict(self) -> dict[str, Any]:
        return {
            "games": [dataclasses.asdict(record) for record in self.games],
            "weights": [dataclasses.asdict(record) for record in self.weights],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> Self:
        return cls(
            games=tuple(
                GameRecord(
                    row=int(record["row"]),
                    date=str(record["date"]),
                    duration=str(record["duration"]),
                    score=str(record["score"]),
                    anomaly=str(record["anomaly"]),
                    players=tuple(str(name) for name in record["players"]),
                )
                for record in data["games"]
            ),
            weights=tuple(
                WeightRecord(name=str(record["name"]), weight=str(record["weight"]))
                for record in data["weights"]
            ),
        )


def fingerprint(snapshot: Snapshot) -> str:
    """Content hash (16 hex chars) over what training actually sees.

    Covers the sorted roster, the scored game records, and the weights —
    deliberately excluding unscored rows, so upcoming games don't change the
    fingerprint and falsely reset the promotion baseline.
    """
    def is_scored(value: str) -> bool:
        try:
            return float(value) >= 0
        except ValueError:
            return False

    roster = sorted({
        name
        for record in snapshot.games
        for name in record.players
        if name != constants.NO_PLAYER
    })
    scored = [
        dataclasses.asdict(record)
        for record in snapshot.games
        if record.score and is_scored(record.score)
    ]
    weights = [dataclasses.asdict(record) for record in snapshot.weights]
    payload = json.dumps(
        {"roster": roster, "scored": scored, "weights": weights},
        sort_keys=True,
        separators=(",", ":"),
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]
