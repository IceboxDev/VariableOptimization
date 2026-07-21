"""Domain model: players and games, independent of any data source.

Constructing a Game does NOT register it anywhere — hypothetical games built
for inference leave player histories untouched. Only Database registers real
games into ``Player.games``.
"""

import dataclasses
import datetime


@dataclasses.dataclass(eq=False)
class Player:
    """A pub-quiz participant. Identity is the name."""

    name: str
    games: set["Game"] = dataclasses.field(default_factory=set, repr=False)

    def __eq__(self, other: object) -> bool:
        return isinstance(other, Player) and self.name == other.name

    def __hash__(self) -> int:
        return hash(self.name)

    def __repr__(self) -> str:
        return self.name

    @property
    def scored_games(self) -> list["Game"]:
        return [game for game in self.games if game.has_score()]


@dataclasses.dataclass(frozen=True)
class Game:
    """One pub-quiz session. ``score is None`` means not (yet) scored."""

    date: datetime.date | None
    duration: datetime.timedelta | None
    score: int | None
    players: tuple[Player, ...]
    is_anomaly: bool = False
    # Worksheet row this game came from; used for write-backs. Not identity.
    sheet_row: int | None = dataclasses.field(default=None, compare=False)

    def has_score(self) -> bool:
        return self.score is not None

    def player_count(self) -> int:
        return len(self.players)
