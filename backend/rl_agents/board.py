"""Board representation shared by both agents.

Cells hold +1 (the agent to move, by convention), -1 (its opponent) or
0 (empty). Keeping the board sign-relative rather than X/O-relative means a
single network can play both sides: flip the signs and the position is simply
"mine to move" again.
"""

from typing import List, Optional, Sequence

LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),  # rows
    (0, 3, 6), (1, 4, 7), (2, 5, 8),  # columns
    (0, 4, 8), (2, 4, 6),             # diagonals
)


class Board:
    __slots__ = ("cells",)

    def __init__(self, cells: Optional[Sequence[int]] = None):
        self.cells = list(cells) if cells is not None else [0] * 9
        if len(self.cells) != 9:
            raise ValueError("a board has exactly 9 cells")

    def __str__(self) -> str:
        """Compact state key, e.g. "10-1000000". Used as the Q-table key."""
        return "".join(str(c) for c in self.cells)

    def __repr__(self) -> str:
        return f"Board({self.cells!r})"

    def legal_moves(self) -> List[int]:
        return [i for i, c in enumerate(self.cells) if c == 0]

    def is_legal(self, idx: int) -> bool:
        return 0 <= idx < 9 and self.cells[idx] == 0

    def play(self, idx: int, player: int) -> "Board":
        """Return a new board with player placed at idx."""
        if not self.is_legal(idx):
            raise ValueError(f"illegal move: {idx}")
        nxt = self.cells[:]
        nxt[idx] = player
        return Board(nxt)

    def is_full(self) -> bool:
        return 0 not in self.cells

    def winner(self) -> int:
        """+1/-1 for a won line, 0 for no winner yet."""
        c = self.cells
        for a, b, d in LINES:
            if c[a] != 0 and c[a] == c[b] == c[d]:
                return c[a]
        return 0

    def winning_line(self) -> Optional[tuple]:
        c = self.cells
        for line in LINES:
            a, b, d = line
            if c[a] != 0 and c[a] == c[b] == c[d]:
                return line
        return None

    def is_over(self) -> bool:
        return self.winner() != 0 or self.is_full()

    def flipped(self) -> "Board":
        """The same position seen from the other player's side."""
        return Board([-c for c in self.cells])
