"""Game rules for the API layer.

The API is stateless: the client sends a position, the server validates it and
answers with one move.  That means these checks are the only thing standing
between a hand-crafted request and a nonsense board, so they are strict.
"""

from typing import List, Optional, Sequence

EMPTY = ""
MARKS = ("X", "O")
FIRST_MOVER = "X"  # X always opens, whichever mark the human chose

LINES = (
    (0, 1, 2), (3, 4, 5), (6, 7, 8),
    (0, 3, 6), (1, 4, 7), (2, 5, 8),
    (0, 4, 8), (2, 4, 6),
)


class InvalidBoard(ValueError):
    """The submitted position could not have arisen from a legal game."""


def other(mark: str) -> str:
    return "O" if mark == "X" else "X"


def legal_moves(board: Sequence[str]) -> List[int]:
    return [i for i, c in enumerate(board) if c == EMPTY]


def winner(board: Sequence[str]) -> Optional[str]:
    """``"X"``, ``"O"``, ``"draw"``, or ``None`` if the game is still running."""
    for a, b, c in LINES:
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return board[a]
    return "draw" if EMPTY not in board else None


def winning_line(board: Sequence[str]) -> Optional[tuple]:
    for line in LINES:
        a, b, c = line
        if board[a] != EMPTY and board[a] == board[b] == board[c]:
            return line
    return None


def validate(board: Sequence[str], to_move: str) -> None:
    """Raise ``InvalidBoard`` unless it really is ``to_move``'s turn here."""
    if len(board) != 9:
        raise InvalidBoard("board must have exactly 9 cells")
    if any(c not in (EMPTY, "X", "O") for c in board):
        raise InvalidBoard("cells must be '', 'X' or 'O'")
    if to_move not in MARKS:
        raise InvalidBoard("mark must be 'X' or 'O'")

    first = board.count(FIRST_MOVER)
    second = board.count(other(FIRST_MOVER))
    if first not in (second, second + 1):
        raise InvalidBoard("move counts are impossible: X moves first and players alternate")

    expected = FIRST_MOVER if first == second else other(FIRST_MOVER)
    if to_move != expected:
        raise InvalidBoard(f"it is {expected}'s turn, not {to_move}'s")

    if winner(board) is not None:
        raise InvalidBoard("the game is already over")


def to_relative(board: Sequence[str], mark: str) -> List[int]:
    """Encode the board as ``+1`` for ``mark``, ``-1`` for the opponent, ``0`` empty.

    This is the representation both agents were trained on, so the network and
    the Q-table always see a position as "mine to move".
    """
    opponent = other(mark)
    return [1 if c == mark else -1 if c == opponent else 0 for c in board]
