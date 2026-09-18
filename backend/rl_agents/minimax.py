"""Perfect play, used as a yardstick when evaluating the learned agents.

This is *not* the opponent you play against in the app -- solving tic-tac-toe
with search would make the RL agents pointless.  It exists so the win/draw
rates reported in the README are measured against a known-optimal baseline
rather than against a random mover, which flatters any agent.
"""

from functools import lru_cache
from typing import List, Tuple

from .board import Board


@lru_cache(maxsize=None)
def _solve(cells: Tuple[int, ...]) -> Tuple[int, Tuple[int, ...]]:
    """Return ``(value, best_actions)`` for the player to move (always ``+1``)."""
    board = Board(cells)
    winner = board.winner()
    if winner != 0:
        # The side that just moved was -1 from this node's perspective.
        return -1, ()
    moves = board.legal_moves()
    if not moves:
        return 0, ()

    best = -2
    best_actions: List[int] = []
    for move in moves:
        # Recurse from the opponent's perspective, then negate.
        child = board.play(move, 1).flipped()
        value = -_solve(tuple(child.cells))[0]
        if value > best:
            best, best_actions = value, [move]
        elif value == best:
            best_actions.append(move)
    return best, tuple(best_actions)


def optimal_value(board: Board) -> int:
    """``+1`` if the player to move wins with perfect play, ``0`` draw, ``-1`` loss."""
    return _solve(tuple(board.cells))[0]


def optimal_moves(board: Board) -> Tuple[int, ...]:
    """Every move that preserves the game-theoretic value for the player to move."""
    return _solve(tuple(board.cells))[1]
