"""Tabular Q-learning agent (easy mode).

    Q(s, a) <- Q(s, a) + alpha * [ r + gamma * max_a' Q(s', a') - Q(s, a) ]

States are keyed from the mover's perspective (mover is always +1), so one
table serves both sides of a self-play game.

Note s' is the board we face on our next turn, after the opponent replies,
not the one right after our own move. Bootstrapping from that would be valuing
a position the opponent is about to act on.
"""

import random
from typing import Dict, List, Optional, Tuple

from .board import Board

QKey = Tuple[str, int]


class QLearningAgent:
    def __init__(
        self,
        q: Optional[Dict[QKey, float]] = None,
        learning_rate: float = 0.2,
        gamma: float = 0.95,
    ):
        self.q: Dict[QKey, float] = q if q is not None else {}
        self.lr = learning_rate
        self.gamma = gamma

    # ---- inference -------------------------------------------------
    def value(self, state: str, action: int) -> float:
        return self.q.get((state, action), 0.0)

    def best_value(self, board: Board) -> float:
        moves = board.legal_moves()
        if not moves:
            return 0.0
        state = str(board)
        return max(self.value(state, m) for m in moves)

    def select_move(self, board: Board, epsilon: float = 0.0) -> Optional[int]:
        """Epsilon-greedy over the legal moves of a board seen as +1 to move."""
        moves = board.legal_moves()
        if not moves:
            return None
        if epsilon and random.random() < epsilon:
            return random.choice(moves)
        state = str(board)
        best = max(self.value(state, m) for m in moves)
        # Break ties at random so the agent is not biased toward low indices.
        return random.choice([m for m in moves if self.value(state, m) == best])

    # ---- learning --------------------------------------------------
    def update(self, state: str, action: int, reward: float, next_board: Optional[Board]) -> None:
        target = reward
        if next_board is not None:
            target += self.gamma * self.best_value(next_board)
        key = (state, action)
        self.q[key] = self.q.get(key, 0.0) + self.lr * (target - self.q.get(key, 0.0))

    # ---- persistence -----------------------------------------------
    def to_nested(self) -> Dict[str, Dict[str, float]]:
        nested: Dict[str, Dict[str, float]] = {}
        for (state, action), value in self.q.items():
            nested.setdefault(state, {})[str(action)] = value
        return nested

    @classmethod
    def from_nested(cls, nested: Dict[str, Dict[str, float]], **kwargs) -> "QLearningAgent":
        q = {
            (state, int(action)): value
            for state, actions in nested.items()
            for action, value in actions.items()
        }
        return cls(q=q, **kwargs)

    def __len__(self) -> int:
        return len(self.q)


def play_self_play_episode(agent: QLearningAgent, epsilon: float) -> int:
    """Play one self-play game, updating Q as it goes. Returns the winner (+1/-1/0).

    Both sides share the table; each turn the board is flipped so the mover is
    +1. A transition is only closed out once the same side is on move
    again, which is what makes s' the correct next state.
    """
    board = Board()
    player = 1
    # Per side: the (state, action) still awaiting its reward.
    pending: Dict[int, Tuple[str, int]] = {}

    while True:
        view = board if player == 1 else board.flipped()
        state = str(view)
        action = agent.select_move(view, epsilon)
        if action is None:
            break

        # This side is on move again, so close out its previous transition
        # with the position it now faces.
        if player in pending:
            prev_state, prev_action = pending.pop(player)
            agent.update(prev_state, prev_action, 0.0, view)

        pending[player] = (state, action)
        board = board.play(action, player)

        if board.is_over():
            winner = board.winner()
            for side, (s, a) in pending.items():
                if winner == 0:
                    reward = 0.0
                else:
                    reward = 1.0 if side == winner else -1.0
                agent.update(s, a, reward, None)
            return winner

        player = -player

    return board.winner()
