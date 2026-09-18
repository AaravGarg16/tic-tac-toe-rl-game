"""Deep Q-Network agent (the "hard" opponent).

A small MLP approximates Q(s, .) for all nine cells at once:

    Linear(9 -> 128) -> ReLU -> Linear(128 -> 64) -> ReLU -> Linear(64 -> 9)

Trained with the three ingredients that make DQN stable:

* **Experience replay** -- transitions are sampled uniformly from a buffer, so
  consecutive, highly-correlated moves do not dominate a gradient step.
* **Target network** -- the bootstrap term uses a periodically-synced copy of
  the weights, so the regression target does not move every step.
* **Bellman target** -- ``y = r + gamma * max_a' Q_target(s', a')`` for
  non-terminal transitions, ``y = r`` otherwise.

As with the tabular agent, every state is stored from the perspective of the
player to move, and ``s'`` is the position that same player faces on its next
turn (i.e. after the opponent replies).
"""

import random
from collections import deque
from typing import Deque, List, NamedTuple, Optional

import torch
import torch.nn as nn

from .board import Board

NEG_INF = float("-inf")


class QNetwork(nn.Module):
    def __init__(self, hidden1: int = 128, hidden2: int = 64):
        super().__init__()
        self.fc1 = nn.Linear(9, hidden1)
        self.fc2 = nn.Linear(hidden1, hidden2)
        self.fc3 = nn.Linear(hidden2, 9)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """``x``: float tensor of shape ``(batch, 9)`` -> Q-values ``(batch, 9)``."""
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


def encode(boards) -> torch.Tensor:
    """Encode one Board, or a sequence of them, as a ``(batch, 9)`` float tensor."""
    if isinstance(boards, Board):
        boards = [boards]
    return torch.tensor([b.cells for b in boards], dtype=torch.float32)


def legal_mask(boards) -> torch.Tensor:
    """Boolean ``(batch, 9)`` mask: ``True`` where a move is legal."""
    if isinstance(boards, Board):
        boards = [boards]
    return torch.tensor([[c == 0 for c in b.cells] for b in boards], dtype=torch.bool)


def masked_q(model: QNetwork, board: Board) -> torch.Tensor:
    """Q-values for one board with illegal cells driven to ``-inf``."""
    q = model(encode(board))
    return q.masked_fill(~legal_mask(board), NEG_INF)


@torch.inference_mode()
def select_move(model: QNetwork, board: Board, epsilon: float = 0.0) -> Optional[int]:
    moves = board.legal_moves()
    if not moves:
        return None
    if epsilon and random.random() < epsilon:
        return random.choice(moves)
    return int(masked_q(model, board).argmax(dim=1).item())


class Transition(NamedTuple):
    state: Board       # position, mover as +1
    action: int
    reward: float
    next_state: Optional[Board]  # same mover's next turn; None if the game ended


class ReplayBuffer:
    def __init__(self, capacity: int = 50_000):
        self.buffer: Deque[Transition] = deque(maxlen=capacity)

    def push(self, t: Transition) -> None:
        self.buffer.append(t)

    def sample(self, batch_size: int) -> List[Transition]:
        return random.sample(self.buffer, min(batch_size, len(self.buffer)))

    def __len__(self) -> int:
        return len(self.buffer)


def bellman_loss(
    online: QNetwork,
    target: QNetwork,
    batch: List[Transition],
    gamma: float,
    criterion: nn.Module,
) -> torch.Tensor:
    states = encode([t.state for t in batch])
    actions = torch.tensor([t.action for t in batch], dtype=torch.long)
    rewards = torch.tensor([t.reward for t in batch], dtype=torch.float32)

    predicted = online(states).gather(1, actions.unsqueeze(1)).squeeze(1)

    # Bootstrap only through the non-terminal transitions.
    bootstrap = torch.zeros(len(batch), dtype=torch.float32)
    live = [(i, t.next_state) for i, t in enumerate(batch) if t.next_state is not None]
    if live:
        idx = torch.tensor([i for i, _ in live], dtype=torch.long)
        nxt = [b for _, b in live]
        with torch.no_grad():
            q_next = target(encode(nxt)).masked_fill(~legal_mask(nxt), NEG_INF)
            bootstrap[idx] = q_next.max(dim=1).values

    return criterion(predicted, rewards + gamma * bootstrap)
