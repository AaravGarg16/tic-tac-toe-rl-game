"""Serving-time inference.

Both agents are loaded once, at process start, and reused for every request, so
no part of a move handler touches the filesystem or rebuilds a network.

Move selection is deliberately *not* memoised.  A cache would save ~0.1 ms and
cost the agent its tie-breaking variety, so every game against a given opening
would play out identically.
"""

import json
import logging
import os
from pathlib import Path
from typing import Optional, Sequence

import torch

from rl_agents.board import Board
from rl_agents.dqn import QNetwork, select_move
from rl_agents.q_learning import QLearningAgent

from .engine import to_relative

log = logging.getLogger(__name__)

# The networks here are ~10k parameters. Spinning up a thread pool to multiply
# a 1x9 vector costs far more than it saves, and on a small container the
# contention actively hurts tail latency.
torch.set_num_threads(1)

ARTIFACTS = Path(__file__).resolve().parent.parent / "rl_agents" / "artifacts"
Q_TABLE_PATH = Path(os.getenv("Q_TABLE_PATH", ARTIFACTS / "q_table.json"))
DQN_PATH = Path(os.getenv("DQN_WEIGHTS", ARTIFACTS / "dqn.pt"))

# Both agents converge to near-optimal play, so an unbeatable "easy" mode is
# not much of a game.  Difficulty is therefore an explicit exploration rate at
# inference time: "easy" keeps the epsilon-greedy policy it was trained with, so
# it blunders often enough to be beatable, while "hard" plays greedily.
DIFFICULTY_EPSILON = {"easy": 0.35, "hard": 0.0}
DIFFICULTIES = tuple(DIFFICULTY_EPSILON)

_q_agent: Optional[QLearningAgent] = None
_dqn: Optional[QNetwork] = None


class AgentUnavailable(RuntimeError):
    """A trained artifact is missing or unreadable."""


def load_agents() -> None:
    """Load both artifacts into memory. Called once from the app's lifespan hook."""
    global _q_agent, _dqn

    if not Q_TABLE_PATH.exists():
        raise AgentUnavailable(f"missing Q-table at {Q_TABLE_PATH} -- run `python train.py`")
    if not DQN_PATH.exists():
        raise AgentUnavailable(f"missing DQN weights at {DQN_PATH} -- run `python train.py`")

    with Q_TABLE_PATH.open(encoding="utf-8") as fh:
        _q_agent = QLearningAgent.from_nested(json.load(fh))

    model = QNetwork()
    model.load_state_dict(torch.load(DQN_PATH, map_location="cpu"))
    model.eval()
    _dqn = model

    log.info("loaded agents: %s Q-values, DQN from %s", f"{len(_q_agent):,}", DQN_PATH.name)


def ready() -> bool:
    return _q_agent is not None and _dqn is not None


def agent_move(board: Sequence[str], mark: str, difficulty: str) -> int:
    """Pick a move for ``mark`` on ``board``. Assumes the board is already validated."""
    if difficulty not in DIFFICULTY_EPSILON:
        raise ValueError(f"difficulty must be one of {DIFFICULTIES}")
    if not ready():
        raise AgentUnavailable("agents are not loaded")

    view = Board(to_relative(board, mark))
    epsilon = DIFFICULTY_EPSILON[difficulty]
    if difficulty == "easy":
        move = _q_agent.select_move(view, epsilon)
    else:
        move = select_move(_dqn, view, epsilon)

    if move is None or board[move] != "":
        # Unreachable for a validated board, but a wrong move would corrupt the
        # game rather than fail loudly, so it is worth asserting.
        raise AgentUnavailable(f"{difficulty} agent returned an illegal move: {move}")
    return move

