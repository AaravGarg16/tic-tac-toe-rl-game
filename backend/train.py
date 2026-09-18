"""Train both agents from scratch and write their artifacts.

    python train.py                 # full run, ~3 min on a laptop CPU
    python train.py --quick         # short run, for smoke-testing the pipeline
    python train.py --agent dqn     # retrain just one of them

Artifacts land in ``rl_agents/artifacts/`` and are what the API serves.
"""

import argparse
import copy
import json
import os
import random
import time
from functools import lru_cache
from pathlib import Path
from typing import Dict, Optional, Tuple

import torch
import torch.nn as nn

from rl_agents.board import Board
from rl_agents.dqn import QNetwork, ReplayBuffer, Transition, bellman_loss, select_move
from rl_agents.dqn import encode, legal_mask
from rl_agents.minimax import optimal_moves, optimal_value
from rl_agents.q_learning import QLearningAgent, play_self_play_episode

# Same environment overrides the API honours, so a training run can be pointed
# at a scratch directory instead of the artifacts currently being served.
ARTIFACTS = Path(__file__).parent / "rl_agents" / "artifacts"
Q_TABLE_PATH = Path(os.getenv("Q_TABLE_PATH", ARTIFACTS / "q_table.json"))
DQN_PATH = Path(os.getenv("DQN_WEIGHTS", ARTIFACTS / "dqn.pt"))


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def _play_match(policy, opponent, agent_first: bool) -> int:
    """Play one game. ``policy``/``opponent`` map a +1-perspective Board to a move."""
    board = Board()
    agent_side = 1 if agent_first else -1
    player = 1
    while not board.is_over():
        view = board if player == 1 else board.flipped()
        move = policy(view) if player == agent_side else opponent(view)
        board = board.play(move, player)
        player = -player
    winner = board.winner()
    if winner == 0:
        return 0
    return 1 if winner == agent_side else -1


def _random_policy(board: Board) -> int:
    return random.choice(board.legal_moves())


def _minimax_policy(board: Board) -> int:
    return random.choice(optimal_moves(board))


def evaluate(policy, games: int = 500) -> Dict[str, float]:
    """Win/draw/loss vs a random mover, and vs perfect play, plus move accuracy."""
    results = {}
    for name, opponent in (("random", _random_policy), ("perfect", _minimax_policy)):
        w = d = l = 0
        for i in range(games):
            outcome = _play_match(policy, opponent, agent_first=(i % 2 == 0))
            if outcome > 0:
                w += 1
            elif outcome == 0:
                d += 1
            else:
                l += 1
        results[f"{name}_win"] = 100 * w / games
        results[f"{name}_draw"] = 100 * d / games
        results[f"{name}_loss"] = 100 * l / games

    # A much sharper signal than win rate against a random mover.
    results["optimal_move_pct"] = optimal_move_rate(policy)
    return results


@lru_cache(maxsize=1)
def _reachable_states():
    """Every position reachable in a legal game, from the mover's perspective."""
    seen = set()

    def walk(board: Board):
        key = tuple(board.cells)
        if key in seen:
            return
        seen.add(key)
        if board.is_over():
            return
        for move in board.legal_moves():
            walk(board.play(move, 1).flipped())

    walk(Board())
    return tuple(seen)


def play_report(policy) -> Tuple[int, int, int]:
    """Audit every game the agent can be drawn into, over all opponent lines.

    A greedy agent's reply to a position is forced, so the set of games it can
    ever play is small enough to enumerate exactly: branch on every legal
    opponent move and follow the agent's own choice. Returns the number of
    games it can be made to lose, the number of immediate wins it declines and
    the number of forced blocks it misses.

    These are the failures a player actually notices, and a high optimal-move
    rate over the whole position space does not imply any of them are zero.
    """
    losses = missed_wins = missed_blocks = 0

    def audit(view: Board, chosen: int) -> None:
        nonlocal missed_wins, missed_blocks
        wins = [m for m in view.legal_moves() if view.play(m, 1).winner() == 1]
        if wins:
            if chosen not in wins:
                missed_wins += 1
            return
        threats = [m for m in view.legal_moves() if view.play(m, -1).winner() == -1]
        if len(threats) == 1 and chosen != threats[0]:
            missed_blocks += 1

    def walk(board: Board, player: int, agent_side: int) -> None:
        nonlocal losses
        if board.is_over():
            won_by = board.winner()
            if won_by != 0 and won_by != agent_side:
                losses += 1
            return
        view = board if player == 1 else board.flipped()
        if player == agent_side:
            move = policy(view)
            audit(view, move)
            walk(board.play(move, player), -player, agent_side)
        else:
            for move in board.legal_moves():
                walk(board.play(move, player), -player, agent_side)

    for agent_side in (1, -1):
        walk(Board(), 1, agent_side)
    return losses, missed_wins, missed_blocks


def optimal_move_rate(policy) -> float:
    """Share of reachable positions where the agent plays a game-theoretically
    optimal move. Cheap enough to run during training as a validation metric."""
    agree = total = 0
    for cells in _reachable_states():
        board = Board(cells)
        if board.is_over():
            continue
        total += 1
        agree += policy(board) in optimal_moves(board)
    return 100 * agree / total


def _report(label: str, stats: Dict[str, float]) -> None:
    print(
        f"  {label:<12} vs random: {stats['random_win']:5.1f}% W "
        f"{stats['random_draw']:5.1f}% D {stats['random_loss']:5.1f}% L   |   "
        f"vs perfect: {stats['perfect_draw']:5.1f}% D {stats['perfect_loss']:5.1f}% L   |   "
        f"optimal moves: {stats['optimal_move_pct']:5.1f}%",
        flush=True,
    )


# ---------------------------------------------------------------------------
# Tabular Q-learning
# ---------------------------------------------------------------------------

def train_q_learning(episodes: int, seed: int) -> QLearningAgent:
    random.seed(seed)
    agent = QLearningAgent(learning_rate=0.2, gamma=0.95)
    print(f"Q-learning: {episodes:,} self-play episodes", flush=True)
    start = time.perf_counter()

    for episode in range(1, episodes + 1):
        # Anneal exploration over the first 70% of training, then hold at 0.05.
        epsilon = max(0.05, 1.0 - episode / (episodes * 0.7))
        play_self_play_episode(agent, epsilon)
        if episode % max(1, episodes // 4) == 0:
            print(f"  ep {episode:>7,}  |  {len(agent):>6,} state-action pairs", flush=True)

    print(f"  done in {time.perf_counter() - start:.1f}s", flush=True)
    return agent


# ---------------------------------------------------------------------------
# DQN
# ---------------------------------------------------------------------------

def train_dqn(
    episodes: int,
    seed: int,
    # 0.9 rather than the more usual 0.95: with a terminal reward of 1, a
    # shallower discount widens the gap between winning now and winning in two
    # more plies, past the network's approximation error. At 0.95 the agent
    # would occasionally pass up a win on the board for a slower one.
    gamma: float = 0.9,
    lr: float = 1e-3,
    batch_size: int = 128,
    buffer_size: int = 50_000,
    target_sync: int = 500,
    warmup: int = 1_000,
    eval_every: Optional[int] = None,
    exploring_starts: float = 0.5,
) -> QNetwork:
    random.seed(seed)
    torch.manual_seed(seed)

    # Validate ~20 times over the run, whatever its length.
    eval_every = eval_every or max(1, min(5_000, episodes // 20))

    # Self-play from the empty board only ever visits positions the current
    # policy walks into, so the network stays ignorant of the rest of the game
    # and misfires there (missing forced blocks, say). Starting a fraction of
    # episodes from a random legal position covers the whole state space.
    starts = [c for c in _reachable_states() if not Board(c).is_over()]

    online = QNetwork()
    target = QNetwork()
    target.load_state_dict(online.state_dict())
    target.eval()

    optimizer = torch.optim.Adam(online.parameters(), lr=lr)
    criterion = nn.SmoothL1Loss()
    buffer = ReplayBuffer(buffer_size)

    print(f"DQN: {episodes:,} self-play episodes (replay + target network)", flush=True)
    start = time.perf_counter()
    steps = 0

    # The last episode is not necessarily the best network: exploration keeps
    # perturbing the weights long after the policy has converged. Validate
    # periodically against perfect play and keep the strongest snapshot.
    B = 10 ** 9
    best_score = (-B, -B, -B, -1.0)
    best_weights = None

    def snapshot_if_best():
        """Rank snapshots by games-lost first, optimal-move rate second.

        Optimising the move rate alone is not enough: a snapshot can play more
        textbook-optimal moves overall and still walk into a losing line or
        decline a win on the board, which are the failures a player notices.
        """
        nonlocal best_score, best_weights
        online.eval()
        policy_fn = lambda b: select_move(online, b)  # noqa: E731
        losses, missed_wins, missed_blocks = play_report(policy_fn)
        score = (-losses, -missed_wins, -missed_blocks, optimal_move_rate(policy_fn))
        online.train()
        if score > best_score:
            best_score = score
            best_weights = copy.deepcopy(online.state_dict())
        return score

    for episode in range(1, episodes + 1):
        epsilon = max(0.05, 1.0 - episode / (episodes * 0.7))
        seed_position = random.choice(starts) if random.random() < exploring_starts else None
        _collect_episode(online, buffer, epsilon, seed_position)

        if len(buffer) >= warmup:
            batch = buffer.sample(batch_size)
            loss = bellman_loss(online, target, batch, gamma, criterion)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(online.parameters(), 1.0)
            optimizer.step()
            steps += 1
            if steps % target_sync == 0:
                target.load_state_dict(online.state_dict())

        if episode % eval_every == 0:
            score = snapshot_if_best()
            if episode % max(eval_every, episodes // 6) == 0:
                print(
                    f"  ep {episode:>7,}  |  {steps:>6,} updates  |  "
                    f"losable {-score[0]:>3}  missed wins {-score[1]:>3}  "
                    f"missed blocks {-score[2]:>3}  optimal {score[3]:5.1f}%",
                    flush=True,
                )

    snapshot_if_best()  # the final network is a candidate too
    online.load_state_dict(best_weights)
    online.eval()
    print(f"  done in {time.perf_counter() - start:.1f}s, kept the best snapshot "
          f"({-best_score[0]} losable games, {-best_score[1]} missed wins, "
          f"{-best_score[2]} missed blocks, {best_score[3]:.1f}% optimal moves)", flush=True)
    return online


def _collect_episode(
    model: QNetwork,
    buffer: ReplayBuffer,
    epsilon: float,
    seed_position: Optional[tuple] = None,
) -> None:
    """Play one self-play game and push its transitions into the replay buffer.

    ``seed_position`` optionally starts the game from a mid-game position
    (mover as +1) rather than from the empty board.
    """
    board = Board(seed_position) if seed_position is not None else Board()
    player = 1
    pending: Dict[int, Tuple[Board, int]] = {}

    while True:
        view = board if player == 1 else board.flipped()
        action = select_move(model, view, epsilon)
        if action is None:
            break

        # Now that this side is on move again, its previous transition has a
        # well-defined next state: the board it is looking at right now.
        if player in pending:
            prev_state, prev_action = pending.pop(player)
            buffer.push(Transition(prev_state, prev_action, 0.0, view))

        pending[player] = (view, action)
        board = board.play(action, player)

        if board.is_over():
            winner = board.winner()
            for side, (state, act) in pending.items():
                reward = 0.0 if winner == 0 else (1.0 if side == winner else -1.0)
                buffer.push(Transition(state, act, reward, None))
            return

        player = -player


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--agent", choices=("q", "dqn", "both"), default="both")
    parser.add_argument("--q-episodes", type=int, default=200_000)
    parser.add_argument("--dqn-episodes", type=int, default=400_000)
    parser.add_argument("--eval-games", type=int, default=500)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--quick",
        action="store_true",
        help="tiny run that exercises the pipeline without overwriting the artifacts",
    )
    args = parser.parse_args()

    # A smoke test must not clobber the trained artifacts the API serves, which
    # took a couple of minutes to produce and are committed to the repo.
    save = not args.quick
    if args.quick:
        args.q_episodes, args.dqn_episodes, args.eval_games = 2_000, 2_000, 50
        print("quick run: artifacts will not be written\n", flush=True)

    if save:
        Q_TABLE_PATH.parent.mkdir(parents=True, exist_ok=True)
        DQN_PATH.parent.mkdir(parents=True, exist_ok=True)

    if args.agent in ("q", "both"):
        agent = train_q_learning(args.q_episodes, args.seed)
        if save:
            Q_TABLE_PATH.write_text(json.dumps(agent.to_nested(), separators=(",", ":")))
            size_kb = Q_TABLE_PATH.stat().st_size / 1024
            print(f"  saved {Q_TABLE_PATH.name} ({len(agent):,} pairs, {size_kb:.0f} KB)", flush=True)
        _report("q-learning", evaluate(lambda b: agent.select_move(b, 0.0), args.eval_games))

    if args.agent in ("dqn", "both"):
        model = train_dqn(args.dqn_episodes, args.seed)
        if save:
            torch.save(model.state_dict(), DQN_PATH)
            size_kb = DQN_PATH.stat().st_size / 1024
            print(f"  saved {DQN_PATH.name} ({size_kb:.0f} KB)", flush=True)
        _report("dqn", evaluate(lambda b: select_move(model, b, 0.0), args.eval_games))


if __name__ == "__main__":
    main()
