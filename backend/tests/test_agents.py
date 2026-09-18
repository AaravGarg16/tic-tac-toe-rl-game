"""Behavioural tests for the trained agents.

These assert on *play strength*, not on specific moves. The agents are
retrainable and break ties at random, so pinning individual cells would make
the suite fragile without testing anything meaningful.

Most assertions range over the positions the hard agent can actually reach in
a game. That distinction matters: because it plays greedily its reply to any
position is forced, so the set of games it can ever play is small and fully
enumerable, while the wider space of legal positions contains many it will
never face. A guarantee over the whole space would be stronger, but a guarantee
over what it actually plays is the one a player experiences.
"""

import pytest

from app import policy
from app.engine import legal_moves, other, winner
from rl_agents.board import Board
from rl_agents.minimax import optimal_value

E = ""


def b(s: str):
    """Build a board from a 9-char string, '.' for empty."""
    return [E if c == "." else c for c in s]


@pytest.fixture(scope="module", autouse=True)
def agents():
    policy.load_agents()


def hard_move(board, mark="O"):
    return policy.agent_move(board, mark, "hard")


def wins_with(board, index, mark):
    probe = list(board)
    probe[index] = mark
    return winner(probe) == mark


def winning_moves(board, mark):
    return [i for i in legal_moves(board) if wins_with(board, index=i, mark=mark)]


def _walk_games(agent_mark, visit):
    """Enumerate every game the hard agent can play, calling `visit(board)` on
    each position it is asked to move in. Branches on all opponent replies."""
    human = other(agent_mark)

    def rec(board, to_move):
        if winner(board) is not None:
            return
        if to_move == agent_mark:
            visit(board)
            nxt = list(board)
            nxt[hard_move(board, agent_mark)] = agent_mark
            rec(nxt, human)
        else:
            for move in legal_moves(board):
                nxt = list(board)
                nxt[move] = human
                rec(nxt, agent_mark)

    rec([E] * 9, "X")


@pytest.fixture(scope="module")
def reachable():
    """Positions the hard agent is actually asked to move in, per mark."""
    found = {}
    for mark in ("X", "O"):
        seen = []
        _walk_games(mark, lambda board: seen.append(list(board)))
        found[mark] = seen
    return found


class TestHardAgentIsUnbeatable:
    @pytest.mark.parametrize("agent_mark", ["X", "O"])
    def test_no_line_of_play_beats_it(self, agent_mark):
        human = other(agent_mark)
        outcomes = {"win": 0, "draw": 0, "loss": 0}

        def rec(board, to_move):
            result = winner(board)
            if result is not None:
                key = "draw" if result == "draw" else ("win" if result == agent_mark else "loss")
                outcomes[key] += 1
                return
            if to_move == agent_mark:
                nxt = list(board)
                nxt[hard_move(board, agent_mark)] = agent_mark
                rec(nxt, human)
            else:
                for move in legal_moves(board):
                    nxt = list(board)
                    nxt[move] = human
                    rec(nxt, agent_mark)

        rec([E] * 9, "X")
        total = sum(outcomes.values())
        # A sanity guard on the walk itself, not a target: a stronger agent ends
        # games sooner, which prunes the tree and legitimately lowers this count.
        assert total > 20, "the search should cover a meaningful number of games"
        assert outcomes["loss"] == 0, (
            f"as {agent_mark}, the agent loses {outcomes['loss']} of {total} possible games"
        )


class TestHardAgentTactics:
    @pytest.mark.parametrize("agent_mark", ["X", "O"])
    def test_takes_every_win_it_is_offered(self, agent_mark, reachable):
        missed = [
            board
            for board in reachable[agent_mark]
            if winning_moves(board, agent_mark)
            and hard_move(board, agent_mark) not in winning_moves(board, agent_mark)
        ]
        assert not missed, f"missed {len(missed)} immediate wins, e.g. {missed[0]}"

    @pytest.mark.parametrize("agent_mark", ["X", "O"])
    def test_blocks_every_forced_threat(self, agent_mark, reachable):
        human = other(agent_mark)
        missed = []
        for board in reachable[agent_mark]:
            if winning_moves(board, agent_mark):
                continue  # winning outright beats blocking
            threats = winning_moves(board, human)
            if len(threats) == 1 and hard_move(board, agent_mark) != threats[0]:
                missed.append((board, threats[0]))
        assert not missed, f"missed {len(missed)} forced blocks, e.g. {missed[0]}"

    @pytest.mark.parametrize("agent_mark", ["X", "O"])
    def test_never_moves_into_a_lost_position(self, agent_mark, reachable):
        """Every move it makes should preserve at least a draw."""
        blunders = []
        for board in reachable[agent_mark]:
            rel = [1 if c == agent_mark else -1 if c else 0 for c in board]
            after = Board(rel).play(hard_move(board, agent_mark), 1).flipped()
            if optimal_value(after) > 0:  # opponent now wins with perfect play
                blunders.append(board)
        assert not blunders, f"{len(blunders)} moves hand the opponent a win, e.g. {blunders[0]}"

    def test_move_is_always_legal(self):
        board = b("XOX.O....")
        assert hard_move(board, "X") in legal_moves(board)


class TestEasyAgent:
    def test_is_beatable_but_not_random(self):
        """Easy explores on purpose, so it should be clearly weaker than hard."""
        board = b("XX.O.....")  # a forced block
        blocks = sum(policy.agent_move(board, "O", "easy") == 2 for _ in range(200))
        assert 100 < blocks < 195, f"blocked {blocks}/200 times"

    def test_always_plays_a_legal_move(self):
        board = b("XOX.O....")
        for _ in range(200):
            assert policy.agent_move(board, "X", "easy") in legal_moves(board)


class TestPolicyGuards:
    def test_rejects_unknown_difficulty(self):
        with pytest.raises(ValueError, match="difficulty must be"):
            policy.agent_move(b("........."), "X", "impossible")
