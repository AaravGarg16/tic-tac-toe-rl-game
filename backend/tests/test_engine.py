import pytest

from app.engine import InvalidBoard, legal_moves, to_relative, validate, winner, winning_line

E = ""


def b(s: str):
    """Build a board from a 9-char string, '.' for empty."""
    return [E if c == "." else c for c in s]


class TestWinner:
    @pytest.mark.parametrize(
        "board,expected",
        [
            ("XXX.O.O..", "X"),
            ("...XXX.OO", "X"),
            ("OO.OO.XXX", "X"),
            ("X..X..X.O", "X"),
            (".O..O..OX", "O"),
            ("X...X...X", "X"),
            ("..O.O.O..", "O"),
            (".........", None),
            ("XX.OO....", None),
            ("XOXXOOOXX", "draw"),
        ],
    )
    def test_detects_result(self, board, expected):
        assert winner(b(board)) == expected

    def test_winning_line_matches_winner(self):
        assert winning_line(b("XXX.O.O..")) == (0, 1, 2)
        assert winning_line(b("XOXXOOOXX")) is None


class TestValidate:
    def test_accepts_empty_board_with_x_to_move(self):
        validate(b("........."), "X")

    def test_rejects_wrong_turn(self):
        with pytest.raises(InvalidBoard, match="X's turn"):
            validate(b("........."), "O")

    def test_rejects_impossible_move_counts(self):
        with pytest.raises(InvalidBoard, match="impossible"):
            validate(b("XX......."), "O")
        with pytest.raises(InvalidBoard, match="impossible"):
            validate(b("OO......."), "X")

    def test_rejects_finished_game(self):
        with pytest.raises(InvalidBoard, match="already over"):
            validate(b("XXXOO...."), "O")

    def test_rejects_bad_cells(self):
        with pytest.raises(InvalidBoard, match="cells must be"):
            validate(["Z"] + [E] * 8, "O")

    def test_rejects_wrong_length(self):
        with pytest.raises(InvalidBoard, match="9 cells"):
            validate([E] * 8, "X")


class TestEncoding:
    def test_agent_is_always_plus_one(self):
        assert to_relative(b("XO......."), "X") == [1, -1, 0, 0, 0, 0, 0, 0, 0]
        assert to_relative(b("XO......."), "O") == [-1, 1, 0, 0, 0, 0, 0, 0, 0]

    def test_legal_moves(self):
        assert legal_moves(b("XO.......")) == [2, 3, 4, 5, 6, 7, 8]
        assert legal_moves(b("XOXOXOXOX")) == []
