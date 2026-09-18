from fastapi.testclient import TestClient

from app.main import app

E = ""


def b(s: str):
    return [E if c == "." else c for c in s]


def client():
    # TestClient runs the lifespan hook, so the agents get loaded.
    return TestClient(app)


class TestHealth:
    def test_reports_loaded_agents(self):
        with client() as c:
            body = c.get("/health").json()
        assert body == {"status": "ok", "agents_loaded": True}


class TestMove:
    def test_answers_with_a_legal_move(self):
        with client() as c:
            r = c.post("/move", json={"board": b("X........"), "ai_mark": "O", "difficulty": "hard"})
        assert r.status_code == 200
        body = r.json()
        assert body["move"] in range(1, 9)
        assert body["board"][body["move"]] == "O"
        assert body["board"].count("O") == 1
        assert body["status"] == "in_progress"

    def test_reports_a_win_and_the_winning_line(self):
        #  O O .      O to move: 2 completes the top row and ends the game,
        #  X X .      even though X is threatening 5.
        #  X . .
        with client() as c:
            r = c.post("/move", json={"board": b("OO.XX.X.."), "ai_mark": "O", "difficulty": "hard"})
        body = r.json()
        assert body["move"] == 2
        assert body["status"] == "O"
        assert sorted(body["winning_line"]) == [0, 1, 2]

    def test_reports_a_draw(self):
        #  X O X      O to move into the last cell; the game ends level.
        #  X O O
        #  O X .
        with client() as c:
            r = c.post("/move", json={"board": b("XOXXOOOX."), "ai_mark": "X", "difficulty": "hard"})
        body = r.json()
        assert body["move"] == 8
        assert body["status"] == "draw"
        assert body["winning_line"] is None

    def test_rejects_a_position_that_is_not_the_agents_turn(self):
        with client() as c:
            r = c.post("/move", json={"board": b("........."), "ai_mark": "O", "difficulty": "hard"})
        assert r.status_code == 422
        assert "X's turn" in r.json()["detail"]

    def test_rejects_an_already_finished_game(self):
        with client() as c:
            r = c.post("/move", json={"board": b("XXXOO...."), "ai_mark": "O", "difficulty": "hard"})
        assert r.status_code == 422
        assert "already over" in r.json()["detail"]

    def test_rejects_a_tampered_board(self):
        with client() as c:
            r = c.post("/move", json={"board": b("XXXXX...."), "ai_mark": "O", "difficulty": "hard"})
        assert r.status_code == 422

    def test_rejects_bad_difficulty(self):
        with client() as c:
            r = c.post("/move", json={"board": b("X........"), "ai_mark": "O", "difficulty": "nightmare"})
        assert r.status_code == 422

    def test_rejects_wrong_board_length(self):
        with client() as c:
            r = c.post("/move", json={"board": [E] * 8, "ai_mark": "X", "difficulty": "hard"})
        assert r.status_code == 422


class TestCors:
    def test_preflight_is_cached_for_a_day(self):
        with client() as c:
            r = c.options(
                "/move",
                headers={
                    "Origin": "https://example.com",
                    "Access-Control-Request-Method": "POST",
                    "Access-Control-Request-Headers": "content-type",
                },
            )
        assert r.status_code == 200
        assert r.headers["access-control-max-age"] == "86400"
