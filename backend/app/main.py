"""FastAPI service exposing the trained agents.

The API is **stateless**: a request carries the whole position and the response
carries the agent's reply.  A tic-tac-toe board is nine bytes, so there is
nothing worth storing server-side, and no session state to lose when a
container restarts or to shard when it runs behind several workers.
"""

import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware

from . import policy
from .engine import InvalidBoard, validate, winner, winning_line
from .schemas import HealthResponse, MoveRequest, MoveResponse

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
log = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Load the artifacts during startup rather than on the first request, so a
    # cold container pays the cost before it starts accepting traffic.
    policy.load_agents()
    yield


app = FastAPI(
    title="Tic Tac Toe RL",
    version="1.0.0",
    summary="Play tic-tac-toe against a tabular Q-learning or Deep Q-Network agent.",
    lifespan=lifespan,
)

# Preflight responses are cached for a day; without this the browser pays an
# extra OPTIONS round trip every 10 minutes, doubling the latency of a move.
_origins = os.getenv("ALLOWED_ORIGINS", "*")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"] if _origins == "*" else [o.strip() for o in _origins.split(",")],
    allow_methods=["GET", "POST"],
    allow_headers=["Content-Type"],
    max_age=86_400,
)
app.add_middleware(GZipMiddleware, minimum_size=500)


@app.get("/health", response_model=HealthResponse, tags=["meta"])
def health() -> HealthResponse:
    """Liveness probe. The client also calls this on page load to warm the container."""
    loaded = policy.ready()
    return HealthResponse(status="ok" if loaded else "loading", agents_loaded=loaded)


@app.post("/move", response_model=MoveResponse, tags=["game"])
def move(req: MoveRequest) -> MoveResponse:
    """Validate the position, then answer with the agent's move."""
    board = list(req.board)
    try:
        validate(board, req.ai_mark)
    except InvalidBoard as exc:
        raise HTTPException(status_code=422, detail=str(exc)) from exc

    try:
        chosen = policy.agent_move(board, req.ai_mark, req.difficulty)
    except policy.AgentUnavailable as exc:
        log.exception("agent failure")
        raise HTTPException(status_code=503, detail=str(exc)) from exc

    board[chosen] = req.ai_mark
    result = winner(board)
    line = winning_line(board)
    return MoveResponse(
        move=chosen,
        board=board,
        status=result or "in_progress",
        winning_line=list(line) if line else None,
    )
