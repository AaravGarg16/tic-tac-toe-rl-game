# Tic Tac Toe RL

Play tic-tac-toe against two reinforcement-learning agents, a tabular
Q-learning agent and a Deep Q-Network, both trained from scratch by self-play
with no game-tree search and no hand-written strategy.

**React · FastAPI · PyTorch**

[![CI](https://github.com/AaravGarg16/tic-tac-toe-rl-game/actions/workflows/ci.yml/badge.svg)](https://github.com/AaravGarg16/tic-tac-toe-rl-game/actions/workflows/ci.yml)

<img width="1225" alt="The game board mid-match" src="https://github.com/user-attachments/assets/a4b1eb58-c684-4d52-98de-50d782b53275" />

---

## The agents

Neither agent is told the rules of good play. Both start from nothing and learn
by playing millions of moves against themselves.

| Mode | Algorithm | vs random (W/D/L) | vs perfect play | Optimal-move rate |
|------|-----------|-------------------|-----------------|-------------------|
| **Easy** | Tabular Q-learning, ε = 0.35 at play time | 76.1 / 9.8 / 14.1 % | draws 56.4 %, loses 43.6 % | 84.8 % |
| **Hard** | Deep Q-Network, greedy | 93.9 / 6.1 / **0.0** % | **draws 100 %, never loses** | 99.7 % |

Measured over 2,000 games per opponent, alternating who opens, plus a sweep of
every legal position. Reproduce with `python train.py`.

**The hard agent cannot be beaten.** It plays greedily, so its reply to any
position is forced, which makes the set of games it can ever play small enough
to enumerate exactly: branch on every legal opponent move, follow the agent's
own choice, and check every leaf. Across all 562 distinct games reachable
against *any* opponent play it never loses, never declines a win on the board,
and never misses a forced block. That search runs as a
[test](backend/tests/test_agents.py), so the guarantee is enforced in CI.

**Optimal-move rate** is the share of reachable positions where the agent picks
a move a perfect player would. It is a far sharper measure than win rate against
a random opponent, which flatters almost anything. The
[minimax solver](backend/rl_agents/minimax.py) that provides this baseline is
used *only* for evaluation. Solving the game with search would defeat the point.

Both agents converge to near-optimal play, so difficulty is an explicit
exploration rate rather than a weaker algorithm. **Easy** keeps the ε-greedy
policy it trained with, so it blunders often enough to be beatable. **Hard**
plays greedily, and the best result available against it is a draw.

---

## How it works

```
  Browser                                   FastAPI
┌───────────────────────┐              ┌──────────────────────────┐
│  React                │              │  POST /move              │
│  ├ draws your mark    │ ──board──▶   │   ├ validate position    │
│  │  immediately       │              │   ├ encode to ±1         │
│  └ settles wins and   │  ◀──move──   │   └ Q-table  or  DQN     │
│    draws locally      │              │      (loaded at startup) │
└───────────────────────┘              └──────────────────────────┘
```

The API is **stateless**: each request carries the whole position and the
response carries one move. A board is nine bytes, so there is no session worth
keeping server-side, and nothing to lose when a container restarts.

Positions are always encoded from the perspective of the player to move (`+1`
is "me", `-1` is the opponent, `0` empty). One table and one network therefore
play both sides: flip the signs and it is your turn again.

### Q-learning (Easy)

A dictionary from `(state, action)` to an expected return, updated with the
Bellman equation:

```
Q(s, a) ← Q(s, a) + α · [ r + γ · max_a' Q(s', a') − Q(s, a) ]
```

`s'` is the position the agent faces on its *next* turn, after the opponent
replies rather than straight after its own move. Bootstrapping from the intermediate
position would value a board the opponent is about to act on.

Final table: 16,162 state-action pairs, α = 0.2, γ = 0.95, 200k self-play
episodes with ε annealed 1.0 → 0.05. It also draws 100 % of games against
perfect play, which is why difficulty is an exploration rate rather than a
weaker algorithm.

### Deep Q-Network (Hard)

`Linear(9→128) → ReLU → Linear(128→64) → ReLU → Linear(64→9)`, about 10k
parameters, predicting all nine Q-values at once and masking illegal cells to
`−∞` before the argmax.

Trained with the three things that make DQN stable:

- **Experience replay.** Transitions are sampled uniformly from a 50k buffer,
  so consecutive, highly-correlated moves cannot dominate a gradient step.
- **Target network.** The bootstrap term uses a copy of the weights synced
  every 500 updates, so the regression target stops chasing its own tail.
- **Bellman target.** `y = r + γ · max_a' Q_target(s', a')`, or `y = r` at a
  terminal state. Huber loss, Adam at 1e-3, gradient-norm clipping at 1.0.

Two further choices carry more weight here than any of the hyperparameters:

- **Exploring starts.** Self-play from the empty board only visits positions
  the current policy walks into, which leaves the rest of the state space
  unseen. Half of all episodes therefore begin from a random legal position, so
  the network is trained across the whole game rather than the narrow band its
  own play reaches.
- **Checkpoint selection on play quality.** The final episode is not
  necessarily the strongest network, and optimal-move rate alone is a poor
  selection criterion: a snapshot can score higher on it while still walking
  into a losing line. Training validates every 5k episodes and keeps the best
  snapshot, ranked by games lost, then wins declined, then blocks missed, then
  move quality.

400k self-play episodes, γ = 0.9, ε annealed 1.0 → 0.05. Trains in about four
minutes on a laptop CPU.

γ is 0.9 rather than the more usual 0.95 deliberately. With a terminal reward
of 1, a shallower discount widens the gap between winning now and winning two
plies later, keeping it clear of the network's approximation error so that a
win already on the board always outranks a slower one.

---

## Design notes

A tic-tac-toe move is 0.022 ms of computation and a network round trip is
several hundred times that, so the interesting latency work is in removing
round trips rather than in the model.

- **Your move renders immediately.** The client draws your mark on click, then
  asks the agent for its reply, so the board never waits on the network to
  acknowledge your own input.
- **Wins and draws never touch the network.** The client implements the same
  rules as the server, so a move that ends the game resolves locally.
- **Starting a game costs no request.** The API is stateless, so a new game as
  X is simply an empty board.
- **The backend is warmed on page load.** A `/health` ping fires as soon as the
  page mounts, so a container that has scaled to zero wakes while you are still
  choosing a mark. If a request is still outstanding the UI says
  "Waking the server…" rather than appearing frozen.
- **`preconnect` to the API origin**, so the first request does not also pay for
  DNS and the TLS handshake.
- **Both agents load once, at startup**, in the app's lifespan hook, so a cold
  container absorbs that cost before it accepts traffic.
- **`torch.set_num_threads(1)`.** Spinning up a thread pool to multiply a 1×9
  vector costs more than it saves and hurts tail latency on a small container.
- **CORS preflight is cached for 24 hours**, rather than the 10-minute default,
  so a move does not pay an extra `OPTIONS` round trip.

Move selection is deliberately *not* memoised. A cache would save roughly
0.1 ms and cost the agent its tie-breaking variety, so every game against a
given opening would play out identically.

---

## Project layout

```
backend/
  app/
    main.py        FastAPI app: routes, CORS, startup hooks
    engine.py      rules and position validation (no framework, no torch)
    policy.py      loads the trained artifacts once, serves moves
    schemas.py     request/response models
  rl_agents/
    board.py       board representation shared by both agents
    q_learning.py  tabular agent
    dqn.py         network, replay buffer, Bellman loss
    minimax.py     perfect play, evaluation baseline only
    artifacts/     trained Q-table and network weights
  train.py         training and evaluation entry point
  tests/           41 tests: rules, validation, agent strength, API
frontend/
  src/
    App.jsx        game state and the optimistic-update flow
    api.js         API client, timeouts, warm-up
    game.js        client-side rules, mirrored from engine.py
    components/    Board, Controls, ResultModal
  src/*.test.*     28 tests
```

---

## Running locally

```bash
# 1. Backend  →  http://localhost:8000  (docs at /docs)
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements-dev.txt
uvicorn app.main:app --reload

# 2. Frontend  →  http://localhost:5173
cd frontend
npm install
npm run dev
```

The frontend defaults to `http://localhost:8000`. To point it elsewhere, copy
`frontend/.env.example` to `frontend/.env` and set `VITE_API_URL`.

### Tests

```bash
cd backend  && pytest     # 41 tests
cd frontend && npm test   # 28 tests
```

### Training the agents

Both agents ship trained, so this is only needed to reproduce them.

```bash
cd backend
python train.py                  # both agents, ~4 minutes, then reports the table above
python train.py --agent dqn      # just the network
python train.py --quick          # short run, to exercise the pipeline
```

Artifacts are written to `backend/rl_agents/artifacts/` and are what the API
serves.

---

## Deployment

The frontend is a static bundle on **Vercel**, built from `frontend/` (see
`frontend/vercel.json`); the
backend runs as a container anywhere that takes a `Dockerfile`.

```bash
cd backend
docker build -t ttt-rl .
docker run -p 8000:8000 ttt-rl
```

The image installs the CPU-only PyTorch wheel (`torch==2.8.0+cpu`). The
default pulls in the whole CUDA stack, which buys nothing for a 9-input network
and costs minutes of cold start. It still lands at about 1 GB, most of it
PyTorch itself; the agents load in roughly 400 ms once the container is up.

Two environment variables:

| Where | Variable | Value |
|-------|----------|-------|
| Vercel | `VITE_API_URL` | the backend's public URL |
| Backend | `ALLOWED_ORIGINS` | the Vercel URL, comma-separated for several (defaults to `*`) |

Because cold starts dominate the remaining latency, a host that keeps the
container warm makes far more difference than any further code change.
