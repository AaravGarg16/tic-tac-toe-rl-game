import { useCallback, useEffect, useMemo, useRef, useState } from "react";

import { requestMove, warmUp } from "./api";
import { Board } from "./components/Board";
import { Controls } from "./components/Controls";
import { ResultModal } from "./components/ResultModal";
import { emptyBoard, other, statusOf, winningLine } from "./game";
import { useConfetti } from "./useConfetti";
import "./styles.css";

/** How long the agent may think before we admit the server is probably asleep. */
const COLD_START_HINT_MS = 1500;

export default function App() {
  const [board, setBoard] = useState(emptyBoard);
  const [player, setPlayer] = useState("X");
  const [difficulty, setDifficulty] = useState("hard");
  const [started, setStarted] = useState(false);
  const [thinking, setThinking] = useState(false);
  // True from the moment we ask the agent until its reply lands. It stays true
  // after a failed request, because the agent still owes us a move -- without
  // it you could click again and put two of your marks on the board in a row.
  const [owed, setOwed] = useState(false);
  const [slow, setSlow] = useState(false);
  const [error, setError] = useState(null);

  const ai = other(player);
  const status = useMemo(() => statusOf(board), [board]);
  const winCells = useMemo(() => winningLine(board), [board]);
  const finished = started && status !== "in_progress";
  const outcome = status === "draw" ? "draw" : status === player ? "win" : "loss";

  const confetti = useConfetti();
  const inFlight = useRef(null);
  const pending = useRef(null);

  // Start waking the backend the moment the page loads, so a sleeping
  // container is already booting while you pick a mark and a difficulty
  // rather than after your first click.
  useEffect(() => {
    warmUp();
  }, []);

  useEffect(() => {
    if (finished && outcome === "win") confetti();
  }, [finished, outcome, confetti]);

  const cancelInFlight = useCallback(() => {
    inFlight.current?.abort();
    inFlight.current = null;
  }, []);

  useEffect(() => cancelInFlight, [cancelInFlight]);

  const askAgent = useCallback(
    async (position, aiMark, level) => {
      cancelInFlight();
      const controller = new AbortController();
      inFlight.current = controller;
      pending.current = { position, aiMark, level };

      setThinking(true);
      setOwed(true);
      setError(null);
      const hint = setTimeout(() => setSlow(true), COLD_START_HINT_MS);

      try {
        const reply = await requestMove(position, aiMark, level, { signal: controller.signal });
        if (controller.signal.aborted) return;
        setBoard(reply.board);
        pending.current = null;
        setOwed(false);
      } catch (cause) {
        if (controller.signal.aborted) return;
        // Your own move stays on the board; only the agent's reply is missing,
        // so a retry just resends the same position.
        setError(cause.message);
      } finally {
        clearTimeout(hint);
        if (inFlight.current === controller) {
          inFlight.current = null;
          setThinking(false);
          setSlow(false);
        }
      }
    },
    [cancelInFlight],
  );

  const startGame = useCallback(() => {
    cancelInFlight();
    pending.current = null;
    const fresh = emptyBoard();
    setBoard(fresh);
    setStarted(true);
    setOwed(false);
    setError(null);
    // X always opens, so if you chose O the agent moves first.
    if (player === "O") askAgent(fresh, "X", difficulty);
  }, [askAgent, cancelInFlight, difficulty, player]);

  const resetAll = useCallback(() => {
    cancelInFlight();
    pending.current = null;
    setStarted(false);
    setThinking(false);
    setOwed(false);
    setSlow(false);
    setError(null);
    setBoard(emptyBoard());
  }, [cancelInFlight]);

  const play = useCallback(
    (index) => {
      if (!started || owed || board[index] !== "" || status !== "in_progress") return;

      // Draw your mark immediately rather than waiting for the round trip.
      const next = [...board];
      next[index] = player;
      setBoard(next);

      // If that move ended the game, there is nothing to ask the agent and the
      // result appears with no network call at all.
      if (statusOf(next) !== "in_progress") return;
      askAgent(next, ai, difficulty);
    },
    [ai, askAgent, board, difficulty, owed, player, started, status],
  );

  const retry = useCallback(() => {
    const last = pending.current;
    if (last) askAgent(last.position, last.aiMark, last.level);
  }, [askAgent]);

  return (
    <div className={`wrap ${finished ? "modal-open" : ""}`}>
      <h1 className="title">
        Tic Tac Toe <span className="pill">RL</span>
      </h1>

      <Controls
        player={player}
        difficulty={difficulty}
        started={started}
        onPlayer={setPlayer}
        onDifficulty={setDifficulty}
        onStart={startGame}
        onReset={resetAll}
      />

      <Board
        board={board}
        winningCells={winCells}
        playable={started && !owed && status === "in_progress"}
        onPlay={play}
        boardClass={[
          "board",
          finished && outcome === "win" ? "win-human" : "",
          finished && outcome === "loss" ? "win-ai" : "",
          finished ? "no-bg glow-first" : "",
        ]
          .filter(Boolean)
          .join(" ")}
        gridClass={["grid", finished && outcome === "loss" ? "shake" : "", finished ? "finished" : ""]
          .filter(Boolean)
          .join(" ")}
      />

      <div className="result" role="status" aria-live="polite">
        {!started && "Pick a mark and a difficulty to start."}
        {started && !finished && !error && (thinking ? <Thinking slow={slow} /> : `You are ${player}`)}
        {started && finished && (outcome === "draw" ? "It's a draw." : outcome === "win" ? "You win! 🎉" : "AI wins! 🤖")}
      </div>

      {error && (
        <div className="error" role="alert">
          <span>{error}</span>
          <button type="button" className="btn small" onClick={retry}>
            Retry
          </button>
        </div>
      )}

      {finished && <ResultModal outcome={outcome} onPlayAgain={startGame} onChangeSettings={resetAll} />}
    </div>
  );
}

function Thinking({ slow }) {
  return (
    <span className="thinking">
      <span className="dots" aria-hidden="true">
        <i />
        <i />
        <i />
      </span>
      {slow ? "Waking the server…" : "Thinking…"}
    </span>
  );
}
