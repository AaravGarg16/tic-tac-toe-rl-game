export function Controls({ player, difficulty, started, onPlayer, onDifficulty, onStart, onReset }) {
  return (
    <div className="controls">
      <div className="seg" role="group" aria-label="Your mark">
        {["X", "O"].map((mark) => (
          <button
            key={mark}
            type="button"
            className={`seg-btn ${player === mark ? "active" : ""}`}
            onClick={() => onPlayer(mark)}
            disabled={started}
            aria-pressed={player === mark}
          >
            Play as {mark}
          </button>
        ))}
      </div>

      <div className="seg" role="group" aria-label="Difficulty">
        {[
          ["easy", "Easy"],
          ["hard", "Hard"],
        ].map(([value, label]) => (
          <button
            key={value}
            type="button"
            className={`seg-btn ${difficulty === value ? "active" : ""}`}
            onClick={() => onDifficulty(value)}
            disabled={started}
            aria-pressed={difficulty === value}
          >
            {label}
          </button>
        ))}
      </div>

      {started ? (
        <button type="button" className="btn small" onClick={onReset}>
          Reset
        </button>
      ) : (
        <button type="button" className="btn primary" onClick={onStart}>
          Start Game
        </button>
      )}
    </div>
  );
}
