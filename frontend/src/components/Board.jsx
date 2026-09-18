export function Board({ board, winningCells, playable, onPlay, boardClass, gridClass }) {
  return (
    <div className={boardClass}>
      <div className={gridClass} role="grid" aria-label="Tic tac toe board">
        {board.map((cell, i) => (
          <button
            key={i}
            type="button"
            role="gridcell"
            className={`cell ${cell || "empty"} ${winningCells?.includes(i) ? "win" : ""}`}
            onClick={() => onPlay(i)}
            disabled={!playable || cell !== ""}
            aria-label={`Row ${Math.floor(i / 3) + 1}, column ${(i % 3) + 1}${cell ? `: ${cell}` : ", empty"}`}
          >
            {cell}
          </button>
        ))}
      </div>
    </div>
  );
}
