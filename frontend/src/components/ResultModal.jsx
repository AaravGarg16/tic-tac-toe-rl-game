export function ResultModal({ outcome, onPlayAgain, onChangeSettings }) {
  const tone = outcome === "win" ? "good" : outcome === "loss" ? "bad" : "";
  const message =
    outcome === "draw" ? "It's a draw." : outcome === "win" ? "You won! 🎉" : "AI wins — try again! 🤖";

  return (
    <div className="overlay" role="dialog" aria-modal="true" aria-label="Game over">
      <div className={`modal ${tone}`}>
        <div className="modal-title single">{message}</div>
        <div className="modal-actions">
          <button type="button" className="btn ghost" onClick={onChangeSettings}>
            Change settings
          </button>
          <button type="button" className="btn primary" onClick={onPlayAgain}>
            Play again
          </button>
        </div>
      </div>
    </div>
  );
}
