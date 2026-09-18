/**
 * Tic-tac-toe rules, mirrored from the backend's `app/engine.py`.
 *
 * The client needs these so it can settle a move the instant you click:
 * your own mark is drawn immediately, and if that move ends the game the
 * result is shown without ever touching the network.
 */

export const EMPTY = "";

export const LINES = [
  [0, 1, 2], [3, 4, 5], [6, 7, 8],
  [0, 3, 6], [1, 4, 7], [2, 5, 8],
  [0, 4, 8], [2, 4, 6],
];

export const emptyBoard = () => Array(9).fill(EMPTY);

export const other = (mark) => (mark === "X" ? "O" : "X");

/** The three cells that won, or null. */
export function winningLine(board) {
  for (const line of LINES) {
    const [a, b, c] = line;
    if (board[a] !== EMPTY && board[a] === board[b] && board[a] === board[c]) return line;
  }
  return null;
}

/** "X" | "O" | "draw" | "in_progress" */
export function statusOf(board) {
  const line = winningLine(board);
  if (line) return board[line[0]];
  return board.includes(EMPTY) ? "in_progress" : "draw";
}
