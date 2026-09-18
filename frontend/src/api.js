/**
 * Client for the FastAPI backend.
 *
 * The API is stateless: every request carries the whole position, so there is
 * no session to create and no `/new` round trip before the first move.
 */

const BASE = (import.meta.env.VITE_API_URL || "http://localhost:8000").replace(/\/$/, "");

/** A free-tier container can take a while to wake, so the budget is generous. */
const REQUEST_TIMEOUT_MS = 25_000;

export class ApiError extends Error {}

/**
 * Combine a caller's abort signal with a timeout.
 *
 * `AbortSignal.any` would do this in one line but is not available everywhere
 * the tests run, and forwarding the caller's signal alone would silently drop
 * the timeout, leaving a hung request spinning forever.
 */
function deadline(callerSignal, ms) {
  const controller = new AbortController();
  let timedOut = false;

  const timer = setTimeout(() => {
    timedOut = true;
    controller.abort();
  }, ms);

  const forward = () => controller.abort();
  if (callerSignal) {
    if (callerSignal.aborted) forward();
    else callerSignal.addEventListener("abort", forward, { once: true });
  }

  return {
    signal: controller.signal,
    get timedOut() {
      return timedOut;
    },
    release() {
      clearTimeout(timer);
      callerSignal?.removeEventListener("abort", forward);
    },
  };
}

async function post(path, body, { signal } = {}) {
  const budget = deadline(signal, REQUEST_TIMEOUT_MS);
  try {
    let response;
    try {
      response = await fetch(`${BASE}${path}`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: budget.signal,
      });
    } catch {
      throw new ApiError(
        budget.timedOut ? "The server took too long to respond." : "Could not reach the server.",
      );
    }

    if (!response.ok) {
      const detail = await response.json().catch(() => null);
      const message = typeof detail?.detail === "string" ? detail.detail : null;
      throw new ApiError(message ?? `Server error (${response.status}).`);
    }
    return response.json();
  } finally {
    budget.release();
  }
}

/** Ask the agent for its reply to `board`. Resolves to `{ move, board, status, winning_line }`. */
export function requestMove(board, aiMark, difficulty, options) {
  return post("/move", { board, ai_mark: aiMark, difficulty }, options);
}

/**
 * Fire-and-forget liveness ping.
 *
 * Called as soon as the page mounts so a sleeping container starts waking while
 * you are still picking a mark and a difficulty, instead of after your first
 * click. Failures are ignored on purpose: it is a warm-up, not a gate.
 */
export function warmUp() {
  const budget = deadline(null, REQUEST_TIMEOUT_MS);
  fetch(`${BASE}/health`, { signal: budget.signal })
    .catch(() => {})
    .finally(() => budget.release());
}

export const API_ORIGIN = (() => {
  try {
    return new URL(BASE).origin;
  } catch {
    return null;
  }
})();
