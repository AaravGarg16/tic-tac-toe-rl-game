import { afterEach, describe, expect, it, vi } from "vitest";

import { ApiError, requestMove } from "./api";

const board = ["X", "", "", "", "", "", "", "", ""];
const ask = (options) => requestMove(board, "O", "hard", options);

afterEach(() => {
  vi.unstubAllGlobals();
  vi.useRealTimers();
});

describe("requestMove", () => {
  it("returns the agent's reply", async () => {
    const reply = { move: 4, board, status: "in_progress" };
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: true, json: async () => reply })));
    await expect(ask()).resolves.toEqual(reply);
  });

  it("surfaces the server's own explanation of a rejected position", async () => {
    vi.stubGlobal(
      "fetch",
      vi.fn(async () => ({ ok: false, status: 422, json: async () => ({ detail: "it is X's turn, not O's" }) })),
    );
    await expect(ask()).rejects.toThrow("it is X's turn, not O's");
  });

  it("falls back to the status code when there is no detail", async () => {
    vi.stubGlobal("fetch", vi.fn(async () => ({ ok: false, status: 503, json: async () => null })));
    await expect(ask()).rejects.toThrow("Server error (503).");
  });

  it("reports an unreachable server", async () => {
    vi.stubGlobal("fetch", vi.fn(() => Promise.reject(new TypeError("failed to fetch"))));
    await expect(ask()).rejects.toBeInstanceOf(ApiError);
    await expect(ask()).rejects.toThrow("Could not reach the server.");
  });

  it("times out a request that never settles, even with a caller signal attached", async () => {
    vi.useFakeTimers();
    // Forwarding a caller signal must not displace the timeout, or a request
    // that never settles would hang indefinitely.
    vi.stubGlobal(
      "fetch",
      vi.fn((_url, { signal }) =>
        new Promise((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
      ),
    );

    const promise = ask({ signal: new AbortController().signal });
    const assertion = expect(promise).rejects.toThrow("The server took too long to respond.");
    await vi.advanceTimersByTimeAsync(25_000);
    await assertion;
  });

  it("does not call a caller-cancelled request a timeout", async () => {
    const controller = new AbortController();
    vi.stubGlobal(
      "fetch",
      vi.fn((_url, { signal }) =>
        new Promise((_resolve, reject) => {
          signal.addEventListener("abort", () => reject(new DOMException("aborted", "AbortError")));
        }),
      ),
    );

    const promise = ask({ signal: controller.signal });
    controller.abort();
    await expect(promise).rejects.toThrow("Could not reach the server.");
  });
});
