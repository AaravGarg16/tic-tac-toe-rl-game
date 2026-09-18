import { render, screen, waitFor } from "@testing-library/react";
import userEvent from "@testing-library/user-event";
import { afterEach, beforeEach, describe, expect, it, vi } from "vitest";

import App from "./App";

const cell = (i) => screen.getAllByRole("gridcell")[i];

/** Stub `/move` with a canned agent reply; `/health` always succeeds. */
function stubApi(reply) {
  const fetchMock = vi.fn((url) => {
    if (String(url).endsWith("/health")) {
      return Promise.resolve({ ok: true, json: async () => ({ status: "ok" }) });
    }
    return typeof reply === "function"
      ? reply()
      : Promise.resolve({ ok: true, json: async () => reply });
  });
  vi.stubGlobal("fetch", fetchMock);
  return fetchMock;
}

const moveCalls = (mock) => mock.mock.calls.filter(([url]) => String(url).endsWith("/move"));

beforeEach(() => {
  vi.stubGlobal("matchMedia", () => ({ matches: true, addEventListener() {}, removeEventListener() {} }));
});

afterEach(() => {
  vi.unstubAllGlobals();
  vi.restoreAllMocks();
});

describe("App", () => {
  it("warms the backend on load, before any game starts", async () => {
    const fetchMock = stubApi({});
    render(<App />);
    await waitFor(() => expect(fetchMock).toHaveBeenCalled());
    expect(String(fetchMock.mock.calls[0][0])).toMatch(/\/health$/);
  });

  it("shows your mark immediately, without waiting for the agent", async () => {
    const user = userEvent.setup();
    // The agent's reply never resolves, so anything rendered came from the client.
    stubApi(() => new Promise(() => {}));
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Start Game" }));
    await user.click(cell(0));

    expect(cell(0)).toHaveTextContent("X");
    expect(await screen.findByText(/Thinking/)).toBeInTheDocument();
  });

  it("applies the agent's reply when it arrives", async () => {
    const user = userEvent.setup();
    stubApi({ move: 4, board: ["X", "", "", "", "O", "", "", "", ""], status: "in_progress" });
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Start Game" }));
    await user.click(cell(0));

    await waitFor(() => expect(cell(4)).toHaveTextContent("O"));
  });

  it("settles a winning move with no network call at all", async () => {
    const user = userEvent.setup();
    const fetchMock = stubApi(({ board }) => ({ ok: true, json: async () => ({ board }) }));
    render(<App />);
    await user.click(screen.getByRole("button", { name: "Start Game" }));

    // Drive to a position where X wins by taking cell 2.
    const replies = [
      { move: 3, board: ["X", "", "", "O", "", "", "", "", ""], status: "in_progress" },
      { move: 6, board: ["X", "X", "", "O", "", "", "O", "", ""], status: "in_progress" },
    ];
    let i = 0;
    vi.stubGlobal(
      "fetch",
      vi.fn((url) =>
        String(url).endsWith("/health")
          ? Promise.resolve({ ok: true, json: async () => ({ status: "ok" }) })
          : Promise.resolve({ ok: true, json: async () => replies[i++] }),
      ),
    );

    await user.click(cell(0));
    await waitFor(() => expect(cell(3)).toHaveTextContent("O"));
    await user.click(cell(1));
    await waitFor(() => expect(cell(6)).toHaveTextContent("O"));

    const before = moveCalls(fetch).length;
    await user.click(cell(2)); // completes the top row

    expect(await screen.findByText(/You win/)).toBeInTheDocument();
    expect(moveCalls(fetch).length).toBe(before);
  });

  it("keeps your move and offers a retry when the server is unreachable", async () => {
    const user = userEvent.setup();
    stubApi(() => Promise.reject(new TypeError("network down")));
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Start Game" }));
    await user.click(cell(0));

    expect(await screen.findByRole("alert")).toHaveTextContent(/Could not reach the server/);
    expect(cell(0)).toHaveTextContent("X");
    expect(screen.getByRole("button", { name: "Retry" })).toBeInTheDocument();
  });

  it("asks the agent to open when you play as O", async () => {
    const user = userEvent.setup();
    const fetchMock = stubApi({
      move: 4,
      board: ["", "", "", "", "X", "", "", "", ""],
      status: "in_progress",
    });
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Play as O" }));
    await user.click(screen.getByRole("button", { name: "Start Game" }));

    await waitFor(() => expect(cell(4)).toHaveTextContent("X"));
    expect(JSON.parse(moveCalls(fetchMock)[0][1].body).ai_mark).toBe("X");
  });

  it("stays locked after a failed request, so you cannot move twice", async () => {
    const user = userEvent.setup();
    stubApi(() => Promise.reject(new TypeError("network down")));
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Start Game" }));
    await user.click(cell(0));
    await screen.findByRole("alert");

    // The agent still owes a reply, so the board must not accept another mark.
    await user.click(cell(1));
    expect(cell(1)).toHaveTextContent("");
    expect(screen.getAllByRole("gridcell").filter((c) => c.textContent === "X")).toHaveLength(1);
  });

  it("resumes after a successful retry", async () => {
    const user = userEvent.setup();
    let fail = true;
    vi.stubGlobal(
      "fetch",
      vi.fn((url, init) => {
        if (String(url).endsWith("/health")) {
          return Promise.resolve({ ok: true, json: async () => ({ status: "ok" }) });
        }
        if (fail) {
          fail = false;
          return Promise.reject(new TypeError("network down"));
        }
        // Echo the submitted position with the agent's reply in the first free
        // cell, so the fake does not clobber marks the client just added.
        const board = [...JSON.parse(init.body).board];
        const move = board.indexOf("");
        board[move] = "O";
        return Promise.resolve({ ok: true, json: async () => ({ move, board, status: "in_progress" }) });
      }),
    );
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Start Game" }));
    await user.click(cell(0));
    await user.click(await screen.findByRole("button", { name: "Retry" }));

    await waitFor(() => expect(cell(1)).toHaveTextContent("O"));
    await user.click(cell(2));
    expect(cell(2)).toHaveTextContent("X");
  });

  it("ignores clicks while the agent is thinking", async () => {
    const user = userEvent.setup();
    stubApi(() => new Promise(() => {}));
    render(<App />);

    await user.click(screen.getByRole("button", { name: "Start Game" }));
    await user.click(cell(0));
    await user.click(cell(1));

    expect(cell(1)).toHaveTextContent("");
  });
});
