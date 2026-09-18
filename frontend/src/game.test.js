import { describe, expect, it } from "vitest";

import { emptyBoard, other, statusOf, winningLine } from "./game";

const b = (s) => [...s].map((c) => (c === "." ? "" : c));

describe("winningLine", () => {
  it.each([
    ["XXX.O.O..", [0, 1, 2]],
    ["OO.OO.XXX", [6, 7, 8]],
    ["X..X..X.O", [0, 3, 6]],
    ["X...X...X", [0, 4, 8]],
    ["..O.O.O..", [2, 4, 6]],
  ])("finds the line in %s", (board, expected) => {
    expect(winningLine(b(board))).toEqual(expected);
  });

  it("returns null when nobody has won", () => {
    expect(winningLine(b("XX.OO...."))).toBeNull();
    expect(winningLine(b("XOXXOOOXX"))).toBeNull();
  });
});

describe("statusOf", () => {
  it.each([
    ["XXX.O.O..", "X"],
    [".O..O..OX", "O"],
    ["XOXXOOOXX", "draw"],
    [".........", "in_progress"],
    ["XX.OO....", "in_progress"],
  ])("reads %s as %s", (board, expected) => {
    expect(statusOf(b(board))).toBe(expected);
  });
});

describe("emptyBoard", () => {
  it("starts with nine blank cells", () => {
    expect(emptyBoard()).toEqual(Array(9).fill(""));
    expect(statusOf(emptyBoard())).toBe("in_progress");
  });
});

describe("other", () => {
  it("swaps marks", () => {
    expect(other("X")).toBe("O");
    expect(other("O")).toBe("X");
  });
});
