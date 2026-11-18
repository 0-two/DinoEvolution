import { describe, expect, it } from "vitest";
import { crosshairText, formatValue, nearestIndex, xFromPointer, type Point } from "./figureMath";

const pts: Point[] = [
  [0, 10],
  [1, 20],
  [2, 30],
  [4, 50],
];

describe("nearestIndex", () => {
  it("finds the closest x, including both ends", () => {
    expect(nearestIndex(pts, -5)).toBe(0);
    expect(nearestIndex(pts, 0.4)).toBe(0);
    expect(nearestIndex(pts, 0.6)).toBe(1);
    expect(nearestIndex(pts, 3.1)).toBe(3);
    expect(nearestIndex(pts, 99)).toBe(3);
  });

  it("handles empty and single-point series", () => {
    expect(nearestIndex([], 1)).toBe(-1);
    expect(nearestIndex([[3, 1]], 100)).toBe(0);
  });

  it("breaks an exact tie toward the lower point", () => {
    expect(nearestIndex(pts, 3)).toBe(2);
  });
});

describe("xFromPointer", () => {
  it("maps the box to the axis range and clamps outside it", () => {
    const box = { x: 100, w: 200 };
    expect(xFromPointer(100, box, [0, 10])).toBe(0);
    expect(xFromPointer(200, box, [0, 10])).toBe(5);
    expect(xFromPointer(900, box, [0, 10])).toBe(10);
    expect(xFromPointer(-50, box, [2, 4])).toBe(2);
  });
});

describe("formatValue", () => {
  it("uses fewer decimals for larger numbers", () => {
    expect(formatValue(1234.567)).toBe("1,235");
    expect(formatValue(12.345)).toBe("12.3");
    expect(formatValue(1.2345)).toBe("1.23");
    expect(formatValue(0)).toBe("0");
  });
});

describe("crosshairText", () => {
  it("lists the x position and aligned series rows", () => {
    const text = crosshairText("M steps", 1.5, [
      { label: "GA", value: 1200 },
      { label: "CMA-ES", value: 40.2 },
    ]);
    expect(text.split("\n")).toEqual(["M steps 1.5", "GA      1,200", "CMA-ES  40.2"]);
  });
});
