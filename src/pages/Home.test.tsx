import { describe, expect, it } from "vitest";
import { getPredictionSummary } from "./Home";

describe("getPredictionSummary", () => {
  it("returns an unknown breed and warning for low-confidence cat predictions", () => {
    const summary = getPredictionSummary({
      category: "Cat",
      name: "Maine Coon",
      confidence: 0.1188,
      predictions: [
        { label: "Maine Coon", confidence: 0.1188 },
        { label: "British Shorthair", confidence: 0.1085 },
        { label: "Bengal", confidence: 0.0998 },
        { label: "Persian", confidence: 0.089 },
        { label: "Bombay", confidence: 0.0847 },
      ],
    });

    expect(summary.animal).toBe("Cat");
    expect(summary.breed).toBe("Unknown");
    expect(summary.confidencePercent).toBe(11.88);
    expect(summary.isReliable).toBe(false);
    expect(summary.warning).toContain("too low");
    expect(summary.topPredictions).toHaveLength(5);
  });
});
