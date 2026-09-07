import { describe, it, expect } from "vitest";
import { severityOf, SEVERITY } from "../src/lib/severity.js";

describe("severity design system", () => {
  it("maps severe collapse predictions to the severe tier", () => {
    const s = severityOf("Severe Collapse");
    expect(s).toBe(SEVERITY.severe);
    expect(s.color).toBe("#DC2626");
    expect(s.icon).toBe("⚠");
  });

  it("maps moderate predictions to the moderate tier", () => {
    expect(severityOf("Moderate")).toBe(SEVERITY.moderate);
  });

  it("treats anything else as none/minimal (color not required alone)", () => {
    expect(severityOf("Green Space")).toBe(SEVERITY.none);
  });

  it("provides a labeled icon so color is never the only signal", () => {
    for (const s of Object.values(SEVERITY)) {
      expect(typeof s.label).toBe("string");
      expect(s.label.length).toBeGreaterThan(0);
    }
  });
});