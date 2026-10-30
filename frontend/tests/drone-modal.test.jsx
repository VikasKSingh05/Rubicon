import { describe, it, expect, vi } from "vitest";
import React from "react";
import { render, screen, fireEvent, waitFor } from "@testing-library/react";
import DroneInspectorModal from "../src/components/DroneInspectorModal.jsx";
import * as apiModule from "../src/lib/api.js";

describe("DroneInspectorModal", () => {
  it("does not render when isOpen is false", () => {
    const { container } = render(
      <DroneInspectorModal isOpen={false} onClose={() => {}} />
    );
    expect(container.firstChild).toBeNull();
  });

  it("renders with parent H3 cell context when open", () => {
    render(
      <DroneInspectorModal
        isOpen={true}
        onClose={() => {}}
        parentCellId="8928308280fffff"
      />
    );
    expect(screen.getByText("Stage 2: Drone Triage Inspector")).toBeInTheDocument();
    expect(screen.getByText("8928308280fffff")).toBeInTheDocument();
  });

  it("submits drone image and displays structural failure modes", async () => {
    const fakeResponse = {
      image_sha256: "abcdef1234567890",
      parent_cell_id: "8928308280fffff",
      damage_percentage: 42.5,
      damage_level: "major",
      severity: "ORANGE",
      failure_modes: {
        roof_collapse: { detected: true, confidence: 0.88, area_pct: 35.0 },
        facade_failure: { detected: false, confidence: 0.12, area_pct: 0.0 },
        debris_field: { detected: true, confidence: 0.74, area_pct: 22.0 },
      },
      model_version: "rubicon-drone-v1",
      exec_ms: 12.4,
    };

    const spy = vi.spyOn(apiModule, "api").mockResolvedValueOnce(fakeResponse);

    render(
      <DroneInspectorModal
        isOpen={true}
        onClose={() => {}}
        parentCellId="8928308280fffff"
      />
    );

    const file = new File(["fake-drone-image-data"], "inspection.png", {
      type: "image/png",
    });

    const fileInput = document.querySelector('input[type="file"]');
    fireEvent.change(fileInput, { target: { files: [file] } });

    const analyzeBtn = screen.getByRole("button", { name: /run drone triage/i });
    fireEvent.click(analyzeBtn);

    await waitFor(() => {
      expect(screen.getByText(/42.5% Area Damaged/i)).toBeInTheDocument();
    });

    expect(screen.getByText(/Roof Structural Collapse/i)).toBeInTheDocument();
    expect(screen.getByText(/Peripheral Debris Field/i)).toBeInTheDocument();

    spy.mockRestore();
  });
});
