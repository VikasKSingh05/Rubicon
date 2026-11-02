import { describe, expect, it, vi } from "vitest";
import React from "react";
import { fireEvent, render, screen } from "@testing-library/react";
import Modal from "../src/components/Modal.jsx";

describe("Modal", () => {
  it("portals above map stacking layers and closes on Escape", () => {
    const onClose = vi.fn();
    const { unmount } = render(
      <>
        <div data-testid="map" style={{ zIndex: 1000 }}>Map</div>
        <Modal onClose={onClose} labelledBy="modal-title" panelClassName="bg-white">
          <h2 id="modal-title">Settings</h2>
        </Modal>
      </>,
    );

    const dialog = screen.getByRole("dialog", { name: "Settings" });
    expect(dialog.parentElement).toBe(document.body);
    expect(dialog).toHaveClass("z-[2000]");
    expect(document.body).toContainElement(dialog);

    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).toHaveBeenCalledOnce();

    unmount();
  });

  it("does not close when Escape is disabled", () => {
    const onClose = vi.fn();
    render(
      <Modal
        onClose={onClose}
        labelledBy="busy-title"
        panelClassName="bg-white"
        closeDisabled
      >
        <h2 id="busy-title">Working</h2>
      </Modal>,
    );

    fireEvent.keyDown(window, { key: "Escape" });
    expect(onClose).not.toHaveBeenCalled();
  });
});
