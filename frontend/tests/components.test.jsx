import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { fireEvent, render, screen, waitFor } from "@testing-library/react";
import { MemoryRouter, Route, Routes } from "react-router-dom";
import { AuthProvider } from "../src/context/AuthContext.jsx";
import TopBar from "../src/components/TopBar.jsx";
import DashboardPage from "../src/pages/DashboardPage.jsx";
import UploadModal from "../src/components/UploadModal.jsx";

vi.mock("react-leaflet", () => ({
  MapContainer: ({ children }) => <div>{children}</div>,
  TileLayer: () => null,
  GeoJSON: ({ children }) => <div>{children}</div>,
  Popup: ({ children }) => <div>{children}</div>,
  useMap: () => ({ flyTo: vi.fn() }),
}));

vi.mock("../src/lib/api.js", () => ({
  api: vi.fn(),
  getToken: () => localStorage.getItem("rubicon_token"),
  setToken: (token) => {
    if (token) localStorage.setItem("rubicon_token", token);
    else localStorage.removeItem("rubicon_token");
  },
}));

import { api } from "../src/lib/api.js";

const asst = (overrides = {}) => ({
  id: "a1",
  prediction: "Moderate",
  confidence: 0.82,
  severity: "moderate",
  state: "analyzed",
  center: { lng: -95.36, lat: 29.76 },
  geojson_polygon: {
    type: "Polygon",
    coordinates: [[[-95.36, 29.76], [-95.36, 29.77], [-95.35, 29.77], [-95.35, 29.76], [-95.36, 29.76]]],
  },
  txHash: null,
  chainVerified: false,
  createdAt: "2026-09-07T12:00:00.000Z",
  ...overrides,
});

function flush() {
  return new Promise((r) => setTimeout(r, 0));
}

beforeEach(() => {
  vi.clearAllMocks();
  localStorage.clear();
});

describe("TopBar", () => {
  it("shows the signed-in user's email and logs out to /login", async () => {
    localStorage.setItem("rubicon_token", "token-123");
    localStorage.setItem("rubicon_user", JSON.stringify({ id: "1", email: "user@example.com" }));

    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/"]}>
          <Routes>
            <Route path="/" element={<TopBar onUpload={() => {}} />} />
            <Route path="/login" element={<div>login-route</div>} />
          </Routes>
        </MemoryRouter>
      </AuthProvider>
    );

    expect(screen.getByText("user@example.com")).toBeInTheDocument();
    fireEvent.click(screen.getByText("Logout"));
    expect(localStorage.getItem("rubicon_user")).toBeNull();
    await screen.findByText("login-route");
  });
});

describe("UploadModal", () => {
  it("rejects unsupported file types before submitting", async () => {
    render(<UploadModal onClose={() => {}} onCreated={() => {}} />);

    const inputs = screen.getAllByRole("button").filter((b) => b.getAttribute("aria-label") === "Choose Hyperspectral (HSI) file");
    const dropzones = screen.getAllByLabelText(/Choose Hyperspectral/i);
    expect(dropzones.length).toBeGreaterThan(0);

    const hsiInput = document.querySelectorAll('input[type="file"]')[0];
    fireEvent.change(hsiInput, {
      target: {
        files: [new File(["x"], "photo.exe", { type: "application/octet-stream" })],
      },
    });

    expect(await screen.findByText(/Only .tif, .tiff files are supported/i)).toBeInTheDocument();
    expect(screen.getByText("Run Assessment")).toBeDisabled();
  });

  it("submits both files and calls onCreated with the created assessment", async () => {
    const onCreated = vi.fn();
    let resolveApi;
    api.mockReturnValue(
      new Promise((resolve) => {
        resolveApi = resolve;
      }),
    );

    render(<UploadModal onClose={() => {}} onCreated={onCreated} />);

    const inputs = document.querySelectorAll('input[type="file"]');
    fireEvent.change(inputs[0], {
      target: { files: [new File(["hsi"], "scan.tiff", { type: "image/tiff" })] },
    });
    fireEvent.change(inputs[1], {
      target: { files: [new File(["lidar"], "scan.laz", { type: "application/octet-stream" })] },
    });

    const submit = screen.getByText("Run Assessment");
    await waitFor(() => expect(submit).not.toBeDisabled());
    fireEvent.click(submit);

    expect(await screen.findByText("Uploading")).toBeInTheDocument();

    resolveApi({
      assessment: asst(),
      links: { detail: "/assessments/a1" },
    });

    await waitFor(() => expect(onCreated).toHaveBeenCalledTimes(1));

    const expected = api.mock.calls[0][1].body;
    expect(api.mock.calls[0][0]).toBe("/upload");
    expect(expected instanceof FormData).toBe(true);
    expect(expected.get("hsi").name).toBe("scan.tiff");
    expect(expected.get("lidar").name).toBe("scan.laz");
  });

  it("shows server errors and blocks resubmission", async () => {
    api.mockRejectedValue(new Error("analysis failed"));
    render(<UploadModal onClose={() => {}} onCreated={() => {}} />);

    const inputs = document.querySelectorAll('input[type="file"]');
    fireEvent.change(inputs[0], {
      target: { files: [new File(["hsi"], "scan.tiff", { type: "image/tiff" })] },
    });
    fireEvent.change(inputs[1], {
      target: { files: [new File(["lidar"], "scan.las", { type: "application/octet-stream" })] },
    });

    const submit = screen.getByText("Run Assessment");
    await waitFor(() => expect(submit).not.toBeDisabled());
    fireEvent.click(submit);

    expect(await screen.findByText("analysis failed")).toBeInTheDocument();
    expect(screen.getByText("Run Assessment")).toBeInTheDocument();
  });
});

describe("DashboardPage", () => {
  it("shows a loading state while the first page is in flight", () => {
    api.mockReturnValue(new Promise(() => {}));
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/"]}>
          <DashboardPage />
        </MemoryRouter>
      </AuthProvider>
    );
    expect(screen.getByText("Loading assessments…")).toBeInTheDocument();
  });

  it("renders empty-state text when there are no assessments", async () => {
    api.mockResolvedValue({ assessments: [], total: 0, limit: 50, offset: 0 });
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/"]}>
          <DashboardPage />
        </MemoryRouter>
      </AuthProvider>
    );
    expect(await screen.findByText("No assessments yet. Upload a scan to get started.")).toBeInTheDocument();
    expect(screen.getByText("No assessments mapped yet")).toBeInTheDocument();
    expect(screen.queryByText("Load more")).not.toBeInTheDocument();
  });

  it("renders the list and hides the load-more button when all items loaded", async () => {
    api.mockResolvedValue({ assessments: [asst()], total: 1, limit: 50, offset: 0 });
    render(
      <AuthProvider>
        <MemoryRouter initialEntries={["/"]}>
          <DashboardPage />
        </MemoryRouter>
      </AuthProvider>
    );

    expect(await screen.findByText("Moderate")).toBeInTheDocument();
    await flush();
    expect(screen.queryByText("Load more")).not.toBeInTheDocument();
  });
});