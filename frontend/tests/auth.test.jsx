import React from "react";
import { beforeEach, describe, expect, it, vi } from "vitest";
import { render, screen, waitFor } from "@testing-library/react";

vi.mock("../src/lib/api.js", () => ({
  api: vi.fn(),
  getToken: () => null,
  setToken: vi.fn(),
  initAuth: vi.fn(async () => null),
  refreshAuth: vi.fn(async () => null),
}));

import { AuthProvider, useAuth } from "../src/context/AuthContext.jsx";
import { api } from "../src/lib/api.js";

function Harness() {
  const { user, error, login } = useAuth();
  return (
    <div>
      <button onClick={() => login("a@b.c", "password")}>login</button>
      {user && <span data-testid="user">{user.email}</span>}
      {error && <span data-testid="error">{error}</span>}
    </div>
  );
}

describe("AuthContext", () => {
  beforeEach(() => {
    vi.clearAllMocks();
    localStorage.clear();
  });

  it("persists the user on a successful login", async () => {
    api.mockResolvedValue({ token: "jwt-new", user: { id: "1", email: "a@b.c" } });
    render(
      <AuthProvider>
        <Harness />
      </AuthProvider>,
    );

    screen.getByText("login").click();
    await waitFor(() => expect(screen.getByTestId("user").textContent).toBe("a@b.c"));
    expect(localStorage.getItem("rubicon_user")).toContain("a@b.c");
  });

  it("surfaces the login error without persisting a user", async () => {
    api.mockRejectedValue(new Error("invalid credentials"));
    render(
      <AuthProvider>
        <Harness />
      </AuthProvider>,
    );

    screen.getByText("login").click();
    await waitFor(() => expect(screen.getByTestId("error").textContent).toBe("invalid credentials"));
    expect(localStorage.getItem("rubicon_user")).toBeNull();
  });
});