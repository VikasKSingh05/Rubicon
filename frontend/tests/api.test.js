import { describe, it, expect, beforeEach, vi } from "vitest";
import { api, setToken, getToken } from "../src/lib/api.js";

// Minimal storage polyfill for the node test environment.
const store = new Map();
globalThis.localStorage = {
  getItem: (k) => store.get(k) ?? null,
  setItem: (k, v) => store.set(k, String(v)),
  removeItem: (k) => store.delete(k),
  clear: () => store.clear(),
};

describe("api client", () => {
  beforeEach(() => {
    localStorage.clear();
    vi.restoreAllMocks();
  });

  it("attaches the bearer token when one is stored", async () => {
    setToken("jwt-123");
    global.fetch = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({}),
    }));

    await api("/assessments");
    const [url, opts] = global.fetch.mock.calls[0];
    expect(url.endsWith("/assessments")).toBe(true);
    expect(opts.headers.Authorization).toBe("Bearer jwt-123");
  });

  it("throws an Error carrying the server message on failure", async () => {
    setToken("jwt-123");
    global.fetch = vi.fn(async () => ({
      ok: false,
      status: 400,
      json: async () => ({ error: "email and password are required" }),
    }));

    await expect(api("/auth/register")).rejects.toThrow(/email and password are required/);
  });

  it("serializes JSON bodies for non-FormData payloads", async () => {
    global.fetch = vi.fn(async () => ({
      ok: true,
      status: 200,
      json: async () => ({}),
    }));

    await api("/auth/login", { method: "POST", body: { email: "a@b.c", password: "x" } });
    const [, opts] = global.fetch.mock.calls[0];
    expect(opts.headers["Content-Type"]).toBe("application/json");
    expect(opts.body).toBe(JSON.stringify({ email: "a@b.c", password: "x" }));
  });

  it("does not JSON-serialize FormData uploads", async () => {
    const form = new FormData();
    global.fetch = vi.fn(async () => ({
      ok: true,
      status: 201,
      json: async () => ({}),
    }));

    await api("/upload", { method: "POST", body: form });
    const [, opts] = global.fetch.mock.calls[0];
    expect(opts.body).toBe(form);
    expect(opts.headers["Content-Type"]).toBeUndefined();
  });

  it("getToken round-trips", () => {
    expect(getToken()).toBeNull();
    setToken("abc");
    expect(getToken()).toBe("abc");
  });
});