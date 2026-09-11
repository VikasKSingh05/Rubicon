import { describe, it, expect, beforeEach, vi } from "vitest";
import { api, setToken, getToken } from "../src/lib/api.js";

describe("api client 401 refresh replay", () => {
  beforeEach(() => {
    setToken(null);
    globalThis.localStorage?.clear?.();
    vi.restoreAllMocks();
  });

  it("refreshes once via the httpOnly cookie when an authed call returns 401", async () => {
    setToken("expired-token");
    const calls = [];
    global.fetch = vi.fn(async (url, opts) => {
      calls.push(url);
      if (String(url).endsWith("/assessments") && calls.length === 1) {
        return { ok: false, status: 401, json: async () => ({ error: "Unauthorized" }) };
      }
      if (String(url).endsWith("/auth/refresh")) {
        return { ok: true, status: 200, json: async () => ({ token: "fresh-token" }) };
      }
      return { ok: true, status: 200, json: async () => ({ assessments: [] }) };
    });

    const data = await api("/assessments");
    expect(data).toEqual({ assessments: [] });
    expect(calls.filter((u) => String(u).includes("/auth/refresh"))).toHaveLength(1);
    expect(getToken()).toBe("fresh-token");
    expect(global.fetch).toHaveBeenCalledTimes(3);
  });

  it("does not retry auth endpoints or loop on a failed refresh", async () => {
    setToken("expired-token");
    const seen = [];
    global.fetch = vi.fn(async (url) => {
      seen.push(String(url));
      return {
        ok: false,
        status: 401,
        json: async () => ({ error: "Unauthorized" }),
      };
    });

    await expect(api("/assessments")).rejects.toThrow(/Unauthorized/);
    // Exactly one refresh attempt, then the original 401 surfaces.
    expect(seen.filter((u) => u.includes("/auth/refresh"))).toHaveLength(1);
    expect(getToken()).toBeNull();
  });
});