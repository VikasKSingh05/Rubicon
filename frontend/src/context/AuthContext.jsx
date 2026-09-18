import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { api, setToken, getToken, initAuth } from "../lib/api.js";

const AuthContext = createContext(null);

const DEMO_TOKEN = "demo-token";
const DEMO_USER = { id: "demo", email: "demo@rubicon.local", role: "admin" };

function readStoredUser() {
  try {
    return JSON.parse(localStorage.getItem("rubicon_user") || "null");
  } catch {
    return null;
  }
}

function storeUser(user) {
  if (user) localStorage.setItem("rubicon_user", JSON.stringify(user));
  else localStorage.removeItem("rubicon_user");
}

export function AuthProvider({ children }) {
  const [token, setTokenState] = useState(() => getToken());
  const [user, setUser] = useState(readStoredUser);
  const [ready, setReady] = useState(false);
  const [error, setError] = useState(null);
  const [demoMode, setDemoMode] = useState(false);

  // On boot, try to restore the session from the httpOnly refresh cookie.
  // If the backend is unreachable, auto-enter demo mode.
  useEffect(() => {
    let cancelled = false;
    initAuth()
      .then((data) => {
        if (cancelled) return;
        if (data && data.token) {
          setTokenState(data.token);
          if (data.user) {
            setUser(data.user);
            storeUser(data.user);
          }
        }
      })
      .catch(() => {
        // Backend unreachable — enable demo mode automatically
        if (!cancelled) {
          setDemoMode(true);
          setToken(DEMO_TOKEN);
          setTokenState(DEMO_TOKEN);
          setUser(DEMO_USER);
          storeUser(DEMO_USER);
        }
      })
      .finally(() => {
        if (!cancelled) setReady(true);
      });
    return () => {
      cancelled = true;
    };
  }, []);

  const persist = (data) => {
    setToken(data.token);
    setTokenState(data.token);
    setUser(data.user);
    storeUser(data.user);
    setError(null);
  };

  const enterDemo = () => {
    setDemoMode(true);
    setToken(DEMO_TOKEN);
    setTokenState(DEMO_TOKEN);
    setUser(DEMO_USER);
    storeUser(DEMO_USER);
    setError(null);
    return { ok: true };
  };

  const login = async (email, password) => {
    try {
      const data = await api("/auth/login", {
        method: "POST",
        body: { email, password },
      });
      persist(data);
      return { ok: true };
    } catch (err) {
      setError(err.message);
      return { ok: false };
    }
  };

  const register = async (email, password) => {
    try {
      const data = await api("/auth/register", {
        method: "POST",
        body: { email, password },
      });
      persist(data);
      return { ok: true };
    } catch (err) {
      setError(err.message);
      return { ok: false };
    }
  };

  const logout = async () => {
    if (!demoMode) {
      try {
        await api("/auth/logout", { method: "POST" });
      } catch {
        // revoke locally even if the server is unreachable
      }
    }
    setDemoMode(false);
    setToken(null);
    setTokenState(null);
    setUser(null);
    storeUser(null);
  };

  const value = useMemo(
    () => ({ token, user, ready, error, demoMode, login, register, logout, enterDemo }),
    [token, user, ready, error, demoMode],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}