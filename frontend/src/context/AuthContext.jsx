import React, { createContext, useContext, useEffect, useMemo, useState } from "react";
import { api, setToken, getToken, initAuth } from "../lib/api.js";

const AuthContext = createContext(null);

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

  // On boot, try to restore the session from the httpOnly refresh cookie.
  useEffect(() => {
    let cancelled = false;
    initAuth().then((data) => {
      if (cancelled) return;
      if (data && data.token) {
        setTokenState(data.token);
        if (data.user) {
          setUser(data.user);
          storeUser(data.user);
        }
      }
    }).finally(() => {
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
    try {
      await api("/auth/logout", { method: "POST" });
    } catch {
      // revoke locally even if the server is unreachable
    }
    setToken(null);
    setTokenState(null);
    setUser(null);
    storeUser(null);
  };

  const value = useMemo(
    () => ({ token, user, ready, error, login, register, logout }),
    [token, user, ready, error],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}