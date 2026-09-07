import React, { createContext, useContext, useMemo, useState } from "react";
import { api, setToken, getToken } from "../lib/api.js";

const AuthContext = createContext(null);

export function AuthProvider({ children }) {
  const [token, setTokenState] = useState(() => getToken());
  const [user, setUser] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("rubicon_user") || "null");
    } catch {
      return null;
    }
  });
  const [error, setError] = useState(null);

  const persist = (data) => {
    setToken(data.token);
    setTokenState(data.token);
    localStorage.setItem("rubicon_user", JSON.stringify(data.user));
    setUser(data.user);
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

  const logout = () => {
    setToken(null);
    setTokenState(null);
    localStorage.removeItem("rubicon_user");
    setUser(null);
  };

  const value = useMemo(
    () => ({ token, user, error, login, register, logout }),
    [token, user, error],
  );

  return <AuthContext.Provider value={value}>{children}</AuthContext.Provider>;
}

export function useAuth() {
  return useContext(AuthContext);
}