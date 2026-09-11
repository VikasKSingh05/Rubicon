const API_BASE = import.meta.env.VITE_API_URL || "http://localhost:4000";

// The short-lived access token lives in process/JS memory only — never
// localStorage — so an XSS cannot exfiltrate a bearer token. A refresh cookie
// (httpOnly, /auth path on the backend) is what restores the session across
// reloads, via initAuth().
let accessToken = null;

export function getToken() {
  return accessToken;
}

export function setToken(token) {
  accessToken = token || null;
}

async function request(path, options = {}, retried) {
  const headers = { ...(options.headers || {}) };
  if (options.body && !(options.body instanceof FormData) && typeof options.body === "object") {
    headers["Content-Type"] = "application/json";
  }
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;

  const res = await fetch(`${API_BASE}${path}`, {
    ...options,
    headers,
    credentials: "include",
    body:
      options.body && !(options.body instanceof FormData) && typeof options.body === "object"
        ? JSON.stringify(options.body)
        : options.body,
  });

  const data = await res.json().catch(() => null);

  if (res.status === 401 && !retried && !path.startsWith("/auth/")) {
    const refreshed = await refreshAuth();
    if (refreshed) return request(path, options, true);
  }

  if (!res.ok) {
    const err = new Error(data?.error || `request failed (${res.status})`);
    err.status = res.status;
    throw err;
  }
  return data;
}

export async function api(path, options = {}) {
  return request(path, options, false);
}

/**
 * Exchange the httpOnly refresh cookie for a fresh access token + rotated
 * cookie. Returns the session payload ({ token, user }) or null when logged out.
 */
export async function refreshAuth() {
  try {
    const res = await fetch(`${API_BASE}/auth/refresh`, {
      method: "POST",
      credentials: "include",
    });
    if (!res.ok) {
      setToken(null);
      return null;
    }
    const data = await res.json();
    setToken(data.token);
    return data;
  } catch {
    setToken(null);
    return null;
  }
}

/** Restore a session on app boot if a valid refresh cookie exists. */
export async function initAuth() {
  return refreshAuth();
}

export { API_BASE };

/** Fetch a file download (e.g. /files/:id/hsi) and return a Blob URL. */
export async function downloadBlob(path, defaultName) {
  const headers = {};
  if (accessToken) headers.Authorization = `Bearer ${accessToken}`;
  const res = await fetch(`${API_BASE}${path}`, { credentials: "include", headers });
  if (!res.ok) throw new Error("download failed");
  const blob = await res.blob();
  const url = URL.createObjectURL(blob);
  const a = document.createElement("a");
  a.href = url;
  a.download = defaultName;
  a.click();
  URL.revokeObjectURL(url);
}