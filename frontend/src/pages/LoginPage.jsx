import React, { useState } from "react";
import { useNavigate } from "react-router-dom";
import { useAuth } from "../context/AuthContext.jsx";
import AuthCard from "../components/AuthCard.jsx";
import FormInput from "../components/FormInput.jsx";
import SubmitButton from "../components/SubmitButton.jsx";

export default function LoginPage() {
  const { login, register, error } = useAuth();
  const navigate = useNavigate();
  const [mode, setMode] = useState("login");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);

  async function handleSubmit(e) {
    e.preventDefault();
    setLoading(true);
    const submit = mode === "login" ? login : register;
    const { ok } = await submit(email, password);
    setLoading(false);
    if (ok) navigate("/", { replace: true });
  }

  return (
    <AuthCard>
      <div className="mb-6 flex rounded-md bg-slate-100 p-1">
        <button
          onClick={() => setMode("login")}
          className={`flex-1 rounded-md py-1.5 text-sm font-semibold transition-colors ${
            mode === "login" ? "bg-white text-primary shadow-sm" : "text-slate-500"
          }`}
        >
          Login
        </button>
        <button
          onClick={() => setMode("register")}
          className={`flex-1 rounded-md py-1.5 text-sm font-semibold transition-colors ${
            mode === "register" ? "bg-white text-primary shadow-sm" : "text-slate-500"
          }`}
        >
          Register
        </button>
      </div>

      <form onSubmit={handleSubmit}>
        <FormInput
          label="Email"
          id="email"
          type="email"
          autoComplete="email"
          value={email}
          onChange={(e) => setEmail(e.target.value)}
          required
        />
        <FormInput
          label="Password"
          id="password"
          type="password"
          autoComplete="current-password"
          value={password}
          onChange={(e) => setPassword(e.target.value)}
          required
        />
        {error && (
          <p className="mb-3 text-sm text-severity-severe" role="alert">
            {error}
          </p>
        )}
        <SubmitButton loading={loading}>
          {mode === "login" ? "Sign in" : "Create account"}
        </SubmitButton>
      </form>

      <p className="mt-4 text-center text-xs text-slate-400">
        Demo build — tokens are stored in localStorage for simplicity (tradeoff noted
        in the report).
      </p>
    </AuthCard>
  );
}