import { useState } from "react";
import { loginUser, registerUser } from "./api";
import "./theme.css";
import "./auth.css";
import xrayImg from "./assets/landing/xray.jpg";

export default function AuthPage({ onAuthSuccess, initialMode = "login", onBack }) {
  const [mode, setMode] = useState(initialMode);
  const [name, setName] = useState("");
  const [email, setEmail] = useState("");
  const [password, setPassword] = useState("");
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const submit = async (event) => {
    event.preventDefault();
    setError("");
    setLoading(true);
    try {
      if (mode === "register") {
        await registerUser({ name: name.trim(), email: email.trim(), password });
      }
      const data = await loginUser({ email: email.trim(), password });
      localStorage.setItem("medverse_token", data.access_token);
      onAuthSuccess({
        user_id: data.user_id,
        name: data.name,
        email: data.email,
        role: data.role,
      });
    } catch (err) {
      setError(err.response?.data?.detail || "Unable to complete the request. Please check your details.");
    } finally {
      setLoading(false);
    }
  };

  const switchMode = (next) => {
    setMode(next);
    setError("");
  };

  const isRegister = mode === "register";

  return (
    <main className="mv-auth">
      <aside className="mv-auth-panel">
        <span className="mv-auth-logo">
          <span className="mv-auth-logo-mark" aria-hidden="true">
            <svg viewBox="0 0 24 24" fill="none" stroke="currentColor" strokeWidth="1.8" strokeLinecap="round" strokeLinejoin="round">
              <path d="M3 12h4l2-5 4 10 2-5h6" />
            </svg>
          </span>
          MedVerse AI
        </span>
        <div className="mv-auth-panel-body">
          <div>
            <h2>Every finding, backed by its evidence</h2>
            <p>X-ray, report and history read together, with the reasoning shown alongside each result.</p>
          </div>
          <div className="mv-auth-diamond" aria-hidden="true">
            <div><img src={xrayImg} alt="" /></div>
          </div>
        </div>
        <p className="mv-auth-note">Research prototype · Not a substitute for clinical judgement</p>
      </aside>

      <section className="mv-auth-main">
        {onBack && (
          <button type="button" className="mv-auth-back" onClick={onBack}>
            <span aria-hidden="true">←</span> Back to home
          </button>
        )}
        <div className="mv-auth-card">
          <p className="mv-auth-eyebrow">{isRegister ? "Patient sign up" : "Welcome back"}</p>
          <h1>{isRegister ? "Create your account" : "Sign in to MedVerse AI"}</h1>
          <p className="mv-auth-sub">
            {isRegister
              ? "Keep your reports, X-rays and results in one secure place."
              : "Access your healthcare workspace."}
          </p>
          <div className="mv-auth-tabs" role="group" aria-label="Account action">
            <button type="button" aria-pressed={!isRegister} onClick={() => switchMode("login")}>Sign in</button>
            <button type="button" aria-pressed={isRegister} onClick={() => switchMode("register")}>Create account</button>
          </div>
          <form onSubmit={submit} className="mv-auth-form">
            {isRegister && (
              <label>
                Full name
                <input value={name} onChange={(e) => setName(e.target.value)} required autoComplete="name" placeholder="Enter your name" />
              </label>
            )}
            <label>
              Email address
              <input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required autoComplete="email" placeholder="you@example.com" />
            </label>
            <label>
              Password
              <input
                type="password"
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                required
                minLength={6}
                autoComplete={isRegister ? "new-password" : "current-password"}
                placeholder={isRegister ? "At least 6 characters" : "Enter your password"}
              />
            </label>
            {error && <div className="mv-auth-error" role="alert">{error}</div>}
            <button className="mv-auth-submit" type="submit" disabled={loading}>
              {loading ? "Please wait…" : isRegister ? "Create account" : "Sign in"}
            </button>
          </form>
          <p className="mv-auth-footnote">Patients can sign up here. Doctor accounts are set up by an administrator.</p>
        </div>
      </section>
    </main>
  );
}
