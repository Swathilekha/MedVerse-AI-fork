import { useState } from "react";
import { loginUser, registerUser } from "./api";

export default function AuthPage({ onAuthSuccess }) {
  const [mode, setMode] = useState("login");
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

  return (
    <main className="auth-shell">
      <section className="auth-card">
        <div className="auth-brand-mark">M</div>
        <p className="auth-kicker">MULTIMODAL CLINICAL PLATFORM</p>
        <h1>Welcome to MedVerse AI</h1>
        <p className="auth-description">Sign in to access your healthcare workspace.</p>
        <div className="auth-tabs">
          <button type="button" className={mode === "login" ? "active" : ""} onClick={() => { setMode("login"); setError(""); }}>Sign in</button>
          <button type="button" className={mode === "register" ? "active" : ""} onClick={() => { setMode("register"); setError(""); }}>Create patient account</button>
        </div>
        <form onSubmit={submit} className="auth-form">
          {mode === "register" && <label>Full name<input value={name} onChange={(e) => setName(e.target.value)} required placeholder="Enter your name" /></label>}
          <label>Email address<input type="email" value={email} onChange={(e) => setEmail(e.target.value)} required placeholder="you@example.com" /></label>
          <label>Password<input type="password" value={password} onChange={(e) => setPassword(e.target.value)} required minLength={6} placeholder="Enter your password" /></label>
          {error && <div className="auth-error">{error}</div>}
          <button className="auth-submit" type="submit" disabled={loading}>{loading ? "Please wait..." : mode === "register" ? "Create account" : "Sign in"}</button>
        </form>
        <p className="auth-footnote">Patient self-registration is available. Doctor accounts are provisioned separately.</p>
      </section>
    </main>
  );
}
