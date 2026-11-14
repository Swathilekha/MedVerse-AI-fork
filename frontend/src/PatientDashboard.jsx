import { useState } from "react";

const sections = ["Overview", "Medical Records", "Upload Reports", "Appointments", "Profile"];

export default function PatientDashboard({ user, onLogout }) {
  const [active, setActive] = useState("Overview");

  const content = () => {
    if (active === "Overview") return (
      <>
        <div className="patient-welcome">
          <div><span className="patient-eyebrow">PATIENT PORTAL</span><h1>Good day, {user.name?.split(" ")[0] || "there"}.</h1><p>Your personal healthcare space, all in one place.</p></div>
          <div className="patient-avatar">{(user.name || "P").charAt(0).toUpperCase()}</div>
        </div>
        <div className="patient-summary-grid">
          <article className="patient-summary-card"><span>Medical records</span><strong>—</strong><small>Records will appear here when connected.</small></article>
          <article className="patient-summary-card"><span>Upcoming appointments</span><strong>—</strong><small>Appointment information is not connected yet.</small></article>
          <article className="patient-summary-card"><span>Reports to review</span><strong>—</strong><small>Doctor-reviewed reports will appear here.</small></article>
        </div>
        <section className="patient-content-card"><h2>Your health workspace</h2><p>Use the sections on the left to view your records and manage your profile. Information will appear once the corresponding patient-record and appointment APIs are connected.</p><div className="patient-quick-actions"><button onClick={() => setActive("Medical Records")}>View medical records</button><button onClick={() => setActive("Upload Reports")}>Upload a report</button><button onClick={() => setActive("Appointments")}>Appointments</button></div></section>
      </>
    );
    if (active === "Medical Records") return <section className="patient-content-card"><span className="patient-eyebrow">MY HEALTH</span><h1>Medical Records</h1><div className="patient-empty-state"><div>▤</div><h3>No records available yet</h3><p>Your uploaded reports and doctor-reviewed records will be listed here once patient record retrieval is connected.</p></div></section>;
    if (active === "Upload Reports") return <section className="patient-content-card"><span className="patient-eyebrow">DOCUMENTS</span><h1>Upload Reports</h1><div className="patient-empty-state"><div>↑</div><h3>Report upload integration is next</h3><p>The patient-facing upload workflow will be connected to the existing report OCR endpoint after its request format is confirmed.</p></div></section>;
    if (active === "Appointments") return <section className="patient-content-card"><span className="patient-eyebrow">CARE SCHEDULE</span><h1>Appointments</h1><div className="patient-empty-state"><div>▦</div><h3>No appointment data connected</h3><p>Appointment booking and history will appear here when the appointment API is available.</p></div></section>;
    return <section className="patient-content-card"><span className="patient-eyebrow">ACCOUNT</span><h1>My Profile</h1><div className="patient-profile-list"><div><span>Name</span><strong>{user.name}</strong></div><div><span>Email</span><strong>{user.email}</strong></div><div><span>Role</span><strong>Patient</strong></div><div><span>Account ID</span><strong>{user.user_id}</strong></div></div></section>;
  };

  return (
    <div className="patient-app">
      <aside className="patient-sidebar">
        <div className="patient-brand"><span className="patient-brand-icon">M</span><div><strong>MedVerse AI</strong><small>Patient Portal</small></div></div>
        <div className="patient-nav-label">WORKSPACE</div>
        <nav>{sections.map((item) => <button type="button" key={item} className={active === item ? "selected" : ""} onClick={() => setActive(item)}>{item}</button>)}</nav>
        <div className="patient-sidebar-bottom"><div className="patient-user-mini"><span>{(user.name || "P").charAt(0).toUpperCase()}</span><div><strong>{user.name}</strong><small>Patient account</small></div></div><button className="patient-logout" onClick={onLogout}>Sign out</button></div>
      </aside>
      <main className="patient-main">
        <header className="patient-topbar"><span>Patient workspace</span><span className="patient-online"><i /> Account active</span></header>
        <div className="patient-page-content">{content()}</div>
        <footer className="patient-footer">MedVerse AI · Patient Portal <span>Research prototype</span></footer>
      </main>
    </div>
  );
}
