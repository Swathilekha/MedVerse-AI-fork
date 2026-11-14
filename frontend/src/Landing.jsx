import { useState } from "react";
import "./landing.css";
import xrayImg from "./assets/landing/xray.jpg";
import overlayImg from "./assets/landing/overlay-normal.jpg";
import xrayNormalImg from "./assets/landing/xray-normal.jpg";

// Held-out evaluation on the shared 723-record test set
// (ai-services/clinical/outputs/common_723/common_723_comparison.csv).
const MODELS = [
  { name: "X-ray only", kind: "Single modality", accuracy: 68.88, f1: 39.02, auc: 67.86 },
  { name: "History only", kind: "Single modality", accuracy: 89.76, f1: 72.18, auc: 89.56 },
  { name: "Report text only", kind: "Single modality", accuracy: 97.1, f1: 92.25, auc: 99.52 },
  { name: "Joint fusion", kind: "MedVerse model", accuracy: 99.45, f1: 98.52, auc: 99.98, featured: true },
];

const NAV = [
  { href: "#platform", label: "Platform" },
  { href: "#capabilities", label: "Capabilities" },
  { href: "#results", label: "Results" },
  { href: "#workflow", label: "How it works" },
];

function Icon({ name }) {
  const paths = {
    lungs: (
      <>
        <path d="M12 4v7" />
        <path d="M12 11c-1.5 1-3 1.5-3 1.5" />
        <path d="M12 11c1.5 1 3 1.5 3 1.5" />
        <path d="M8.5 7C6 7 4 10.5 4 15c0 2.5 1 4 2.5 4S9 17.5 9 15V8.5C9 7.7 8.9 7 8.5 7Z" />
        <path d="M15.5 7C18 7 20 10.5 20 15c0 2.5-1 4-2.5 4S15 17.5 15 15V8.5c0-.8.1-1.5.5-1.5Z" />
      </>
    ),
    report: (
      <>
        <path d="M7 3h7l4 4v14H7z" />
        <path d="M14 3v4h4" />
        <path d="M10 12h5M10 15h5M10 18h3" />
      </>
    ),
    fusion: (
      <>
        <circle cx="7" cy="7" r="3" />
        <circle cx="17" cy="7" r="3" />
        <circle cx="12" cy="17" r="3" />
        <path d="M9 9l2 5M15 9l-2 5" />
      </>
    ),
    book: (
      <>
        <path d="M4 5a2 2 0 0 1 2-2h13v16H6a2 2 0 0 0-2 2z" />
        <path d="M4 21V5" />
        <path d="M9 8h6M9 11h4" />
      </>
    ),
    check: <path d="M5 12.5l4 4L19 7" />,
    arrow: <path d="M5 12h14M13 6l6 6-6 6" />,
    shield: (
      <>
        <path d="M12 3l7 3v6c0 4.5-3 7.5-7 9-4-1.5-7-4.5-7-9V6z" />
        <path d="M9 12l2 2 4-4" />
      </>
    ),
    upload: (
      <>
        <path d="M12 16V4M7 9l5-5 5 5" />
        <path d="M5 16v3a1 1 0 0 0 1 1h12a1 1 0 0 0 1-1v-3" />
      </>
    ),
    pulse: <path d="M3 12h4l2-5 4 10 2-5h6" />,
    stethoscope: (
      <>
        <path d="M6 3v6a4 4 0 0 0 8 0V3" />
        <path d="M10 13v2a5 5 0 0 0 10 0v-1" />
        <circle cx="20" cy="12" r="2" />
      </>
    ),
    user: (
      <>
        <circle cx="12" cy="8" r="4" />
        <path d="M4 21c1.5-4 4.5-6 8-6s6.5 2 8 6" />
      </>
    ),
    menu: <path d="M4 7h16M4 12h16M4 17h16" />,
    close: <path d="M6 6l12 12M18 6L6 18" />,
  };
  return (
    <svg
      className="lp-icon"
      viewBox="0 0 24 24"
      fill="none"
      stroke="currentColor"
      strokeWidth="1.8"
      strokeLinecap="round"
      strokeLinejoin="round"
      aria-hidden="true"
    >
      {paths[name]}
    </svg>
  );
}

function Logo() {
  return (
    <a className="lp-logo" href="#top" aria-label="MedVerse AI home">
      <span className="lp-logo-mark" aria-hidden="true">
        <Icon name="pulse" />
      </span>
      <span>
        Med<span className="lp-logo-accent">Verse</span> AI
      </span>
    </a>
  );
}

export default function Landing({ onStart }) {
  const [menuOpen, setMenuOpen] = useState(false);

  return (
    <div className="lp" id="top">
      <a className="lp-skip" href="#main">Skip to content</a>

      <div className="lp-strip">
        <div className="lp-container lp-strip-inner">
          <span>
            <Icon name="shield" /> Research prototype · Not a substitute for clinical judgement
          </span>
          <span className="lp-strip-right">Multimodal pneumonia decision support</span>
        </div>
      </div>

      <header className="lp-nav">
        <div className="lp-container lp-nav-inner">
          <Logo />
          <nav aria-label="Primary" className={`lp-nav-links ${menuOpen ? "open" : ""}`}>
            {NAV.map((item) => (
              <a key={item.href} href={item.href} onClick={() => setMenuOpen(false)}>
                {item.label}
              </a>
            ))}
            <button type="button" className="lp-link-btn" onClick={() => onStart("login")}>
              Sign in
            </button>
          </nav>
          <div className="lp-nav-actions">
            <button type="button" className="lp-btn lp-btn-primary lp-btn-sm" onClick={() => onStart("register")}>
              Get started
            </button>
            <button
              type="button"
              className="lp-menu-btn"
              aria-expanded={menuOpen}
              aria-label={menuOpen ? "Close menu" : "Open menu"}
              onClick={() => setMenuOpen((v) => !v)}
            >
              <Icon name={menuOpen ? "close" : "menu"} />
            </button>
          </div>
        </div>
      </header>

      <main id="main">
        {/* HERO */}
        <section className="lp-hero" aria-labelledby="hero-title">
          <div className="lp-hero-blob lp-hero-blob-a" aria-hidden="true" />
          <div className="lp-hero-blob lp-hero-blob-b" aria-hidden="true" />
          <div className="lp-container lp-hero-inner">
            <div className="lp-hero-copy">
              <p className="lp-eyebrow lp-eyebrow-light">AI clinical evidence platform</p>
              <h1 id="hero-title">
                Pneumonia Decisions Backed by Every Piece of Evidence
              </h1>
              <p className="lp-hero-text">
                MedVerse AI reads the chest X-ray, the radiology report and the patient history
                together, then shows clinicians exactly which evidence supports the finding.
              </p>
              <div className="lp-hero-ctas">
                <button type="button" className="lp-btn lp-btn-light" onClick={() => onStart("register")}>
                  Create patient account <Icon name="arrow" />
                </button>
                <button type="button" className="lp-btn lp-btn-ghost" onClick={() => onStart("login")}>
                  Clinician sign in
                </button>
              </div>
              <dl className="lp-hero-stats">
                <div>
                  <dt>Accuracy</dt>
                  <dd>99.45%</dd>
                </div>
                <div>
                  <dt>Recall</dt>
                  <dd>100%</dd>
                </div>
                <div>
                  <dt>Held-out cases</dt>
                  <dd>723</dd>
                </div>
              </dl>
            </div>

            <div className="lp-hero-visual">
              <div className="lp-diamond">
                <div className="lp-diamond-inner">
                  <img src={xrayImg} alt="Frontal chest X-ray analysed by MedVerse AI" />
                </div>
              </div>
              <div className="lp-float lp-float-a">
                <span className="lp-float-dot" aria-hidden="true" />
                <div>
                  <strong>Consolidation</strong>
                  <span>Present · report evidence</span>
                </div>
              </div>
              <div className="lp-float lp-float-b">
                <Icon name="fusion" />
                <div>
                  <strong>Fusion confidence</strong>
                  <span>X-ray + text + history</span>
                </div>
              </div>
            </div>
          </div>
        </section>

        {/* ABOUT */}
        <section className="lp-section" id="platform" aria-labelledby="about-title">
          <div className="lp-container lp-about">
            <div className="lp-about-visual">
              <div className="lp-about-frame">
                <img src={xrayNormalImg} alt="Chest radiograph from the evaluation set" />
                <span className="lp-about-tag">
                  <Icon name="lungs" /> Chest X-ray
                </span>
              </div>
              <div className="lp-about-circle">
                <img src={overlayImg} alt="Grad-CAM overlay highlighting the regions the model focused on" />
              </div>
            </div>
            <div className="lp-about-copy">
              <p className="lp-eyebrow">About the platform</p>
              <h2 id="about-title">
                <span className="lp-accent">One case,</span> three kinds of evidence
              </h2>
              <p>
                A single image rarely tells the whole story. On its own, our X-ray model reaches 68.9%
                accuracy. Fused with the report and history, it reaches 99.45% on the same patients.
              </p>
              <ul className="lp-checklist">
                <li><Icon name="check" /> OCR reads uploaded medical reports and extracts clinical values</li>
                <li><Icon name="check" /> Grad-CAM heatmaps show where the image model looked</li>
                <li><Icon name="check" /> Every finding is linked to the text that supports it</li>
              </ul>
              <a className="lp-btn lp-btn-primary" href="#capabilities">
                Explore capabilities <Icon name="arrow" />
              </a>
            </div>
          </div>
        </section>

        {/* CAPABILITIES */}
        <section className="lp-section lp-section-tint" id="capabilities" aria-labelledby="cap-title">
          <div className="lp-container">
            <div className="lp-section-head">
              <p className="lp-eyebrow">What MedVerse does</p>
              <h2 id="cap-title">Our capabilities</h2>
            </div>
            <div className="lp-cards">
              {[
                { icon: "lungs", tone: "amber", title: "Chest X-ray analysis", text: "CNN classifier for frontal chest radiographs, with Grad-CAM attention maps for every prediction." },
                { icon: "report", tone: "sky", title: "Clinical report NLP", text: "Extracts findings such as consolidation, effusion and opacity as present, absent or uncertain." },
                { icon: "fusion", tone: "lime", title: "Multimodal fusion", text: "Joint and early fusion models combine image, report and history into one calibrated risk score." },
                { icon: "book", tone: "rose", title: "Grounded explanations", text: "A retrieval-augmented assistant answers questions using a curated pneumonia knowledge base." },
              ].map((c) => (
                <article className="lp-card" key={c.title}>
                  <span className={`lp-card-icon tone-${c.tone}`}>
                    <Icon name={c.icon} />
                  </span>
                  <h3>{c.title}</h3>
                  <p>{c.text}</p>
                </article>
              ))}
            </div>
          </div>
        </section>

        {/* RESULTS */}
        <section className="lp-results" id="results" aria-labelledby="results-title">
          <div className="lp-container">
            <div className="lp-section-head lp-section-head-light">
              <p className="lp-eyebrow lp-eyebrow-light">Evaluated, not assumed</p>
              <h2 id="results-title">Fusion beats every single source</h2>
              <p>All four models were scored on the same 723 held-out patient records.</p>
            </div>
            <div className="lp-model-grid">
              {MODELS.map((m) => (
                <article className={`lp-model ${m.featured ? "featured" : ""}`} key={m.name}>
                  <div className="lp-model-top">
                    <span className="lp-model-kind">{m.kind}</span>
                    <h3>{m.name}</h3>
                  </div>
                  <p className="lp-model-score">
                    {m.accuracy.toFixed(2)}<span>%</span>
                  </p>
                  <p className="lp-model-label">Accuracy</p>
                  {[
                    ["F1", m.f1],
                    ["ROC-AUC", m.auc],
                  ].map(([label, v]) => (
                    <div className="lp-meter" key={label}>
                      <div className="lp-meter-row">
                        <span>{label}</span>
                        <span>{v.toFixed(2)}%</span>
                      </div>
                      <div className="lp-meter-track" role="presentation">
                        <div className="lp-meter-fill" style={{ width: `${v}%` }} />
                      </div>
                    </div>
                  ))}
                </article>
              ))}
            </div>
            <p className="lp-results-note">
              Research results on a retrospective dataset. Not clinically validated or approved for
              diagnostic use.
            </p>
          </div>
        </section>

        {/* WORKFLOW */}
        <section className="lp-section" id="workflow" aria-labelledby="flow-title">
          <div className="lp-container">
            <div className="lp-section-head">
              <p className="lp-eyebrow">How it works</p>
              <h2 id="flow-title">From upload to an explained answer</h2>
            </div>
            <ol className="lp-steps">
              {[
                { icon: "upload", title: "Upload the case", text: "Patients or clinicians add a chest X-ray and the radiology report. OCR extracts the text and values." },
                { icon: "fusion", title: "Analyse every source", text: "Image, report and history models each score the case. The fusion layer combines and calibrates them." },
                { icon: "stethoscope", title: "Review with evidence", text: "Doctors see the risk score, the supporting findings and the heatmap, and make the final call." },
              ].map((s, i) => (
                <li className="lp-step" key={s.title}>
                  <div className="lp-step-head">
                    <span className="lp-step-icon"><Icon name={s.icon} /></span>
                    <span className="lp-step-num">0{i + 1}</span>
                  </div>
                  <h3>{s.title}</h3>
                  <p>{s.text}</p>
                </li>
              ))}
            </ol>
          </div>
        </section>

        {/* ROLES */}
        <section className="lp-section lp-section-tint" aria-labelledby="roles-title">
          <div className="lp-container">
            <div className="lp-section-head">
              <p className="lp-eyebrow">Get started</p>
              <h2 id="roles-title">One platform, two workspaces</h2>
            </div>
            <div className="lp-roles">
              <article className="lp-role">
                <span className="lp-role-icon"><Icon name="user" /></span>
                <h3>For patients</h3>
                <p>Create an account, upload your reports and X-rays, and keep your records in one secure place.</p>
                <button type="button" className="lp-btn lp-btn-primary" onClick={() => onStart("register")}>
                  Create patient account <Icon name="arrow" />
                </button>
              </article>
              <article className="lp-role">
                <span className="lp-role-icon"><Icon name="stethoscope" /></span>
                <h3>For clinicians</h3>
                <p>Review fused predictions, clinical evidence and explanations. Doctor accounts are set up by an administrator.</p>
                <button type="button" className="lp-btn lp-btn-outline" onClick={() => onStart("login")}>
                  Clinician sign in
                </button>
              </article>
            </div>
          </div>
        </section>

        {/* CTA */}
        <section className="lp-cta" aria-labelledby="cta-title">
          <div className="lp-container lp-cta-inner">
            <h2 id="cta-title">See the evidence behind every diagnosis</h2>
            <p>Start with a patient account in under a minute.</p>
            <button type="button" className="lp-btn lp-btn-light" onClick={() => onStart("register")}>
              Get started <Icon name="arrow" />
            </button>
          </div>
        </section>
      </main>

      <footer className="lp-footer">
        <div className="lp-container lp-footer-grid">
          <div className="lp-footer-brand">
            <Logo />
            <p>
              Multimodal clinical decision support that combines imaging, reports and history to make AI
              findings explainable.
            </p>
          </div>
          <div>
            <h4>Platform</h4>
            <a href="#platform">Overview</a>
            <a href="#capabilities">Capabilities</a>
            <a href="#results">Results</a>
          </div>
          <div>
            <h4>Workspaces</h4>
            <button type="button" className="lp-link-btn" onClick={() => onStart("register")}>Patient sign up</button>
            <button type="button" className="lp-link-btn" onClick={() => onStart("login")}>Clinician sign in</button>
          </div>
          <div>
            <h4>Models</h4>
            <span>Chest X-ray CNN</span>
            <span>Clinical NLP</span>
            <span>Joint fusion</span>
          </div>
        </div>
        <div className="lp-container lp-footer-bottom">
          <span>© {new Date().getFullYear()} MedVerse AI</span>
          <span>For research and educational use only.</span>
        </div>
      </footer>
    </div>
  );
}
