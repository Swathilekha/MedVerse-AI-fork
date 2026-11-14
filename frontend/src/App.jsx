import { useState } from "react";
import AuthPage from "./AuthPage";
import PatientDashboard from "./PatientDashboard";
import {
  getFusionResult,
  getClinicalEvidence,
  uploadXray,
  askRAG,
} from "./api";
import "./index.css";

function formatPercent(value) {
  if (value === null || value === undefined || Number.isNaN(Number(value))) {
    return "—";
  }

  return `${(Number(value) * 100).toFixed(2)}%`;
}


function StatusBadge({ children, type = "neutral" }) {
  return (
    <span className={`status-badge ${type}`}>
      {children}
    </span>
  );
}


function MetricCard({ label, value, subtext, accent = "" }) {
  return (
    <div className={`metric-card ${accent}`}>
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value}</div>
      {subtext && <div className="metric-subtext">{subtext}</div>}
    </div>
  );
}


function EvidenceRow({ name, state, confidence, support }) {
  let type = "neutral";

  if (state === "PRESENT") type = "positive";
  if (state === "ABSENT") type = "negative";
  if (state === "UNCERTAIN") type = "warning";
  if (state === "MIXED") type = "warning";

  return (
    <div className="evidence-row">
      <div className="evidence-name">
        <span>{name}</span>
        <StatusBadge type={type}>{state}</StatusBadge>
      </div>

      <div className="evidence-bar-container">
        <div
          className="evidence-bar"
          style={{
            width: `${Math.min(Number(confidence || 0) * 100, 100)}%`,
          }}
        />
      </div>

      <div className="evidence-confidence">
        {formatPercent(confidence)}
      </div>

      <div className="evidence-support">
        Support: {formatPercent(support)}
      </div>
    </div>
  );
}


function DoctorDashboard({ user, onLogout }) {
  const [patientId, setPatientId] = useState("patient11586");

  const [fusion, setFusion] = useState(null);
  const [clinicalEvidence, setClinicalEvidence] = useState([]);

  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const [xrayFile, setXrayFile] = useState(null);
  const [xrayResult, setXrayResult] = useState(null);
  const [xrayLoading, setXrayLoading] = useState(false);

  const [ragQuestion, setRagQuestion] = useState("");
  const [ragResult, setRagResult] = useState(null);
  const [ragLoading, setRagLoading] = useState(false);
  const [ragError, setRagError] = useState("");

  const analyzePatient = async () => {
    if (!patientId.trim()) {
      setError("Please enter a patient ID.");
      return;
    }

    setLoading(true);
    setError("");

    try {
      const [fusionResult, evidenceResult] = await Promise.all([
        getFusionResult(patientId.trim()),
        getClinicalEvidence(patientId.trim()).catch(() => []),
      ]);

      setFusion(fusionResult);

      if (Array.isArray(evidenceResult)) {
        setClinicalEvidence(evidenceResult);
      } else {
        setClinicalEvidence([]);
      }

    } catch (err) {
      console.error(err);

      setFusion(null);
      setClinicalEvidence([]);

      if (err.response?.data?.detail) {
        setError(err.response.data.detail);
      } else {
        setError(
          "Unable to connect to the MedVerse AI backend. Make sure FastAPI is running on port 8000."
        );
      }
    } finally {
      setLoading(false);
    }
  };


  const getDecisionType = () => {
    if (!fusion) return "neutral";

    if (fusion.decision === "PNEUMONIA_SUPPORT") {
      return "positive";
    }

    if (fusion.decision === "DEFER_TO_CLINICIAN") {
      return "warning";
    }

    return "neutral";
  };


  const getDecisionText = () => {
    if (!fusion) return "Awaiting Analysis";

    if (fusion.decision === "PNEUMONIA_SUPPORT") {
      return "Pneumonia Support";
    }

    if (fusion.decision === "DEFER_TO_CLINICIAN") {
      return "Defer to Clinician";
    }

    if (fusion.decision === "NO_PNEUMONIA_SUPPORT") {
      return "No Pneumonia Support";
    }

    return fusion.decision || "Unknown";
  };

  const analyzeUploadedXray = async () => {
  if (!xrayFile) {
    setError("Please select an X-ray image first.");
    return;
  }

  if (!patientId.trim()) {
    setError("Please enter a patient ID.");
    return;
  }

  setXrayLoading(true);
  setError("");

  try {
    const result = await uploadXray(
      patientId.trim(),
      xrayFile
    );

    setXrayResult(result);

  } catch (err) {
    console.error(err);

    if (err.response?.data?.detail) {
      setError(err.response.data.detail);
    } else {
      setError(
        "X-ray analysis failed. Make sure the FastAPI backend is running."
      );
    }

  } finally {
    setXrayLoading(false);
  }
};

  const handleRAGQuery = async () => {
  if (!ragQuestion.trim()) {
    setRagError("Please enter a question.");
    return;
  }

  setRagLoading(true);
  setRagError("");
  setRagResult(null);

  try {
    const result = await askRAG(ragQuestion.trim());
    setRagResult(result);
  } catch (err) {
    console.error("RAG Error:", err);

    setRagError(
      err.response?.data?.detail ||
      "Unable to connect to the RAG service. Please try again."
    );
  } finally {
    setRagLoading(false);
  }
};

  return (
    <div className="app">

      {/* HEADER */}
      <header className="topbar">

        <div className="brand">
          <div className="brand-icon">
            M
          </div>

          <div>
            <div className="brand-name">
              MedVerse AI
            </div>

            <div className="brand-subtitle">
              Multimodal Clinical Decision Support
            </div>
          </div>
        </div>

        <div className="header-right">
          <span className="system-status">
            <span className="status-dot" />
            AI System Online
          </span>

          <span className="doctor-badge">
            {user?.name || "Doctor"} · Doctor Dashboard
          </span>
          <button className="logout-button" onClick={onLogout}>Logout</button>
        </div>

      </header>


      {/* MAIN */}
      <main className="dashboard">

        {/* PATIENT SEARCH */}
        <section className="search-section">

          <div>
            <h1>Clinical Analysis</h1>

            <p>
              Analyze multimodal patient evidence using the
              AARF clinical reasoning pipeline.
            </p>
          </div>

          <div className="search-box">

            <input
              type="text"
              value={patientId}
              onChange={(e) => setPatientId(e.target.value)}
              onKeyDown={(e) => {
                if (e.key === "Enter") {
                  analyzePatient();
                }
              }}
              placeholder="Enter Patient ID"
            />

            <button
              onClick={analyzePatient}
              disabled={loading}
            >
              {loading ? "Analyzing..." : "Analyze Patient"}
            </button>

          </div>

        </section>

        {/* LIVE X-RAY UPLOAD */}

<section className="upload-panel">

  <div className="upload-header">

    <div>
      <span className="section-eyebrow">
        LIVE AI INFERENCE
      </span>

      <h3>
        Upload Chest X-Ray
      </h3>

      <p>
        Run the trained DenseNet121 model on a new X-ray.
      </p>
    </div>

  </div>


  <div className="upload-controls">

    <label className="file-input">

      <input
        type="file"
        accept=".png,.jpg,.jpeg,.bmp,.webp"
        onChange={(e) => {
          setXrayFile(
            e.target.files?.[0] || null
          );
        }}
      />

      <span>
        {xrayFile
          ? xrayFile.name
          : "Choose X-ray image"}
      </span>

    </label>


    <button
      className="upload-button"
      onClick={analyzeUploadedXray}
      disabled={xrayLoading}
    >
      {xrayLoading
        ? "Running DenseNet121..."
        : "Analyze X-Ray"}
    </button>

  </div>


  {xrayResult && (

    <div className="live-xray-result">

      <div className="live-result-title">
        LIVE X-RAY ANALYSIS
      </div>


      <div className="live-result-grid">

        <div>
          <span>Prediction</span>

          <strong>
            {xrayResult.analysis.prediction}
          </strong>
        </div>


        <div>
          <span>Pneumonia Probability</span>

          <strong>
            {formatPercent(
              xrayResult.analysis.pneumonia_probability
            )}
          </strong>
        </div>


        <div>
          <span>Normal Probability</span>

          <strong>
            {formatPercent(
              xrayResult.analysis.normal_probability
            )}
          </strong>
        </div>


        <div>
          <span>Confidence</span>

          <strong>
            {formatPercent(
              xrayResult.analysis.confidence
            )}
          </strong>
        </div>


        <div>
          <span>Model</span>

          <strong>
            {xrayResult.analysis.model}
          </strong>
        </div>


        <div>
          <span>Device</span>

          <strong>
            {xrayResult.analysis.device.toUpperCase()}
          </strong>
        </div>

      </div>

    </div>

  )}

</section>
{/* RAG KNOWLEDGE ASSISTANT */}

<section className="panel rag-panel">

  <div className="panel-header">
    <div>
      <span className="section-eyebrow">
        RETRIEVAL AUGMENTED GENERATION
      </span>

      <h3>Clinical Knowledge Assistant</h3>

      <p>
        Ask pneumonia-related questions and retrieve
        answers grounded in curated CDC and IDSA evidence.
      </p>
    </div>

    <StatusBadge type="positive">
      RAG ENABLED
    </StatusBadge>
  </div>

  <div className="rag-input-area">

    <textarea
      value={ragQuestion}
      onChange={(e) => setRagQuestion(e.target.value)}
      placeholder="Ask a clinical knowledge question..."
      rows={3}
    />

    <button
      onClick={handleRAGQuery}
      disabled={ragLoading}
    >
      {ragLoading ? "Retrieving Evidence..." : "Ask Assistant"}
    </button>

  </div>

  {ragError && (
    <div className="error-box">
      {ragError}
    </div>
  )}

  {ragResult && (

    <div className="rag-result">

      <div className="panel-header">

        <h3>Retrieved Answer</h3>

        <StatusBadge
          type={
            ragResult.status === "answered"
              ? "positive"
              : "warning"
          }
        >
          {ragResult.status === "answered"
            ? "Evidence Found"
            : "Abstained"}
        </StatusBadge>

      </div>

      <div className="reasoning-box">
        <p>{ragResult.answer}</p>
      </div>

      {ragResult.sources?.length > 0 && (

        <div className="rag-sources">

          <h4>Supporting Evidence Sources</h4>

          {ragResult.sources.map((source, index) => (

            <div className="trace-item" key={source.id || index}>

              <div className="trace-number">
                {String(index + 1).padStart(2, "0")}
              </div>

              <div>

                <strong>{source.id}</strong>

                <p>
                  Section: {source.section}
                </p>

                <p>
                  Semantic similarity:{" "}
                  {(source.similarity * 100).toFixed(2)}%
                </p>

                <a
                  href={source.url}
                  target="_blank"
                  rel="noopener noreferrer"
                >
                  View Original Source ↗
                </a>

              </div>

            </div>

          ))}

        </div>

      )}

      <p className="rag-disclaimer">
        Research prototype: Retrieved evidence is intended
        for informational support and does not replace
        professional clinical judgment.
      </p>

    </div>

  )}

</section>
        {/* ERROR */}
        {error && (
          <div className="error-box">
            <strong>Analysis Error</strong>
            <span>{error}</span>
          </div>
        )}


        {/* EMPTY STATE */}
        {!fusion && !loading && !error && (
          <section className="empty-state">

            <div className="empty-icon">
              +
            </div>

            <h2>Ready for Clinical Analysis</h2>

            <p>
              Enter a patient ID above to retrieve X-ray,
              clinical evidence, and multimodal fusion results.
            </p>

            <button
              className="demo-button"
              onClick={analyzePatient}
            >
              Analyze patient11586
            </button>

          </section>
        )}


        {/* RESULTS */}
        {fusion && (

          <>

            {/* PATIENT HEADER */}
            <section className="patient-header">

              <div>
                <span className="section-eyebrow">
                  PATIENT RECORD
                </span>

                <h2>{fusion.patient_id}</h2>

                <p>
                  Validation case •{" "}
                  {fusion.final_split?.toUpperCase()}
                </p>
              </div>

              <StatusBadge type={getDecisionType()}>
                {getDecisionText()}
              </StatusBadge>

            </section>


            {/* AARF DECISION */}
            <section className="decision-card">

              <div className="decision-main">

                <div className="section-eyebrow">
                  AARF V3 CLINICAL DECISION SUPPORT
                </div>

                <h2>
                  {getDecisionText()}
                </h2>

                <p>
                  {fusion.evidence_interpretation
                    ?.replaceAll("_", " ")
                    .toLowerCase()
                    .replace(/\b\w/g, (c) => c.toUpperCase())}
                </p>

              </div>


              <div className="decision-score">

                <div className="score-label">
                  Pneumonia Probability
                </div>

                <div className="score-value">
                  {formatPercent(fusion.aarf_v3_probability)}
                </div>

                <div className="score-track">
                  <div
                    className="score-fill"
                    style={{
                      width: `${Number(
                        fusion.aarf_v3_probability
                      ) * 100}%`,
                    }}
                  />
                </div>

              </div>

            </section>


            {/* TOP METRICS */}
            <section className="metrics-grid">

              <MetricCard
                label="X-Ray Probability"
                value={formatPercent(fusion.xray_probability)}
                subtext={`Prediction: ${fusion.prediction}`}
              />

              <MetricCard
                label="Clinical Evidence"
                value={formatPercent(fusion.pneumonia_confidence)}
                subtext={`Pneumonia: ${fusion.pneumonia_state}`}
                accent="green"
              />

              <MetricCard
                label="Evidence Coverage"
                value={formatPercent(fusion.evidence_coverage)}
                subtext={`${fusion.evidence_findings_present} finding detected`}
              />

              <MetricCard
                label="X-Ray Confidence"
                value={formatPercent(fusion.xray_confidence)}
                subtext={`Visual state: ${fusion.visual_state}`}
              />

            </section>


            {/* TWO COLUMN ANALYSIS */}
            <section className="two-column">

              {/* X-RAY */}
              <div className="panel">

                <div className="panel-header">

                  <div>
                    <span className="section-eyebrow">
                      VISION AI
                    </span>

                    <h3>X-Ray Evidence</h3>
                  </div>

                  <StatusBadge type="neutral">
                    DenseNet121
                  </StatusBadge>

                </div>


                <div className="xray-result">

                  <div className="xray-big-number">
                    {formatPercent(fusion.xray_probability)}
                  </div>

                  <div>
                    <div className="result-title">
                      Pneumonia Probability
                    </div>

                    <div className="result-description">
                      Model prediction:{" "}
                      <strong>{fusion.prediction}</strong>
                    </div>
                  </div>

                </div>


                <div className="mini-metrics">

                  <div>
                    <span>Normal Probability</span>
                    <strong>
                      {formatPercent(fusion.normal_probability)}
                    </strong>
                  </div>

                  <div>
                    <span>Pneumonia Probability</span>
                    <strong>
                      {formatPercent(fusion.pneumonia_probability)}
                    </strong>
                  </div>

                  <div>
                    <span>Confidence</span>
                    <strong>
                      {formatPercent(fusion.xray_confidence)}
                    </strong>
                  </div>

                </div>

              </div>


              {/* DIRECT CLINICAL EVIDENCE */}
              <div className="panel">

                <div className="panel-header">

                  <div>
                    <span className="section-eyebrow">
                      NLP / REPORT
                    </span>

                    <h3>Clinical Evidence</h3>
                  </div>

                  <StatusBadge
                    type={
                      fusion.pneumonia_state === "PRESENT"
                        ? "positive"
                        : "neutral"
                    }
                  >
                    {fusion.pneumonia_state}
                  </StatusBadge>

                </div>


                <div className="clinical-highlight">

                  <div className="clinical-icon">
                    +
                  </div>

                  <div>

                    <div className="result-title">
                      Pneumonia
                    </div>

                    <div className="result-description">
                      Direct report evidence detected
                    </div>

                  </div>

                  <strong>
                    {formatPercent(
                      fusion.pneumonia_confidence
                    )}
                  </strong>

                </div>


                <div className="mini-metrics">

                  <div>
                    <span>Evidence Count</span>
                    <strong>
                      {fusion.pneumonia_count}
                    </strong>
                  </div>

                  <div>
                    <span>Support Score</span>
                    <strong>
                      {formatPercent(
                        fusion.pneumonia_support
                      )}
                    </strong>
                  </div>

                  <div>
                    <span>Uncertainty</span>
                    <strong>
                      {formatPercent(
                        fusion.text_uncertainty
                      )}
                    </strong>
                  </div>

                </div>

              </div>

            </section>


            {/* FINDINGS */}
            <section className="panel">

              <div className="panel-header">

                <div>
                  <span className="section-eyebrow">
                    MULTIMODAL EVIDENCE
                  </span>

                  <h3>Clinical Finding Profile</h3>
                </div>

                <span className="coverage-label">
                  {formatPercent(fusion.evidence_coverage)} coverage
                </span>

              </div>


              <div className="evidence-list">

                <EvidenceRow
                  name="Pneumonia"
                  state={fusion.pneumonia_state}
                  confidence={fusion.pneumonia_confidence}
                  support={fusion.pneumonia_support}
                />

                <EvidenceRow
                  name="Consolidation"
                  state={fusion.consolidation_state}
                  confidence={fusion.consolidation_confidence}
                  support={fusion.consolidation_support}
                />

                <EvidenceRow
                  name="Lung Opacity"
                  state={fusion.lung_opacity_state}
                  confidence={fusion.lung_opacity_confidence}
                  support={fusion.lung_opacity_support}
                />

                <EvidenceRow
                  name="Atelectasis"
                  state={fusion.atelectasis_state}
                  confidence={fusion.atelectasis_confidence}
                  support={fusion.atelectasis_support}
                />

                <EvidenceRow
                  name="Pleural Effusion"
                  state={fusion.pleural_effusion_state}
                  confidence={fusion.pleural_effusion_confidence}
                  support={fusion.pleural_effusion_support}
                />

                <EvidenceRow
                  name="Pneumothorax"
                  state={fusion.pneumothorax_state}
                  confidence={fusion.pneumothorax_confidence}
                  support={fusion.pneumothorax_support}
                />

              </div>

            </section>


            {/* CROSS MODAL */}
            <section className="panel">

              <div className="panel-header">

                <div>
                  <span className="section-eyebrow">
                    CROSS-MODAL REASONING
                  </span>

                  <h3>Evidence Consistency Analysis</h3>
                </div>

              </div>


              <div className="cross-grid">

                <div className="cross-item">

                  <span>
                    Semantic Conflict
                  </span>

                  <strong>
                    {fusion.semantic_conflict}
                  </strong>

                </div>


                <div className="cross-item">

                  <span>
                    Numerical Discordance
                  </span>

                  <strong>
                    {formatPercent(
                      fusion.numerical_discordance
                    )}
                  </strong>

                </div>


                <div className="cross-item">

                  <span>
                    Visual State
                  </span>

                  <strong>
                    {fusion.visual_state}
                  </strong>

                </div>


                <div className="cross-item">

                  <span>
                    Text Uncertainty
                  </span>

                  <strong>
                    {formatPercent(
                      fusion.text_uncertainty
                    )}
                  </strong>

                </div>

              </div>


              <div className="reasoning-box">

                <div className="reasoning-title">
                  Evidence Interpretation
                </div>

                <p>
                  {fusion.evidence_interpretation
                    ?.replaceAll("_", " ")
                    .toLowerCase()
                    .replace(/\b\w/g, (c) => c.toUpperCase())}
                </p>

                <div className="reasoning-flow">

                  <span>
                    X-Ray
                  </span>

                  <span className="arrow">
                    →
                  </span>

                  <span>
                    Clinical NLP
                  </span>

                  <span className="arrow">
                    →
                  </span>

                  <span>
                    AARF Fusion
                  </span>

                  <span className="arrow">
                    →
                  </span>

                  <span className="final-node">
                    Decision
                  </span>

                </div>

              </div>

            </section>


            {/* EVIDENCE TRACE */}
            <section className="panel">

              <div className="panel-header">

                <div>
                  <span className="section-eyebrow">
                    EXPLAINABILITY
                  </span>

                  <h3>Clinical Evidence Trace</h3>
                </div>

              </div>


              <div className="trace">

                <div className="trace-item">

                  <div className="trace-number">
                    01
                  </div>

                  <div>
                    <strong>
                      X-Ray Vision Model
                    </strong>

                    <p>
                      Pneumonia probability:{" "}
                      {formatPercent(
                        fusion.xray_probability
                      )}
                      . Model prediction:{" "}
                      {fusion.prediction}.
                    </p>
                  </div>

                </div>


                <div className="trace-item">

                  <div className="trace-number">
                    02
                  </div>

                  <div>
                    <strong>
                      Clinical NLP Evidence
                    </strong>

                    <p>
                      Pneumonia assertion:{" "}
                      {fusion.pneumonia_state}
                      {" "}with{" "}
                      {formatPercent(
                        fusion.pneumonia_confidence
                      )}
                      {" "}confidence.
                    </p>
                  </div>

                </div>


                <div className="trace-item">

                  <div className="trace-number">
                    03
                  </div>

                  <div>
                    <strong>
                      Multimodal Fusion
                    </strong>

                    <p>
                      AARF v3 probability:{" "}
                      {formatPercent(
                        fusion.aarf_v3_probability
                      )}
                      .
                    </p>
                  </div>

                </div>


                <div className="trace-item">

                  <div className="trace-number">
                    04
                  </div>

                  <div>
                    <strong>
                      Final Clinical Decision Support
                    </strong>

                    <p>
                      Decision:{" "}
                      <strong>
                        {getDecisionText()}
                      </strong>
                    </p>
                  </div>

                </div>

              </div>

            </section>


            {/* CLINICIAN WARNING */}
            {fusion.decision === "DEFER_TO_CLINICIAN" && (

              <section className="warning-panel">

                <div className="warning-icon">
                  !
                </div>

                <div>

                  <strong>
                    Clinician Review Required
                  </strong>

                  <p>
                    The fusion model has deferred this case
                    because of insufficient confidence.
                    AI output should be reviewed by a qualified
                    clinician.
                  </p>

                </div>

              </section>

            )}


            {/* FOOTER */}
            <footer className="dashboard-footer">

              <span>
                MedVerse AI • Multimodal Clinical Decision Support
              </span>

              <span>
                AARF v3 • Research Prototype
              </span>

            </footer>

          </>

        )}

      </main>

    </div>
  );
}


function App() {
  const [user, setUser] = useState(() => {
    try {
      return JSON.parse(localStorage.getItem("medverse_user") || "null");
    } catch {
      return null;
    }
  });

  const handleAuthSuccess = (userData) => {
    localStorage.setItem("medverse_user", JSON.stringify(userData));
    setUser(userData);
  };

  const handleLogout = () => {
    localStorage.removeItem("medverse_user");
    localStorage.removeItem("medverse_token");
    setUser(null);
  };

  if (!user) return <AuthPage onAuthSuccess={handleAuthSuccess} />;

  if (user.role === "patient") {
    return <PatientDashboard user={user} onLogout={handleLogout} />;
  }

  if (user.role === "doctor") {
    return <DoctorDashboard user={user} onLogout={handleLogout} />;
  }

  return (
    <div className="auth-shell">
      <div className="auth-card">
        <h2>Account role not supported</h2>
        <p>Please contact the administrator to configure your dashboard access.</p>
        <button onClick={handleLogout}>Sign out</button>
      </div>
    </div>
  );
}

export default App;