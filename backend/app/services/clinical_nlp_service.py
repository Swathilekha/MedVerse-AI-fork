import os
import joblib


# --------------------------------------------------
# Project root
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
)


MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ai-services",
    "clinical",
    "models",
    "best_clinical_nlp_classifier.joblib"
)


# --------------------------------------------------
# Clinical findings used by AARF
# --------------------------------------------------

FINDINGS = [
    "pneumonia",
    "consolidation",
    "lung_opacity",
    "atelectasis",
    "pulmonary_edema",
    "pleural_effusion",
    "pneumothorax",
    "cardiomegaly",
]


# --------------------------------------------------
# Model cache
# --------------------------------------------------

_model = None


def load_model():

    global _model

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Clinical NLP model not found: {MODEL_PATH}"
        )

    print("Loading clinical NLP model...")

    _model = joblib.load(
        MODEL_PATH
    )

    print(
        "Clinical NLP model loaded successfully."
    )

    return _model


# --------------------------------------------------
# Predict assertion for one finding
# --------------------------------------------------

def predict_finding(
    text: str,
    finding: str
):

    model = load_model()

    text = str(text or "").strip()

    if not text:
        return {
            "finding": finding,
            "status": "NO_EVIDENCE",
            "confidence": 0.0,
            "prob_present": 0.0,
            "prob_absent": 0.0,
            "prob_uncertain": 0.0,
        }

    prediction = model.predict(
        [text]
    )[0]

    probabilities = model.predict_proba(
        [text]
    )[0]

    classes = model.classes_

    probability_dict = {
        str(cls): float(prob)
        for cls, prob in zip(
            classes,
            probabilities
        )
    }

    confidence = max(
        probability_dict.values()
    )

    return {
        "finding": finding,
        "status": str(prediction).upper(),
        "confidence": round(
            confidence,
            6
        ),
        "prob_present": round(
            probability_dict.get(
                "present",
                0.0
            ),
            6
        ),
        "prob_absent": round(
            probability_dict.get(
                "absent",
                0.0
            ),
            6
        ),
        "prob_uncertain": round(
            probability_dict.get(
                "uncertain",
                0.0
            ),
            6
        ),
    }


# --------------------------------------------------
# Analyze report
# --------------------------------------------------

def analyze_report(
    text: str
):

    text = str(text or "").strip()

    if not text:

        return {
            "clinical_text": "",
            "findings": [],
            "evidence_count": 0,
        }


    findings = []

    for finding in FINDINGS:

        result = predict_finding(
            text,
            finding
        )

        # Only treat an assertion as actual
        # evidence when the model predicts it.
        if result["status"] in {
            "PRESENT",
            "ABSENT",
            "UNCERTAIN"
        }:

            findings.append(result)


    return {
        "clinical_text": text,
        "findings": findings,
        "evidence_count": len(findings),
        "model": "best_clinical_nlp_classifier.joblib",
    }