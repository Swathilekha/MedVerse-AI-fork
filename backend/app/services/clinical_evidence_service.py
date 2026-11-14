import os
import re
import joblib


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
)

MODEL_FILE = os.path.join(
    PROJECT_ROOT,
    "ai-services",
    "clinical",
    "models",
    "best_clinical_nlp_classifier.joblib"
)


# ============================================================
# FINDINGS
# ============================================================

FINDING_PATTERNS = {

    "pneumonia": [
        r"\bpneumonia\b",
        r"\bpneumonic\b"
    ],

    "atelectasis": [
        r"\batelectasis\b",
        r"\batelectatic\b"
    ],

    "consolidation": [
        r"\bconsolidation\b",
        r"\bconsolidative\b"
    ],

    "pleural_effusion": [
        r"\bpleural\s+effusion\b",
        r"\bpleural\s+effusions\b"
    ],

    "pneumothorax": [
        r"\bpneumothorax\b",
        r"\bpneumothoraces\b"
    ],

    "pulmonary_edema": [
        r"\bpulmonary\s+edema\b",
        r"\bpulmonary\s+oedema\b"
    ],

    "cardiomegaly": [
        r"\bcardiomegaly\b",
        r"\benlarged\s+heart\b",
        r"\benlarged\s+cardiac\s+silhouette\b"
    ],

    "lung_opacity": [
        r"\blung\s+opacity\b",
        r"\blung\s+opacities\b",
        r"\bpulmonary\s+opacity\b",
        r"\bpulmonary\s+opacities\b",
        r"\bairspace\s+opacity\b",
        r"\bairspace\s+opacities\b",
        r"\bparenchymal\s+opacity\b",
        r"\bparenchymal\s+opacities\b",
        r"\bstreaky\s+opacities\b",
        r"\bpatchy\s+opacities\b",
        r"\blinear\s+opacities\b",
        r"\bretrocardiac\s+opacity\b",
        r"\bretrocardiac\s+opacities\b",
        r"\bopacification\b"
    ]
}


# ============================================================
# NEGATION
# ============================================================

NEGATION_PATTERNS = [

    r"\bno\s+evidence\s+of\b",
    r"\bno\s+definite\b",
    r"\bno\s+focal\b",
    r"\bno\s+acute\b",
    r"\bno\b",
    r"\bwithout\b",
    r"\babsence\s+of\b",
    r"\babsent\b",
    r"\bnegative\s+for\b",
    r"\bfree\s+of\b"
]


# ============================================================
# UNCERTAINTY
# ============================================================

DIRECT_UNCERTAINTY_PATTERNS = [

    r"\bmay\s+represent\b",
    r"\bmay\s+be\b",
    r"\bmay\s+reflect\b",
    r"\bmay\s+correspond\s+to\b",

    r"\bpossibly\b",
    r"\bpossible\b",

    r"\bcould\s+represent\b",
    r"\bcould\s+be\b",

    r"\bcannot\s+exclude\b",

    r"\bquestion\s+of\b",
    r"\bquestionable\b",

    r"\bsuggestive\s+of\b",
    r"\bconcerning\s+for\b",

    r"\blikely\b",
    r"\bprobably\b",
    r"\bprobable\b",

    r"\bsuspicious\s+for\b"
]


# ============================================================
# MODEL
# ============================================================

_model = None


def load_model():

    global _model

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_FILE):
        raise FileNotFoundError(
            f"NLP model not found: {MODEL_FILE}"
        )

    print("Loading clinical NLP model...")

    _model = joblib.load(
        MODEL_FILE
    )

    print("Clinical NLP model loaded.")

    return _model


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(text):

    if text is None:
        return ""

    text = str(text)

    text = text.replace(
        "\n",
        " "
    )

    text = text.replace(
        "\r",
        " "
    )

    # Same report-format fixes as the research pipeline.
    text = re.sub(
        r"pleural\s*effusions",
        "pleural effusions",
        text,
        flags=re.IGNORECASE
    )

    text = re.sub(
        r"pleuraleffusion",
        "pleural effusion",
        text,
        flags=re.IGNORECASE
    )

    # Numbered impression items.
    text = re.sub(
        r"(\d+)\.",
        r" \1. ",
        text
    )

    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


def normalize(text):

    return clean_text(text).lower()


# ============================================================
# CLAUSE SPLITTING
# ============================================================

def split_into_clauses(text):

    text = clean_text(text)

    if not text:
        return []

    text = re.sub(
        r"\s+\d+\.\s*",
        ". ",
        text
    )

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z])",
        text
    )

    return [
        part.strip()
        for part in parts
        if part.strip()
    ]


# ============================================================
# FINDING SEARCH
# ============================================================

def find_occurrences(
    text,
    finding
):

    lower = normalize(text)

    occurrences = []

    for pattern in FINDING_PATTERNS[finding]:

        for match in re.finditer(
            pattern,
            lower
        ):

            occurrences.append(
                (
                    match.start(),
                    match.end(),
                    match.group()
                )
            )

    occurrences.sort(
        key=lambda x: x[0]
    )

    # Remove overlapping matches.
    cleaned = []

    for occurrence in occurrences:

        if not cleaned:
            cleaned.append(occurrence)
            continue

        previous = cleaned[-1]

        if occurrence[0] >= previous[1]:
            cleaned.append(occurrence)

    return cleaned


# ============================================================
# LOCAL CONTEXT
# ============================================================

def get_local_context(
    text,
    start,
    end,
    window=90
):

    left = max(
        0,
        start - window
    )

    right = min(
        len(text),
        end + window
    )

    return text[left:right]


# ============================================================
# NEGATION
# ============================================================

def detect_negation(
    text,
    start,
    end
):

    lower = normalize(text)

    before = lower[
        max(0, start - 100):start
    ]

    # Don't carry negation across unrelated sentences.
    sentence_break = max(
        before.rfind("."),
        before.rfind(";")
    )

    if sentence_break >= 0:

        before = before[
            sentence_break + 1:
        ]

    for pattern in NEGATION_PATTERNS:

        matches = list(
            re.finditer(
                pattern,
                before
            )
        )

        if not matches:
            continue

        last = matches[-1]

        words_after = before[
            last.end():
        ].split()

        if len(words_after) <= 8:
            return True

    # Coordinated negation:
    # "No pleural effusion or pneumothorax"
    last_no = before.rfind("no ")

    if last_no >= 0:

        no_part = before[last_no:]

        if len(no_part.split()) <= 10:
            return True

    return False


# ============================================================
# UNCERTAINTY
# ============================================================

def detect_uncertainty(
    text,
    start,
    end
):

    lower = normalize(text)

    context = get_local_context(
        lower,
        start,
        end,
        window=45
    )

    # Direct uncertainty.
    for pattern in DIRECT_UNCERTAINTY_PATTERNS:

        if re.search(
            pattern,
            context
        ):

            return True, "uncertainty"

    # Differential:
    # atelectasis versus consolidation
    versus_matches = list(
        re.finditer(
            r"\b(?:versus|v[./]\s*s[.]?)\b",
            lower
        )
    )

    for versus in versus_matches:

        finding_before = (
            0 <= versus.start() - end <= 25
        )

        finding_after = (
            0 <= start - versus.end() <= 25
        )

        if finding_before or finding_after:

            return True, "differential"

    return False, None


# ============================================================
# ASSERTION
# ============================================================

def determine_assertion(
    text,
    start,
    end
):

    # Same priority as your research pipeline:
    # 1. negation
    # 2. uncertainty
    # 3. differential
    # 4. present

    if detect_negation(
        text,
        start,
        end
    ):

        return (
            "ABSENT",
            "rule_negation"
        )

    uncertain, reason = detect_uncertainty(
        text,
        start,
        end
    )

    if uncertain:

        if reason == "differential":

            return (
                "UNCERTAIN",
                "rule_differential"
            )

        return (
            "UNCERTAIN",
            "rule_uncertainty"
        )

    return (
        "PRESENT",
        "rule_positive"
    )


# ============================================================
# MODEL PREDICTION
# ============================================================

def get_model_prediction(
    model,
    text
):

    try:

        probabilities = model.predict_proba(
            [text]
        )[0]

        classes = list(
            model.classes_
        )

        probability_map = {
            str(cls): float(prob)
            for cls, prob in zip(
                classes,
                probabilities
            )
        }

        predicted = max(
            probability_map,
            key=probability_map.get
        )

        confidence = probability_map[
            predicted
        ]

        return (
            predicted,
            confidence
        )

    except Exception:

        return (
            None,
            None
        )


# ============================================================
# FINDING-LEVEL ANALYSIS
# ============================================================

def analyze_report_text(text):

    text = clean_text(text)

    if not text:

        return {
            "clinical_text": "",
            "findings": [],
            "evidence_count": 0,
            "model": os.path.basename(
                MODEL_FILE
            )
        }

    model = load_model()

    clauses = split_into_clauses(
        text
    )

    evidence = []

    for clause in clauses:

        model_status, model_confidence = (
            get_model_prediction(
                model,
                clause
            )
        )

        for finding in FINDING_PATTERNS:

            occurrences = find_occurrences(
                clause,
                finding
            )

            if not occurrences:
                continue

            for start, end, matched_text in occurrences:

                status, assertion_source = (
                    determine_assertion(
                        clause,
                        start,
                        end
                    )
                )

                context = get_local_context(
                    clause,
                    start,
                    end,
                    window=120
                )

                # Same confidence semantics as
                # the research evidence generator.
                if assertion_source == "rule_negation":

                    extraction_confidence = 0.95

                elif assertion_source in (
                    "rule_uncertainty",
                    "rule_differential"
                ):

                    extraction_confidence = 0.90

                else:

                    extraction_confidence = (
                        model_confidence
                        if model_confidence is not None
                        else 0.50
                    )

                evidence.append({

                    "finding": finding,

                    "status": status,

                    "evidence_text": clause,

                    "matched_text": matched_text,

                    "model_predicted_status": (
                        str(model_status).upper()
                        if model_status
                        else ""
                    ),

                    "model_confidence": (
                        round(
                            model_confidence,
                            4
                        )
                        if model_confidence is not None
                        else None
                    ),

                    "extraction_confidence": round(
                        extraction_confidence,
                        4
                    ),

                    "assertion_source":
                        assertion_source,

                    "context": context
                })

    # Same duplicate concept as research pipeline.
    unique = []
    seen = set()

    for item in evidence:

        key = (
            item["finding"],
            item["status"],
            item["evidence_text"]
        )

        if key in seen:
            continue

        seen.add(key)
        unique.append(item)

    return {
        "clinical_text": text,
        "findings": unique,
        "evidence_count": len(unique),
        "model": os.path.basename(
            MODEL_FILE
        )
    }