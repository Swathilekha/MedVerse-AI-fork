import os
import re
import pandas as pd
import joblib


# ============================================================
# CONFIG
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

INPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "clinical_text",
    "multimodal_dataset.csv"
)

MODEL_FILE = os.path.join(
    BASE_DIR,
    "ai-services",
    "clinical",
    "models",
    "best_clinical_nlp_classifier.joblib"
)

OUTPUT_FILE = os.path.join(
    BASE_DIR,
    "data",
    "clinical_text",
    "ai_evidence_final.csv"
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
        r"\bpleural effusion\b",
        r"\bpleural effusions\b",
        r"\beffusion\b",
        r"\beffusions\b"
    ],

    "pneumothorax": [
        r"\bpneumothorax\b",
        r"\bpneumothoraces\b"
    ],

    "pulmonary_edema": [
        r"\bpulmonary edema\b",
        r"\bpulmonary oedema\b",
        r"\bedema\b",
        r"\boedema\b"
    ],

    "cardiomegaly": [
        r"\bcardiomegaly\b",
        r"\bcardiomegal(y|ic)\b",
        r"\benlarged heart\b",
        r"\benlarged cardiac silhouette\b"
    ],

    "lung_opacity": [
        r"\blung opacity\b",
        r"\blung opacities\b",
        r"\bpulmonary opacity\b",
        r"\bpulmonary opacities\b",
        r"\bairspace opacity\b",
        r"\bairspace opacities\b",
        r"\bparenchymal opacity\b",
        r"\bparenchymal opacities\b",
        r"\bstreaky opacities\b",
        r"\bpatchy opacities\b",
        r"\blinear opacities\b"
    ]
}


# ============================================================
# ASSERTION PATTERNS
# ============================================================

# Strong negation patterns.
# These are intentionally local rather than checking the whole
# sentence for the word "no".

NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bwithout\b",
    r"\bnegative for\b",
    r"\bno evidence of\b",
    r"\bno definite\b",
    r"\bno focal\b",
    r"\bno acute\b",
    r"\babsence of\b",
    r"\babsent\b",
    r"\bfree of\b"
]


# Uncertainty / possibility / differential language.
UNCERTAINTY_PATTERNS = [
    r"\bmay represent\b",
    r"\bmay be\b",
    r"\bmay reflect\b",
    r"\bpossibly\b",
    r"\bpossible\b",
    r"\bquestion of\b",
    r"\bquestionable\b",
    r"\bcannot exclude\b",
    r"\bcould represent\b",
    r"\bcould be\b",
    r"\bsuggestive of\b",
    r"\bconcerning for\b",
    r"\bconsistent with\b",
    r"\blikely\b",
    r"\bprobably\b",
    r"\bversus\b",
    r"\bv[./ ]?s\.?\b",
    r"\bversus\b",
    r"\bindeterminate\b",
    r"\bsuspicious for\b"
]


# ============================================================
# SECTION HANDLING
# ============================================================

SECTION_COLUMNS = [
    ("clinical_history", "section_clinical_history"),
    ("findings", "section_findings"),
    ("impression", "section_impression")
]


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def clean_text(value):
    if pd.isna(value):
        return ""

    text = str(value).strip()

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text


def normalize_text(text):
    text = clean_text(text)

    return text.lower()


def sentence_split(text):
    """
    Conservative sentence splitter for radiology reports.
    Keeps medical abbreviations reasonably intact.
    """

    text = clean_text(text)

    if not text:
        return []

    # Normalize numbered items
    text = re.sub(r"\s+\d+\.\s+", ". ", text)

    # Split after punctuation followed by uppercase/lowercase text.
    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Za-z])",
        text
    )

    return [
        p.strip()
        for p in parts
        if p.strip()
    ]


def contains_finding(text, finding):
    """
    Check whether a finding occurs in text.
    """

    text_lower = normalize_text(text)

    for pattern in FINDING_PATTERNS[finding]:
        if re.search(pattern, text_lower):
            return True

    return False


def finding_match_positions(text, finding):
    """
    Return all character spans where the finding occurs.
    """

    text_lower = normalize_text(text)

    matches = []

    for pattern in FINDING_PATTERNS[finding]:
        for match in re.finditer(pattern, text_lower):
            matches.append(match.span())

    # Remove duplicate/overlapping spans
    matches = sorted(matches)

    filtered = []

    for span in matches:
        if not filtered:
            filtered.append(span)
            continue

        previous = filtered[-1]

        if span[0] >= previous[1]:
            filtered.append(span)

    return filtered


def local_context(text, start, end, window=100):
    """
    Extract context around one specific finding.

    This is important because:

        NO PNEUMOTHORAX OR PLEURAL EFFUSION

    should not cause a global sentence-level assertion error.
    """

    start_context = max(0, start - window)
    end_context = min(len(text), end + window)

    return text[start_context:end_context]


def has_local_negation(context):
    """
    Detect negation immediately before/around the finding.

    We intentionally avoid treating every "no" anywhere in the
    sentence as negation.
    """

    text = normalize_text(context)

    # Direct patterns
    for pattern in NEGATION_PATTERNS:

        # Search only within the preceding region.
        matches = list(re.finditer(pattern, text))

        if matches:
            # If a negation exists, ensure it is reasonably close
            # to the finding.
            last_match = matches[-1]

            after = text[last_match.end():]

            # Very short local distance = strong evidence of negation
            if len(after.split()) <= 10:
                return True

    return False


def has_uncertainty(context):
    """
    Detect uncertainty language.
    """

    text = normalize_text(context)

    for pattern in UNCERTAINTY_PATTERNS:
        if re.search(pattern, text):
            return True

    return False


def detect_assertion(text, start, end):
    """
    Determine assertion for a specific finding occurrence.

    Priority:

        NEGATED > UNCERTAIN > PRESENT

    This prevents "no pneumonia" from becoming uncertain just
    because another uncertainty phrase occurs later.
    """

    context = local_context(text, start, end)

    # --------------------------------------------------------
    # NEGATION
    # --------------------------------------------------------

    if has_local_negation(context):
        return "absent", "rule_negation"

    # --------------------------------------------------------
    # UNCERTAINTY
    # --------------------------------------------------------

    if has_uncertainty(context):
        return "uncertain", "rule_uncertainty"

    # --------------------------------------------------------
    # DEFAULT
    # --------------------------------------------------------

    return "present", "rule_positive"


def extract_location(text, start, end):
    """
    Extract a simple anatomical location from nearby context.
    """

    context = normalize_text(
        local_context(text, start, end, window=80)
    )

    locations = [
        "left lower lobe",
        "right lower lobe",
        "left upper lobe",
        "right upper lobe",
        "left middle lobe",
        "right middle lobe",
        "left lung base",
        "right lung base",
        "left base",
        "right base",
        "bibasilar",
        "bilateral",
        "left lung",
        "right lung",
        "left",
        "right",
        "apical",
        "basilar",
        "lower lobe",
        "upper lobe"
    ]

    for location in locations:
        if location in context:
            return location

    return "unspecified"


def get_alternative_findings(text, current_finding):
    """
    Detect differential expressions such as:

        atelectasis versus pneumonia
        atelectasis vs consolidation
        pneumonia vs edema

    """

    lower = normalize_text(text)

    alternatives = []

    differential_patterns = [
        r"\bversus\b",
        r"\bv[./ ]?s\.?\b",
        r"\bcannot distinguish\b"
    ]

    has_differential = any(
        re.search(pattern, lower)
        for pattern in differential_patterns
    )

    if not has_differential:
        return alternatives

    for finding in FINDING_PATTERNS:

        if finding == current_finding:
            continue

        if contains_finding(text, finding):
            alternatives.append(finding)

    return alternatives


def section_has_finding(section_text, finding):
    return contains_finding(section_text, finding)


# ============================================================
# MODEL LOADING
# ============================================================

print("=" * 70)
print("FINAL CLINICAL AI EVIDENCE GENERATION")
print("=" * 70)

print(f"Input : {INPUT_FILE}")
print(f"Model : {MODEL_FILE}")
print(f"Output: {OUTPUT_FILE}")
print()

if not os.path.exists(INPUT_FILE):
    raise FileNotFoundError(
        f"Input file not found: {INPUT_FILE}"
    )

if not os.path.exists(MODEL_FILE):
    raise FileNotFoundError(
        f"Model not found: {MODEL_FILE}"
    )


print("Loading multimodal dataset...")

df = pd.read_csv(INPUT_FILE)

print(f"Rows loaded: {len(df)}")

print("Loading calibrated NLP model...")

model = joblib.load(MODEL_FILE)

print("Model loaded successfully.")
print()


# ============================================================
# GENERATE EVIDENCE
# ============================================================

evidence_records = []

processed = 0


for _, row in df.iterrows():

    patient_id = row.get("patient_id", "")

    png_path = row.get("png_path", "")

    # --------------------------------------------------------
    # Process each native report section independently.
    # --------------------------------------------------------

    for section_name, column_name in SECTION_COLUMNS:

        section_text = clean_text(
            row.get(column_name, "")
        )

        if not section_text:
            continue

        sentences = sentence_split(section_text)

        for sentence in sentences:

            sentence = clean_text(sentence)

            if not sentence:
                continue

            # ------------------------------------------------
            # Find every clinical finding in this sentence.
            # ------------------------------------------------

            for finding in FINDING_PATTERNS:

                spans = finding_match_positions(
                    sentence,
                    finding
                )

                if not spans:
                    continue

                for start, end in spans:

                    matched_text = sentence[start:end]

                    # ----------------------------------------
                    # Assertion
                    # ----------------------------------------

                    status, assertion_source = detect_assertion(
                        sentence,
                        start,
                        end
                    )

                    # ----------------------------------------
                    # Context
                    # ----------------------------------------

                    context = local_context(
                        sentence,
                        start,
                        end,
                        window=120
                    )

                    # ----------------------------------------
                    # Anatomical location
                    # ----------------------------------------

                    location = extract_location(
                        sentence,
                        start,
                        end
                    )

                    # ----------------------------------------
                    # Alternatives / differential
                    # ----------------------------------------

                    alternatives = get_alternative_findings(
                        sentence,
                        finding
                    )

                    # If differential wording is present,
                    # force involved findings to uncertain.
                    if alternatives and status == "present":

                        if re.search(
                            r"\b(versus|v[./ ]?s\.?|cannot distinguish)\b",
                            normalize_text(sentence)
                        ):
                            status = "uncertain"
                            assertion_source = "rule_differential"

                    # ----------------------------------------
                    # Model confidence
                    # ----------------------------------------

                    model_confidence = None

                    model_predicted_status = None

                    try:

                        probabilities = model.predict_proba(
                            [sentence]
                        )[0]

                        classes = list(model.classes_)

                        probability_map = {
                            str(cls): float(prob)
                            for cls, prob in zip(
                                classes,
                                probabilities
                            )
                        }

                        model_predicted_status = max(
                            probability_map,
                            key=probability_map.get
                        )

                        model_confidence = max(
                            probability_map.values()
                        )

                    except Exception:
                        pass

                    # ----------------------------------------
                    # Final confidence
                    # ----------------------------------------
                    #
                    # IMPORTANT:
                    # This is not "clinical certainty".
                    #
                    # It represents confidence in the NLP
                    # evidence extraction decision.
                    # ----------------------------------------

                    if assertion_source.startswith("rule_"):

                        # Rule-confirmed negation/uncertainty
                        # is given high extraction confidence,
                        # but we explicitly label its source.
                        extraction_confidence = 0.95

                    else:
                        extraction_confidence = (
                            model_confidence
                            if model_confidence is not None
                            else 0.50
                        )

                    evidence_records.append({

                        "patient_id": patient_id,

                        "png_path": png_path,

                        "finding": finding,

                        "status": status,

                        "evidence_text": sentence,

                        "matched_text": matched_text,

                        "section": section_name,

                        "anatomical_location": location,

                        "alternative_findings": (
                            "|".join(alternatives)
                            if alternatives
                            else ""
                        ),

                        "model_predicted_status": (
                            model_predicted_status
                            if model_predicted_status
                            else ""
                        ),

                        "model_confidence": (
                            round(model_confidence, 4)
                            if model_confidence is not None
                            else None
                        ),

                        "extraction_confidence": round(
                            extraction_confidence,
                            4
                        ),

                        "assertion_source": assertion_source,

                        "context": context
                    })

    processed += 1

    if processed % 500 == 0:
        print(
            f"Processed {processed}/{len(df)} "
            f"({processed / len(df) * 100:.1f}%)"
        )


# ============================================================
# SAVE
# ============================================================

evidence_df = pd.DataFrame(evidence_records)

os.makedirs(
    os.path.dirname(OUTPUT_FILE),
    exist_ok=True
)

evidence_df.to_csv(
    OUTPUT_FILE,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print()
print("=" * 70)
print("FINAL EVIDENCE GENERATION COMPLETE")
print("=" * 70)

print(f"Input records        : {len(df)}")
print(f"Evidence records     : {len(evidence_df)}")

if len(evidence_df) > 0:

    print()
    print("Finding distribution:")
    print(
        evidence_df["finding"]
        .value_counts()
        .to_string()
    )

    print()
    print("Status distribution:")
    print(
        evidence_df["status"]
        .value_counts()
        .to_string()
    )

    print()
    print("Section distribution:")
    print(
        evidence_df["section"]
        .value_counts()
        .to_string()
    )

    print()
    print("Assertion source:")
    print(
        evidence_df["assertion_source"]
        .value_counts()
        .to_string()
    )

    print()
    print(
        "Average model confidence:",
        round(
            evidence_df["model_confidence"].dropna().mean(),
            4
        )
    )

    print(
        "Average extraction confidence:",
        round(
            evidence_df["extraction_confidence"].mean(),
            4
        )
    )

print()
print(f"Saved to:")
print(OUTPUT_FILE)