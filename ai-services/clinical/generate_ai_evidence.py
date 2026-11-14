import os
import re
import pandas as pd
import joblib


# ============================================================
# PATHS
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
    "ai_evidence.csv"
)


# ============================================================
# FINDING PATTERNS
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
# UNCERTAINTY PATTERNS
# ============================================================

UNCERTAINTY_PATTERNS = [

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

    r"\bsuspicious\s+for\b",

    r"\bversus\b",
    r"\bv[./]s\.?\b"
]


# ============================================================
# STRONG NEGATION PATTERNS
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
# LOCATION PATTERNS
# ============================================================

LOCATION_PATTERNS = [

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

    "left lung",
    "right lung",

    "left lower",
    "right lower",

    "left upper",
    "right upper",

    "bibasilar",
    "bilateral",

    "apical",
    "basilar",

    "lower lobe",
    "upper lobe",

    "left",
    "right"
]


# ============================================================
# HELPERS
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value)

    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # Fix common PDF/report formatting issue:
    # "PLEURALEFFUSIONS" -> "PLEURAL EFFUSIONS"
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

    # Add spacing after numbered impression items.
    text = re.sub(
        r"(\d+)\.",
        r" \1. ",
        text
    )

    # Normalize whitespace
    text = re.sub(r"\s+", " ", text)

    return text.strip()


def normalize(text):

    return clean_text(text).lower()


# ============================================================
# SECTION PROCESSING
# ============================================================

def get_sections(row):

    sections = []

    clinical_history = clean_text(
        row.get("section_clinical_history", "")
    )

    findings = clean_text(
        row.get("section_findings", "")
    )

    impression = clean_text(
        row.get("section_impression", "")
    )

    # IMPORTANT:
    # Clinical history is intentionally NOT converted into
    # radiological evidence.
    #
    # It is contextual patient information, not confirmed
    # imaging evidence.

    if findings:
        sections.append(
            ("findings", findings)
        )

    if impression:
        sections.append(
            ("impression", impression)
        )

    return sections


# ============================================================
# SENTENCE / CLAUSE SPLITTING
# ============================================================

def split_into_clauses(text):

    text = clean_text(text)

    if not text:
        return []

    # --------------------------------------------------------
    # First split numbered impression items.
    # --------------------------------------------------------

    text = re.sub(
        r"\s+\d+\.\s*",
        ". ",
        text
    )

    # --------------------------------------------------------
    # Split normal sentences.
    # --------------------------------------------------------

    parts = re.split(
        r"(?<=[.!?])\s+(?=[A-Z])",
        text
    )

    final_parts = []

    for part in parts:

        part = part.strip()

        if not part:
            continue

        # ----------------------------------------------------
        # Handle common report constructions.
        #
        # Example:
        #
        # NO PLEURAL EFFUSION OR PNEUMOTHORAX
        #
        # Keep this as one clause because the local negation
        # applies to both findings.
        # ----------------------------------------------------

        final_parts.append(part)

    return final_parts


# ============================================================
# FINDING SEARCH
# ============================================================

def find_occurrences(text, finding):

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

    # Sort
    occurrences.sort(
        key=lambda x: x[0]
    )

    # Remove overlaps
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
# NEGATION DETECTION
# ============================================================

def detect_negation(
    text,
    start,
    end
):

    lower = normalize(text)

    # --------------------------------------------------------
    # Look backwards from the finding.
    # --------------------------------------------------------

    before = lower[
        max(0, start - 100):start
    ]

    # Remove unrelated previous clauses.
    #
    # This prevents:
    #
    # NO PNEUMOTHORAX. MILD PULMONARY EDEMA.
    #
    # from making pulmonary edema negative.
    #

    # Use the most recent sentence punctuation.
    sentence_break = max(
        before.rfind("."),
        before.rfind(";")
    )

    if sentence_break >= 0:

        before = before[
            sentence_break + 1:
        ]

    # --------------------------------------------------------
    # Direct negation.
    # --------------------------------------------------------

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

        # Negation should be close to finding.
        if len(words_after) <= 8:
            return True

    # --------------------------------------------------------
    # Special coordinated negation:
    #
    # "NO A OR B"
    # "NO A, B OR C"
    #
    # If "no" occurs shortly before the first finding,
    # it can apply to subsequent coordinated findings.
    # --------------------------------------------------------

    last_no = before.rfind("no ")

    if last_no >= 0:

        no_part = before[last_no:]

        # Avoid carrying negation across too much text.
        if len(no_part.split()) <= 10:
            return True

    return False


# ============================================================
# UNCERTAINTY DETECTION
# ============================================================

def detect_uncertainty(
    text,
    start,
    end
):
    """
    Determine whether THIS specific finding is uncertain.

    Important:
    A differential expression elsewhere in the clause must
    not make an unrelated finding uncertain.

    Example:

        LEFT PLEURAL EFFUSION WITH ATELECTASIS VERSUS
        CONSOLIDATION

    Expected:

        pleural_effusion -> present
        atelectasis      -> uncertain
        consolidation    -> uncertain
    """

    lower = normalize(text)

    # --------------------------------------------------------
    # Context around the specific finding
    # --------------------------------------------------------

    context = get_local_context(
        lower,
        start,
        end,
        window=45
    )

    # --------------------------------------------------------
    # 1. Direct uncertainty close to the finding
    # --------------------------------------------------------

    direct_uncertainty_patterns = [

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

    for pattern in direct_uncertainty_patterns:

        if re.search(
            pattern,
            context
        ):

            return True, "uncertainty"

    # --------------------------------------------------------
    # 2. Differential only when "versus" is close enough
    #    to THIS finding.
    #
    # Example:
    #
    # ATELECTASIS VERSUS CONSOLIDATION
    #
    # Both findings are close to "versus".
    #
    # But:
    #
    # PLEURAL EFFUSION ... ATELECTASIS VERSUS CONSOLIDATION
    #
    # should NOT make pleural_effusion uncertain.
    # --------------------------------------------------------

    versus_matches = list(
        re.finditer(
            r"\b(?:versus|v[./]\s*s[.]?)\b",
            lower
        )
    )

    for versus in versus_matches:

        # Text immediately before "versus"
        before_versus = lower[max(0, versus.start() - 60):versus.start()]

        # Text immediately after "versus"
        after_versus = lower[versus.end():min(len(lower), versus.end() + 60)]

        # --------------------------------------------------------
        # The finding must be directly involved in the
        # "X versus Y" relationship.
        # --------------------------------------------------------

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
    """
    Determine the assertion state for THIS finding only.

    Priority:
        1. Negation
        2. Local uncertainty
        3. Local differential
        4. Present

    This prevents a differential involving one finding from
    incorrectly changing the status of another finding in the
    same radiology sentence.

    Example:

        LEFT PLEURAL EFFUSION WITH ASSOCIATED
        LEFT BASILAR ATELECTASIS VERSUS CONSOLIDATION

    Result:

        pleural_effusion -> present
        atelectasis      -> uncertain
        consolidation    -> uncertain
    """

    # --------------------------------------------------------
    # 1. NEGATION
    # --------------------------------------------------------

    if detect_negation(
        text,
        start,
        end
    ):
        return (
            "absent",
            "rule_negation"
        )

    # --------------------------------------------------------
    # 2. LOCAL UNCERTAINTY
    # --------------------------------------------------------

    uncertain, reason = detect_uncertainty(
        text,
        start,
        end
    )

    if uncertain:

        if reason == "differential":
            return (
                "uncertain",
                "rule_differential"
            )

        return (
            "uncertain",
            "rule_uncertainty"
        )

    # --------------------------------------------------------
    # 3. PRESENT
    # --------------------------------------------------------

    return (
        "present",
        "rule_positive"
    )


# ============================================================
# LOCATION
# ============================================================

def extract_location(
    text,
    start,
    end
):

    context = normalize(
        get_local_context(
            text,
            start,
            end,
            window=100
        )
    )

    # Most specific first.
    for location in LOCATION_PATTERNS:

        if location in context:

            return location

    return "unspecified"


# ============================================================
# DIFFERENTIAL / ALTERNATIVE FINDINGS
# ============================================================

def extract_alternatives(
    text,
    current_finding
):

    lower = normalize(text)

    has_differential = re.search(
        r"\b(versus|v[./]s\.?)\b",
        lower
    )

    if not has_differential:

        return []

    alternatives = []

    for finding in FINDING_PATTERNS:

        if finding == current_finding:
            continue

        if find_occurrences(
            text,
            finding
        ):

            alternatives.append(
                finding
            )

    return alternatives


# ============================================================
# MODEL CONFIDENCE
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

        return predicted, confidence

    except Exception:

        return None, None


# ============================================================
# MAIN
# ============================================================

print("=" * 70)
print("CLINICAL AI EVIDENCE GENERATION")
print("=" * 70)

print(
    f"Input : {INPUT_FILE}"
)

print(
    f"Model : {MODEL_FILE}"
)

print(
    f"Output: {OUTPUT_FILE}"
)

print()


# ============================================================
# LOAD DATA
# ============================================================

if not os.path.exists(INPUT_FILE):

    raise FileNotFoundError(
        f"Dataset not found: {INPUT_FILE}"
    )

if not os.path.exists(MODEL_FILE):

    raise FileNotFoundError(
        f"NLP model not found: {MODEL_FILE}"
    )


print("Loading multimodal dataset...")

df = pd.read_csv(
    INPUT_FILE
)

print(
    f"Rows loaded: {len(df)}"
)

print()

print("Loading calibrated NLP model...")

model = joblib.load(
    MODEL_FILE
)

print(
    "Model loaded successfully."
)

print()


# ============================================================
# GENERATE EVIDENCE
# ============================================================

evidence_records = []

processed = 0


for _, row in df.iterrows():

    patient_id = row.get(
        "patient_id",
        ""
    )

    png_path = row.get(
        "png_path",
        ""
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # Only Findings + Impression are used for imaging evidence.
    #
    # Clinical history is retained in the source dataset but
    # does not generate confirmed radiological evidence.
    # --------------------------------------------------------

    sections = get_sections(
        row
    )

    for section_name, section_text in sections:

        clauses = split_into_clauses(
            section_text
        )

        for clause in clauses:

            clause = clean_text(
                clause
            )

            if not clause:
                continue

            # ------------------------------------------------
            # Model prediction for this clause.
            # ------------------------------------------------

            model_status, model_confidence = (
                get_model_prediction(
                    model,
                    clause
                )
            )

            # ------------------------------------------------
            # Process every finding separately.
            # ------------------------------------------------

            for finding in FINDING_PATTERNS:

                occurrences = find_occurrences(
                    clause,
                    finding
                )

                if not occurrences:
                    continue

                for start, end, matched_text in occurrences:

                    # ----------------------------------------
                    # Assertion
                    # ----------------------------------------

                    status, assertion_source = (
                        determine_assertion(
                            clause,
                            start,
                            end
                        )
                    )

                    # ----------------------------------------
                    # Context
                    # ----------------------------------------

                    context = get_local_context(
                        clause,
                        start,
                        end,
                        window=120
                    )

                    # ----------------------------------------
                    # Location
                    # ----------------------------------------

                    location = extract_location(
                        clause,
                        start,
                        end
                    )

                    # ----------------------------------------
                    # Alternatives
                    # ----------------------------------------

                    alternatives = (
                        extract_alternatives(
                            clause,
                            finding
                        )
                    )

                    # ----------------------------------------
                    # Extraction confidence
                    # ----------------------------------------
                    #
                    # This is NOT clinical probability.
                    #
                    # It represents confidence in the
                    # assertion-normalization process.
                    # ----------------------------------------

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

                    evidence_records.append({

                        "patient_id": patient_id,

                        "png_path": png_path,

                        "finding": finding,

                        "status": status,

                        "evidence_text": clause,

                        "matched_text": matched_text,

                        "section": section_name,

                        "anatomical_location": location,

                        "alternative_findings": (
                            "|".join(
                                alternatives
                            )
                            if alternatives
                            else ""
                        ),

                        "model_predicted_status": (
                            model_status
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

                        "extraction_confidence": (
                            round(
                                extraction_confidence,
                                4
                            )
                        ),

                        "assertion_source": (
                            assertion_source
                        ),

                        "context": context
                    })

    processed += 1

    if processed % 500 == 0:

        print(
            f"Processed "
            f"{processed}/{len(df)} "
            f"({processed / len(df) * 100:.1f}%)"
        )


# ============================================================
# DATAFRAME
# ============================================================

evidence_df = pd.DataFrame(
    evidence_records
)


# ============================================================
# REMOVE DUPLICATE EVIDENCE
# ============================================================

if len(evidence_df) > 0:

    evidence_df = (
        evidence_df
        .drop_duplicates(
            subset=[
                "patient_id",
                "finding",
                "status",
                "evidence_text",
                "section"
            ]
        )
        .reset_index(
            drop=True
        )
    )


# ============================================================
# SAVE
# ============================================================

os.makedirs(
    os.path.dirname(
        OUTPUT_FILE
    ),
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
print("CLINICAL AI EVIDENCE GENERATION COMPLETE")
print("=" * 70)

print(
    f"Input records    : {len(df)}"
)

print(
    f"Evidence records : {len(evidence_df)}"
)

if len(evidence_df) > 0:

    print()

    print("Finding distribution:")
    print(
        evidence_df[
            "finding"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print("Status distribution:")
    print(
        evidence_df[
            "status"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print("Section distribution:")
    print(
        evidence_df[
            "section"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print("Assertion source:")
    print(
        evidence_df[
            "assertion_source"
        ]
        .value_counts()
        .to_string()
    )

    print()

    print(
        "Average model confidence:",
        round(
            evidence_df[
                "model_confidence"
            ]
            .dropna()
            .mean(),
            4
        )
    )

    print(
        "Average extraction confidence:",
        round(
            evidence_df[
                "extraction_confidence"
            ]
            .mean(),
            4
        )
    )

print()

print(
    f"Saved to:\n{OUTPUT_FILE}"
)