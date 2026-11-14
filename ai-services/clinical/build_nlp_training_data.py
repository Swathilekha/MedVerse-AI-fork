import os
import re
import pandas as pd


INPUT_PATH = "data/clinical_text/multimodal_dataset.csv"
OUTPUT_PATH = "data/clinical_text/nlp_training_data.csv"


FINDINGS = {
    "pneumonia": [
        "pneumonia"
    ],
    "atelectasis": [
        "atelectasis"
    ],
    "consolidation": [
        "consolidation",
        "consolidated"
    ],
    "pleural_effusion": [
        "pleural effusion",
        "pleural effusions"
    ],
    "pneumothorax": [
        "pneumothorax"
    ],
    "pulmonary_edema": [
        "pulmonary edema",
        "interstitial pulmonary edema"
    ],
    "cardiomegaly": [
        "cardiomegaly",
        "enlarged heart"
    ],
    "lung_opacity": [
        "opacity",
        "opacities",
        "airspace opacity",
        "airspace opacities"
    ]
}


NEGATION_PATTERNS = [
    r"\bno\b",
    r"\bwithout\b",
    r"\bno evidence of\b",
    r"\bwithout evidence of\b",
    r"\bnegative for\b",
    r"\babsence of\b",
    r"\babsent\b"
]


UNCERTAINTY_PATTERNS = [
    r"\bpossible\b",
    r"\bpossibly\b",
    r"\bmay be\b",
    r"\bmay represent\b",
    r"\bcannot be excluded\b",
    r"\bsuspicious for\b",
    r"\bquestionable\b",
    r"\bversus\b",
    r"\bvs\b",
    r"\bconcerning for\b",
    r"\bsuggestive of\b"
]


def split_sentences(text):

    return re.split(
        r"(?<=[.!?])\s+",
        str(text)
    )


def contains_pattern(text, patterns):

    text = text.lower()

    for pattern in patterns:

        if re.search(pattern, text):
            return True

    return False


def determine_status(sentence, term):

    lower = sentence.lower()

    position = lower.find(term.lower())

    if position == -1:
        return None

    # Examine local context immediately before the finding.
    before = lower[
        max(0, position - 120):position
    ]

    # Negation has priority.
    for pattern in NEGATION_PATTERNS:

        if re.search(pattern, before):
            return "absent"

    # Uncertainty.
    after = lower[
        position:min(len(lower), position + 120)
    ]

    context = before + " " + after

    for pattern in UNCERTAINTY_PATTERNS:

        if re.search(pattern, context):
            return "uncertain"

    return "present"


def main():

    print("=" * 70)
    print("BUILDING CLINICAL NLP TRAINING DATA")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    print("Input reports:", len(df))

    records = []

    for _, row in df.iterrows():

        text = str(row.get("clinical_text", ""))

        if not text or text.lower() == "nan":
            continue

        sentences = split_sentences(text)

        for sentence in sentences:

            sentence = sentence.strip()

            if len(sentence) < 10:
                continue

            lower_sentence = sentence.lower()

            for finding, terms in FINDINGS.items():

                matched_term = None

                for term in terms:

                    if term.lower() in lower_sentence:

                        matched_term = term
                        break

                if matched_term is None:
                    continue

                status = determine_status(
                    sentence,
                    matched_term
                )

                if status is None:
                    continue

                records.append({
                    "patient_id": row["patient_id"],
                    "png_path": row["png_path"],
                    "final_split": row["final_split"],
                    "finding": finding,
                    "text": sentence,
                    "label": status
                })

    result = pd.DataFrame(records)

    # Remove duplicate examples.
    result = result.drop_duplicates(
        subset=[
            "patient_id",
            "finding",
            "text",
            "label"
        ]
    )

    os.makedirs(
        os.path.dirname(OUTPUT_PATH),
        exist_ok=True
    )

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print("\nTraining examples:", len(result))

    print("\nFindings:")
    print(
        result["finding"]
        .value_counts()
        .to_string()
    )

    print("\nLabels:")
    print(
        result["label"]
        .value_counts()
        .to_string()
    )

    print("\nFinding × Label:")
    print(
        pd.crosstab(
            result["finding"],
            result["label"]
        ).to_string()
    )

    print("\nSamples:")
    print(
        result[
            ["finding", "label", "text"]
        ].sample(
            min(20, len(result)),
            random_state=42
        ).to_string(index=False)
    )

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()