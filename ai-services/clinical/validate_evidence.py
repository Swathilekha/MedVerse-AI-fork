import os
import pandas as pd


BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

FILE = os.path.join(
    BASE_DIR,
    "data",
    "clinical_text",
    "ai_evidence_final.csv"
)


df = pd.read_csv(FILE)

print("=" * 80)
print("EVIDENCE VALIDATION")
print("=" * 80)

print(f"Rows: {len(df)}")
print()


# ============================================================
# 1. GENERAL SUMMARY
# ============================================================

print("STATUS DISTRIBUTION")
print("-" * 80)
print(df["status"].value_counts())
print()

print("SECTION DISTRIBUTION")
print("-" * 80)
print(df["section"].value_counts())
print()

print("ASSERTION SOURCES")
print("-" * 80)
print(df["assertion_source"].value_counts())
print()


# ============================================================
# 2. SEARCH FOR IMPORTANT EDGE CASES
# ============================================================

patterns = [
    "no evidence of pulmonary edema",
    "no focal air space consolidation",
    "no pleural effusion",
    "no pneumothorax",
    "may represent atelectasis",
    "may correspond to a trace pleural effusion",
    "atelectasis versus consolidation",
    "consolidation and left pleural effusion",
    "rule out infiltrate",
    "likely atelectasis"
]


for pattern in patterns:

    print()
    print("=" * 80)
    print(f"SEARCH: {pattern}")
    print("=" * 80)

    matches = df[
        df["evidence_text"]
        .fillna("")
        .str.lower()
        .str.contains(
            pattern.lower(),
            regex=False
        )
    ]

    if len(matches) == 0:
        print("No matching evidence.")
        continue

    columns = [
        "patient_id",
        "finding",
        "status",
        "section",
        "anatomical_location",
        "model_confidence",
        "extraction_confidence",
        "assertion_source",
        "evidence_text"
    ]

    print(
        matches[columns]
        .head(10)
        .to_string(index=False)
    )