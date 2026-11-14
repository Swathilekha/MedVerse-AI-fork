import os
import re
import pandas as pd


# ============================================================
# MEDVERSE AI
# CHEXPERT PLUS - CLINICAL TEXT PREPARATION
# ============================================================

print("\n========================================")
print("MEDVERSE AI - CLINICAL TEXT PREPARATION")
print("========================================")


# ============================================================
# PROJECT PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIR = os.path.join(BASE_DIR, "data")

INPUT_CSV = os.path.join(
    DATA_DIR,
    "df_chexpert_plus_240401.csv"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "clinical_text"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "clinical_text_dataset.csv"
)


# ============================================================
# LOAD DATA
# ============================================================

print("\nLoading CheXpert Plus metadata...")

df = pd.read_csv(
    INPUT_CSV,
    low_memory=False
)

print("Rows:", len(df))
print("Columns:", len(df.columns))


# ============================================================
# REQUIRED COLUMNS
# ============================================================

required_columns = [
    "path_to_image",
    "deid_patient_id",
    "report",
    "section_clinical_history",
    "section_findings",
    "section_impression",
    "age",
    "sex",
    "race",
    "ethnicity",
    "frontal_lateral",
    "ap_pa"
]

missing = [
    col for col in required_columns
    if col not in df.columns
]

if missing:
    raise ValueError(
        f"\nMissing columns:\n{missing}"
    )


# ============================================================
# TEXT CLEANING
# ============================================================

def clean_text(value):

    if pd.isna(value):
        return ""

    text = str(value)

    # Normalize line breaks
    text = text.replace("\n", " ")
    text = text.replace("\r", " ")

    # Remove excessive whitespace
    text = re.sub(
        r"\s+",
        " ",
        text
    )

    return text.strip()


text_columns = [
    "report",
    "section_clinical_history",
    "section_findings",
    "section_impression"
]

for column in text_columns:

    df[column] = df[column].apply(
        clean_text
    )


# ============================================================
# CREATE CLEAN DATASET
# ============================================================

clinical_df = df[
    [
        "path_to_image",
        "deid_patient_id",
        "age",
        "sex",
        "race",
        "ethnicity",
        "frontal_lateral",
        "ap_pa",
        "section_clinical_history",
        "section_findings",
        "section_impression",
        "report"
    ]
].copy()


# ============================================================
# REMOVE COMPLETELY EMPTY REPORTS
# ============================================================

clinical_df = clinical_df[
    clinical_df["report"].str.len() > 0
].copy()


# ============================================================
# CREATE COMBINED CLINICAL TEXT
# ============================================================

def combine_text(row):

    parts = []

    if row["section_clinical_history"]:
        parts.append(
            "CLINICAL HISTORY: "
            + row["section_clinical_history"]
        )

    if row["section_findings"]:
        parts.append(
            "FINDINGS: "
            + row["section_findings"]
        )

    if row["section_impression"]:
        parts.append(
            "IMPRESSION: "
            + row["section_impression"]
        )

    return " ".join(parts)


clinical_df["clinical_text"] = clinical_df.apply(
    combine_text,
    axis=1
)


# ============================================================
# SAVE
# ============================================================

clinical_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("CLINICAL TEXT DATASET CREATED")
print("========================================")

print(
    "Records:",
    len(clinical_df)
)

print(
    "Unique patients:",
    clinical_df["deid_patient_id"].nunique()
)

print(
    "Records with clinical text:",
    (
        clinical_df["clinical_text"].str.len() > 0
    ).sum()
)

print(
    "\nOutput:",
    OUTPUT_CSV
)

print("\nSample clinical text:")
print("----------------------------------------")

sample = clinical_df.iloc[0]

print(
    sample["clinical_text"][:1000]
)

print("----------------------------------------")
print("DONE")