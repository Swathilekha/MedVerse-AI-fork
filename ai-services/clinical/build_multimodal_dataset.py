import os
import re
import pandas as pd


# ============================================================
# MEDVERSE AI
# CHEXPERT PLUS - MULTIMODAL DATASET BUILDER
# ============================================================

print("\n========================================")
print("MEDVERSE AI - MULTIMODAL DATASET BUILDER")
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

MANIFEST_PATH = os.path.join(
    DATA_DIR,
    "manifests",
    "chexpert_final_manifest.csv"
)

CLINICAL_PATH = os.path.join(
    DATA_DIR,
    "clinical_text",
    "clinical_text_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    DATA_DIR,
    "clinical_text"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "multimodal_dataset.csv"
)


# ============================================================
# LOAD FILES
# ============================================================

print("\nLoading final image manifest...")

manifest = pd.read_csv(
    MANIFEST_PATH,
    low_memory=False
)

print(
    "Final manifest records:",
    len(manifest)
)

print("\nLoading clinical text dataset...")

clinical = pd.read_csv(
    CLINICAL_PATH,
    low_memory=False
)

print(
    "Clinical records:",
    len(clinical)
)


# ============================================================
# NORMALIZE IMAGE PATH
# ============================================================

def normalize_image_path(path):

    if pd.isna(path):
        return ""

    path = str(path).strip()

    # Convert Windows separators
    path = path.replace("\\", "/")

    # Remove leading train/
    if path.startswith("train/"):
        path = path[6:]

    # Remove leading valid/
    if path.startswith("valid/"):
        path = path[6:]

    # Remove leading val/
    if path.startswith("val/"):
        path = path[4:]

    # Convert jpg/jpeg/png to common representation
    path = re.sub(
        r"\.(jpg|jpeg|png)$",
        "",
        path,
        flags=re.IGNORECASE
    )

    return path.lower()


# ============================================================
# CREATE JOIN KEYS
# ============================================================

print("\nCreating image matching keys...")

manifest["join_key"] = (
    manifest["png_path"]
    .apply(normalize_image_path)
)

clinical["join_key"] = (
    clinical["path_to_image"]
    .apply(normalize_image_path)
)


# ============================================================
# CHECK DUPLICATES IN CLINICAL DATA
# ============================================================

duplicate_count = clinical["join_key"].duplicated().sum()

print(
    "Duplicate clinical image keys:",
    duplicate_count
)


# Keep one exact report record per image path.
# The source dataset normally maps each image path to its
# corresponding report.
clinical_unique = clinical.drop_duplicates(
    subset=["join_key"],
    keep="first"
).copy()


# ============================================================
# SELECT CLINICAL INFORMATION
# ============================================================

clinical_columns = [
    "join_key",
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
    "report",
    "clinical_text"
]

clinical_subset = clinical_unique[
    clinical_columns
].copy()


# ============================================================
# MERGE
# ============================================================

print("\nJoining image manifest with clinical information...")

multimodal = manifest.merge(
    clinical_subset,
    on="join_key",
    how="left",
    indicator=True
)


# ============================================================
# MATCHING STATISTICS
# ============================================================

matched = (
    multimodal["_merge"] == "both"
).sum()

unmatched = (
    multimodal["_merge"] != "both"
).sum()

print("\n========================================")
print("IMAGE ↔ REPORT MATCHING")
print("========================================")

print(
    "Manifest images:",
    len(manifest)
)

print(
    "Matched with clinical record:",
    matched
)

print(
    "Unmatched:",
    unmatched
)

match_percentage = (
    matched / len(manifest) * 100
)

print(
    f"Match percentage: {match_percentage:.2f}%"
)


# ============================================================
# REPORT AVAILABILITY
# ============================================================

has_report = (
    multimodal["report"]
    .fillna("")
    .str.strip()
    .str.len()
    > 0
)

print(
    "Records with report text:",
    has_report.sum()
)

print(
    "Records without report text:",
    (~has_report).sum()
)


# ============================================================
# PATIENT CONSISTENCY CHECK
# ============================================================

patient_match = (
    multimodal["patient_id"].astype(str)
    ==
    multimodal["deid_patient_id"].astype(str)
)

patient_match_count = (
    patient_match & multimodal["deid_patient_id"].notna()
).sum()

patient_mismatch = (
    multimodal["deid_patient_id"].notna()
    &
    (~patient_match)
).sum()

print(
    "\nPatient ID mismatches:",
    patient_mismatch
)


# ============================================================
# REMOVE INTERNAL COLUMNS
# ============================================================

multimodal = multimodal.drop(
    columns=["_merge", "join_key"]
)


# ============================================================
# REORDER COLUMNS
# ============================================================

desired_columns = [
    "png_path",
    "local_path",
    "class_name",
    "patient_id",
    "final_split",

    "age",
    "sex",
    "race",
    "ethnicity",

    "frontal_lateral",
    "ap_pa",

    "section_clinical_history",
    "section_findings",
    "section_impression",

    "report",
    "clinical_text"
]

desired_columns = [
    col
    for col in desired_columns
    if col in multimodal.columns
]

multimodal = multimodal[
    desired_columns
]


# ============================================================
# SAVE
# ============================================================

multimodal.to_csv(
    OUTPUT_PATH,
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n========================================")
print("MULTIMODAL DATASET CREATED")
print("========================================")

print(
    "Total records:",
    len(multimodal)
)

print(
    "Unique patients:",
    multimodal["patient_id"].nunique()
)

print("\nClass distribution:")

print(
    multimodal["class_name"]
    .value_counts()
)

print("\nSplit distribution:")

print(
    multimodal["final_split"]
    .value_counts()
)

print(
    "\nOutput:",
    OUTPUT_PATH
)

print("\nFirst record:")

print(
    multimodal.iloc[0].to_string()
)

print("\n========================================")
print("DONE")
print("========================================")