import pandas as pd
from pathlib import Path


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

INPUT_FILE = PROJECT_ROOT / "data" / "clinical_text" / "multimodal_dataset.csv"
OUTPUT_FILE = PROJECT_ROOT / "data" / "clinical_text" / "fusion_experiment_dataset.csv"


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 60)
print("CREATING FUSION EXPERIMENT MANIFEST")
print("=" * 60)

print(f"\nInput: {INPUT_FILE}")

df = pd.read_csv(INPUT_FILE)

print(f"Total records: {len(df)}")

print("\nAvailable columns:")
print(df.columns.tolist())


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "patient_id",
    "local_path",
    "clinical_text",
    "class_name",
    "final_split"
]

missing = [col for col in required_columns if col not in df.columns]

if missing:
    raise ValueError(
        f"\nMissing required columns: {missing}"
    )


# ============================================================
# CREATE CLEAN DATASET
# ============================================================

fusion_df = pd.DataFrame()

fusion_df["patient_id"] = df["patient_id"]

# Use the actual local image path
fusion_df["image_path"] = df["local_path"]

fusion_df["clinical_text"] = (
    df["clinical_text"]
    .fillna("")
    .astype(str)
)

# NORMAL = 0
# PNEUMONIA = 1
fusion_df["label"] = (
    df["class_name"]
    .astype(str)
    .str.upper()
    .map({
        "NORMAL": 0,
        "PNEUMONIA": 1
    })
)

fusion_df["final_split"] = (
    df["final_split"]
    .astype(str)
    .str.lower()
)


# ============================================================
# VALIDATION
# ============================================================

print("\nLabel distribution:")

print(
    fusion_df["label"]
    .value_counts(dropna=False)
)


print("\nSplit distribution:")

print(
    fusion_df["final_split"]
    .value_counts(dropna=False)
)


# Check missing values
print("\nMissing values:")

print(
    fusion_df.isnull().sum()
)


# ============================================================
# REMOVE INVALID RECORDS
# ============================================================

before = len(fusion_df)

fusion_df = fusion_df.dropna(
    subset=[
        "patient_id",
        "image_path",
        "label",
        "final_split"
    ]
)

fusion_df = fusion_df[
    fusion_df["clinical_text"].str.strip() != ""
]

after = len(fusion_df)

print(f"\nRemoved invalid records: {before - after}")


# ============================================================
# SAVE
# ============================================================

fusion_df.to_csv(
    OUTPUT_FILE,
    index=False
)

print("\n" + "=" * 60)
print("MANIFEST CREATED")
print("=" * 60)

print(f"\nOutput: {OUTPUT_FILE}")
print(f"Records: {len(fusion_df)}")

print("\nFinal columns:")
print(fusion_df.columns.tolist())

print("\nFinal split:")
print(
    fusion_df["final_split"]
    .value_counts()
)

print("\nFinal labels:")
print(
    fusion_df["label"]
    .value_counts()
)

print("\nFirst 5 records:")
print(
    fusion_df.head().to_string(index=False)
)

print("\nDone.")