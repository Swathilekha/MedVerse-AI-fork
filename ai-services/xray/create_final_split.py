import os
import shutil
import pandas as pd
from sklearn.model_selection import train_test_split


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MANIFEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "manifests",
    "chexpert_png_download_list.csv"
)

SOURCE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "chexpert_prepared"
)

FINAL_DIR = os.path.join(
    BASE_DIR,
    "data",
    "chexpert_final"
)

CLASSES = [
    "NORMAL",
    "PNEUMONIA"
]

VALIDATION_SIZE = 0.20
RANDOM_STATE = 42


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def extract_patient_id(path):

    """
    Example:

    patient00403/study2/view1_frontal.png

    becomes:

    patient00403
    """

    path = str(path).replace("\\", "/")

    parts = path.split("/")

    if len(parts) < 2:
        raise ValueError(
            f"Cannot extract patient ID from path: {path}"
        )

    return parts[0]


def clear_directory(directory):

    """
    Remove an existing directory so that an old split
    cannot contaminate the new split.
    """

    if os.path.exists(directory):

        print(f"Removing existing directory: {directory}")

        shutil.rmtree(directory)


# ============================================================
# START
# ============================================================

print("\n========================================")
print("MEDVERSE AI - FINAL DATASET SPLIT")
print("========================================")


# ============================================================
# CHECK MANIFEST
# ============================================================

if not os.path.exists(MANIFEST_PATH):

    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST_PATH}"
    )

print("\nLoading download manifest...")

df = pd.read_csv(MANIFEST_PATH)

print("Manifest rows:", len(df))


# ============================================================
# CHECK REQUIRED COLUMNS
# ============================================================

required_columns = [
    "split",
    "class_name",
    "png_path"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"Required column missing: {column}"
        )


# ============================================================
# CHECK CLASSES
# ============================================================

df = df[
    df["class_name"].isin(CLASSES)
].copy()

print("\nClass distribution:")

print(
    df["class_name"].value_counts()
)


# ============================================================
# BUILD LOCAL IMAGE PATHS
# ============================================================

def find_local_image(row):

    split = str(row["split"]).lower()
    class_name = row["class_name"]
    png_path = str(row["png_path"]).replace(
        "\\", "/"
    )

    # Current downloader structure:
    #
    # data/
    #   chexpert_prepared/
    #       train/
    #           NORMAL/
    #               patient.../
    #
    #       valid/
    #           NORMAL/
    #               patient.../

    path = os.path.join(
        SOURCE_DIR,
        split,
        class_name,
        *png_path.split("/")
    )

    return path


df["local_path"] = df.apply(
    find_local_image,
    axis=1
)


# ============================================================
# VERIFY ALL IMAGES
# ============================================================

print("\nChecking local images...")

missing = df[
    ~df["local_path"].apply(os.path.exists)
]

if len(missing) > 0:

    print(
        f"\nERROR: {len(missing)} image(s) are missing."
    )

    print(
        missing[
            [
                "split",
                "class_name",
                "png_path"
            ]
        ].head(20).to_string(index=False)
    )

    raise FileNotFoundError(
        "Cannot create final split because images are missing."
    )

print(
    f"All {len(df)} images are available locally."
)


# ============================================================
# EXTRACT PATIENT ID
# ============================================================

df["patient_id"] = df["png_path"].apply(
    extract_patient_id
)


# ============================================================
# CHECK PATIENT-LEVEL CLASS CONSISTENCY
# ============================================================

patient_class_counts = (
    df.groupby("patient_id")["class_name"]
    .nunique()
)

mixed_patients = patient_class_counts[
    patient_class_counts > 1
]

if len(mixed_patients) > 0:

    print(
        "\nWARNING:"
    )

    print(
        f"{len(mixed_patients)} patients contain "
        "multiple class labels."
    )

    print(
        "These patients will be removed to avoid "
        "patient-level label ambiguity."
    )

    df = df[
        ~df["patient_id"].isin(
            mixed_patients.index
        )
    ].copy()


# ============================================================
# PATIENT TABLE
# ============================================================

patient_df = (
    df[
        [
            "patient_id",
            "class_name"
        ]
    ]
    .drop_duplicates()
)

print("\nUnique patients:", len(patient_df))

print("\nPatients by class:")

print(
    patient_df["class_name"].value_counts()
)


# ============================================================
# PATIENT-LEVEL TRAIN / VALIDATION SPLIT
# ============================================================

train_patients, val_patients = train_test_split(
    patient_df,
    test_size=VALIDATION_SIZE,
    random_state=RANDOM_STATE,
    stratify=patient_df["class_name"]
)


train_patient_ids = set(
    train_patients["patient_id"]
)

val_patient_ids = set(
    val_patients["patient_id"]
)


# ============================================================
# SAFETY CHECK
# ============================================================

overlap = train_patient_ids.intersection(
    val_patient_ids
)

if len(overlap) > 0:

    raise RuntimeError(
        "Patient leakage detected! "
        "Some patients appear in both splits."
    )

print("\nPatient-level split successful.")

print(
    "Training patients:",
    len(train_patient_ids)
)

print(
    "Validation patients:",
    len(val_patient_ids)
)


# ============================================================
# CREATE FINAL DIRECTORIES
# ============================================================

clear_directory(FINAL_DIR)

for split in [
    "train",
    "val"
]:

    for class_name in CLASSES:

        os.makedirs(
            os.path.join(
                FINAL_DIR,
                split,
                class_name
            ),
            exist_ok=True
        )


# ============================================================
# ASSIGN IMAGES
# ============================================================

df["final_split"] = df["patient_id"].apply(
    lambda patient:
        "train"
        if patient in train_patient_ids
        else "val"
)


# ============================================================
# COPY IMAGES
# ============================================================

print("\nCreating final dataset...")

copied = 0

for _, row in df.iterrows():

    source = row["local_path"]

    final_split = row["final_split"]

    class_name = row["class_name"]

    patient_id = row["patient_id"]

    filename = os.path.basename(
        row["png_path"]
    )

    # Preserve patient/study structure
    destination = os.path.join(
        FINAL_DIR,
        final_split,
        class_name,
        patient_id,
        filename
    )

    os.makedirs(
        os.path.dirname(destination),
        exist_ok=True
    )

    shutil.copy2(
        source,
        destination
    )

    copied += 1

    if copied % 500 == 0:

        print(
            f"Copied {copied}/{len(df)} images..."
        )


# ============================================================
# SAVE FINAL MANIFEST
# ============================================================

final_manifest = os.path.join(
    BASE_DIR,
    "data",
    "manifests",
    "chexpert_final_manifest.csv"
)

df[
    [
        "png_path",
        "class_name",
        "patient_id",
        "final_split",
        "local_path"
    ]
].to_csv(
    final_manifest,
    index=False
)


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n========================================")
print("FINAL DATASET")
print("========================================")


for split in [
    "train",
    "val"
]:

    split_df = df[
        df["final_split"] == split
    ]

    print(f"\n{split.upper()}")

    print(
        "Patients:",
        split_df["patient_id"].nunique()
    )

    print(
        split_df["class_name"]
        .value_counts()
        .to_string()
    )

    print(
        "Total images:",
        len(split_df)
    )


# ============================================================
# FINAL LEAKAGE CHECK
# ============================================================

train_ids = set(
    df[
        df["final_split"] == "train"
    ]["patient_id"]
)

val_ids = set(
    df[
        df["final_split"] == "val"
    ]["patient_id"]
)

overlap = train_ids.intersection(
    val_ids
)

print("\n========================================")
print("PATIENT LEAKAGE CHECK")
print("========================================")

print(
    "Training patients:",
    len(train_ids)
)

print(
    "Validation patients:",
    len(val_ids)
)

print(
    "Patient overlap:",
    len(overlap)
)

if len(overlap) != 0:

    raise RuntimeError(
        "ERROR: Patient leakage detected."
    )

print(
    "Patient-level separation verified."
)


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("FINAL SPLIT COMPLETE")
print("========================================")

print(
    "\nFinal dataset:"
)

print(FINAL_DIR)

print(
    "\nFinal manifest:"
)

print(final_manifest)

print(
    "\nYou can now train DenseNet121 using:"
)

print(
    "data/chexpert_final/train"
)

print(
    "data/chexpert_final/val"
)

print("\n========================================")