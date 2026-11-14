import os
import json
import shutil
import pandas as pd


# ============================================================
# MEDVERSE AI
# CHEXPERT PLUS DATASET PREPARATION
# ============================================================

print("\n========================================")
print("MEDVERSE AI - CHEXPERT PLUS PREPARATION")
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

CSV_PATH = os.path.join(
    DATA_DIR,
    "df_chexpert_plus_240401.csv"
)

FINDINGS_PATH = os.path.join(
    DATA_DIR,
    "findings_fixed.json"
)

PNG_TRAIN_DIR = os.path.join(
    DATA_DIR,
    "PNG_train"
)

PNG_VALID_DIR = os.path.join(
    DATA_DIR,
    "PNG_valid"
)

PREPARED_DIR = os.path.join(
    DATA_DIR,
    "chexpert_prepared"
)

MANIFEST_DIR = os.path.join(
    DATA_DIR,
    "manifests"
)


# ============================================================
# CONFIGURATION
# ============================================================

TARGET_LABEL = "Pneumonia"

# Maximum images per class.
# Set to None later if we want the complete dataset.
MAX_IMAGES_PER_CLASS = 5000

# Copy actual PNG files into prepared dataset.
COPY_IMAGES = True


# ============================================================
# CREATE DIRECTORIES
# ============================================================

os.makedirs(PREPARED_DIR, exist_ok=True)
os.makedirs(MANIFEST_DIR, exist_ok=True)

for split in ["train", "val"]:
    for class_name in ["NORMAL", "PNEUMONIA"]:
        os.makedirs(
            os.path.join(
                PREPARED_DIR,
                split,
                class_name
            ),
            exist_ok=True
        )


# ============================================================
# CHECK REQUIRED FILES
# ============================================================

print("\nChecking dataset files...")

required_files = [
    CSV_PATH,
    FINDINGS_PATH
]

for path in required_files:

    if not os.path.exists(path):

        raise FileNotFoundError(
            f"\nRequired file not found:\n{path}\n"
        )

    print("FOUND:", path)


# ============================================================
# LOAD METADATA CSV
# ============================================================

print("\nLoading metadata CSV...")

df = pd.read_csv(CSV_PATH)

print("Metadata shape:", df.shape)

print("\nMetadata columns:")
print(df.columns.tolist())


# ============================================================
# CHECK REQUIRED CSV COLUMNS
# ============================================================

required_columns = [
    "path_to_image",
    "deid_patient_id",
    "split"
]

for column in required_columns:

    if column not in df.columns:

        raise ValueError(
            f"\nRequired column '{column}' "
            f"not found in metadata CSV."
        )


print("\nRequired CSV columns found.")


# ============================================================
# LOAD FINDINGS JSONL
# ============================================================
#
# IMPORTANT:
# findings_fixed.json is JSON Lines (JSONL).
# Each line contains one JSON object.
#
# Therefore json.load() CANNOT be used.
# We read one JSON object per line.
# ============================================================

print("\nLoading findings JSONL...")

findings_data = []

with open(
    FINDINGS_PATH,
    "r",
    encoding="utf-8"
) as f:

    for line_number, line in enumerate(f, start=1):

        line = line.strip()

        if not line:
            continue

        try:

            record = json.loads(line)

            findings_data.append(record)

        except json.JSONDecodeError as e:

            raise ValueError(
                f"\nInvalid JSON on line {line_number} "
                f"of findings_fixed.json:\n{e}"
            )


print(
    "Number of label records:",
    len(findings_data)
)


# ============================================================
# CONVERT LABELS TO DATAFRAME
# ============================================================

labels_df = pd.DataFrame(findings_data)

print(
    "Label dataframe shape:",
    labels_df.shape
)

print("\nLabel columns:")
print(labels_df.columns.tolist())


# ============================================================
# FIND IMAGE PATH COLUMN
# ============================================================

possible_path_columns = [
    "o_image",
    "path_to_image",
    "image",
    "image_path"
]

json_path_column = None

for column in possible_path_columns:

    if column in labels_df.columns:

        json_path_column = column
        break


if json_path_column is None:

    raise ValueError(
        "\nCould not find image path column "
        "inside findings JSONL.\n"
        f"Available columns: {labels_df.columns.tolist()}"
    )


labels_df = labels_df.rename(
    columns={
        json_path_column: "path_to_image"
    }
)


# ============================================================
# NORMALIZE PATHS
# ============================================================

def normalize_path(path):

    if pd.isna(path):
        return None

    path = str(path)

    path = path.replace("\\", "/")

    path = path.lstrip("./")

    return path


df["path_to_image"] = (
    df["path_to_image"]
    .apply(normalize_path)
)

labels_df["path_to_image"] = (
    labels_df["path_to_image"]
    .apply(normalize_path)
)


# ============================================================
# CHECK TARGET LABEL
# ============================================================

if TARGET_LABEL not in labels_df.columns:

    raise ValueError(
        f"\nTarget label '{TARGET_LABEL}' "
        "was not found in findings JSONL.\n\n"
        "Available labels:\n"
        + "\n".join(
            labels_df.columns.tolist()
        )
    )


if "No Finding" not in labels_df.columns:

    raise ValueError(
        "\n'No Finding' label was not found "
        "in findings JSONL."
    )


# ============================================================
# KEEP REQUIRED LABEL COLUMNS
# ============================================================

labels = labels_df[
    [
        "path_to_image",
        TARGET_LABEL,
        "No Finding"
    ]
].copy()


# ============================================================
# MERGE CSV + LABELS
# ============================================================

print("\nMerging metadata with labels...")

merged_df = df.merge(
    labels,
    on="path_to_image",
    how="inner"
)

print(
    "Rows after merge:",
    len(merged_df)
)


if len(merged_df) == 0:

    raise ValueError(
        "\nThe CSV and findings JSONL did not match "
        "on path_to_image."
    )


# ============================================================
# REMOVE DUPLICATE IMAGE PATHS
# ============================================================

merged_df = merged_df.drop_duplicates(
    subset=["path_to_image"]
).copy()

print(
    "Rows after removing duplicate images:",
    len(merged_df)
)


# ============================================================
# LABEL POLICY
# ============================================================
#
# CheXpert:
#
# 1  = positive
# 0  = negative
# -1 = uncertain
# NaN = missing
#
# Our first binary model:
#
# PNEUMONIA:
#     Pneumonia == 1
#
# NORMAL:
#     No Finding == 1
#
# Uncertain and missing cases are excluded.
# ============================================================

print("\nCreating binary classification labels...")


# ------------------------------------------------------------
# PNEUMONIA
# ------------------------------------------------------------

pneumonia_df = merged_df[
    merged_df[TARGET_LABEL] == 1
].copy()

pneumonia_df["class_name"] = "PNEUMONIA"


# ------------------------------------------------------------
# NORMAL
# ------------------------------------------------------------

normal_df = merged_df[
    merged_df["No Finding"] == 1
].copy()

normal_df["class_name"] = "NORMAL"


# ============================================================
# REMOVE OVERLAPPING IMAGES
# ============================================================

pneumonia_paths = set(
    pneumonia_df["path_to_image"]
)

normal_df = normal_df[
    ~normal_df["path_to_image"].isin(
        pneumonia_paths
    )
].copy()


# ============================================================
# DISPLAY CLASS COUNTS
# ============================================================

print("\nAvailable classes:")

print(
    "PNEUMONIA:",
    len(pneumonia_df)
)

print(
    "NORMAL:",
    len(normal_df)
)


# ============================================================
# COMBINE
# ============================================================

dataset_df = pd.concat(
    [
        pneumonia_df,
        normal_df
    ],
    ignore_index=True
)


# ============================================================
# USE CHEXPERT PLUS PROVIDED SPLIT
# ============================================================
#
# The metadata CSV already contains a "split" column.
#
# We use:
#     train -> train
#     valid/validation -> val
#
# This avoids creating our own random split.
# ============================================================

print("\nUsing CheXpert Plus provided split...")

print("\nOriginal split counts:")
print(
    dataset_df["split"].value_counts(dropna=False)
)


def normalize_split(value):

    if pd.isna(value):
        return None

    value = str(value).strip().lower()

    if value == "train":
        return "train"

    if value in ["valid", "validation", "val"]:
        return "val"

    return None


dataset_df["model_split"] = (
    dataset_df["split"]
    .apply(normalize_split)
)


# Remove rows that are not train/validation
dataset_df = dataset_df[
    dataset_df["model_split"].notna()
].copy()


# ============================================================
# LIMIT EACH CLASS
# ============================================================

def limit_class(
    dataframe,
    max_images
):

    if (
        max_images is None
        or len(dataframe) <= max_images
    ):

        return dataframe

    return dataframe.sample(
        n=max_images,
        random_state=42
    )


train_pneumonia = dataset_df[
    (dataset_df["model_split"] == "train")
    &
    (dataset_df["class_name"] == "PNEUMONIA")
].copy()

train_normal = dataset_df[
    (dataset_df["model_split"] == "train")
    &
    (dataset_df["class_name"] == "NORMAL")
].copy()

val_pneumonia = dataset_df[
    (dataset_df["model_split"] == "val")
    &
    (dataset_df["class_name"] == "PNEUMONIA")
].copy()

val_normal = dataset_df[
    (dataset_df["model_split"] == "val")
    &
    (dataset_df["class_name"] == "NORMAL")
].copy()


train_pneumonia = limit_class(
    train_pneumonia,
    MAX_IMAGES_PER_CLASS
)

train_normal = limit_class(
    train_normal,
    MAX_IMAGES_PER_CLASS
)

val_pneumonia = limit_class(
    val_pneumonia,
    MAX_IMAGES_PER_CLASS
)

val_normal = limit_class(
    val_normal,
    MAX_IMAGES_PER_CLASS
)


# ============================================================
# FINAL TRAIN / VALIDATION DATAFRAMES
# ============================================================

train_df = pd.concat(
    [
        train_pneumonia,
        train_normal
    ],
    ignore_index=True
)

val_df = pd.concat(
    [
        val_pneumonia,
        val_normal
    ],
    ignore_index=True
)

train_df["split"] = "train"
val_df["split"] = "val"


# ============================================================
# IMAGE PATH RESOLUTION
# ============================================================
#
# CSV/JSON example:
#
# train/patient00001/study1/view1_frontal.jpg
#
# Local PNG example:
#
# patient00001/study1/view1_frontal.png
#
# Therefore we remove the leading "train/" or "valid/"
# when searching inside PNG_train / PNG_valid.
# ============================================================

def find_image_path(
    relative_path,
    split
):

    if pd.isna(relative_path):
        return None

    relative_path = normalize_path(
        relative_path
    )

    if relative_path is None:
        return None


    # Remove dataset split prefix
    prefixes = [
        "train/",
        "valid/",
        "validation/",
        "val/"
    ]

    clean_path = relative_path

    for prefix in prefixes:

        if clean_path.startswith(prefix):

            clean_path = clean_path[
                len(prefix):
            ]

            break


    # Convert extension
    candidates = [
        clean_path,
        clean_path.replace(
            ".jpg",
            ".png"
        ),
        clean_path.replace(
            ".jpeg",
            ".png"
        )
    ]


    # Search appropriate directory first

    if split == "train":

        search_dirs = [
            PNG_TRAIN_DIR,
            PNG_VALID_DIR
        ]

    else:

        search_dirs = [
            PNG_VALID_DIR,
            PNG_TRAIN_DIR
        ]


    for base_dir in search_dirs:

        for candidate in candidates:

            path = os.path.join(
                base_dir,
                candidate
            )

            if os.path.exists(path):

                return path


    return None


# ============================================================
# CHECK LOCAL PNG AVAILABILITY
# ============================================================

print("\nChecking local PNG availability...")

train_df["local_image_path"] = train_df.apply(
    lambda row: find_image_path(
        row["path_to_image"],
        "train"
    ),
    axis=1
)

val_df["local_image_path"] = val_df.apply(
    lambda row: find_image_path(
        row["path_to_image"],
        "val"
    ),
    axis=1
)


train_found = (
    train_df["local_image_path"]
    .notna()
    .sum()
)

val_found = (
    val_df["local_image_path"]
    .notna()
    .sum()
)


print(
    f"Training images found locally: "
    f"{train_found}/{len(train_df)}"
)

print(
    f"Validation images found locally: "
    f"{val_found}/{len(val_df)}"
)


# ============================================================
# SAVE MANIFEST FILES
# ============================================================

print("\nSaving manifest files...")


manifest_columns = [
    "path_to_image",
    "local_image_path",
    "deid_patient_id",
    "age",
    "sex",
    "race",
    "ethnicity",
    "frontal_lateral",
    "ap_pa",
    TARGET_LABEL,
    "No Finding",
    "class_name",
    "split"
]


available_manifest_columns = [
    column
    for column in manifest_columns
    if column in train_df.columns
]


train_manifest_path = os.path.join(
    MANIFEST_DIR,
    "chexpert_train.csv"
)

val_manifest_path = os.path.join(
    MANIFEST_DIR,
    "chexpert_val.csv"
)

complete_manifest_path = os.path.join(
    MANIFEST_DIR,
    "chexpert_binary_dataset.csv"
)


train_df[
    available_manifest_columns
].to_csv(
    train_manifest_path,
    index=False
)

val_df[
    available_manifest_columns
].to_csv(
    val_manifest_path,
    index=False
)


complete_df = pd.concat(
    [
        train_df,
        val_df
    ],
    ignore_index=True
)

complete_df[
    available_manifest_columns
].to_csv(
    complete_manifest_path,
    index=False
)


# ============================================================
# COPY IMAGES
# ============================================================

if COPY_IMAGES:

    if train_found == 0 and val_found == 0:

        print(
            "\nWARNING:"
        )

        print(
            "PNG images were not found locally."
        )

        print(
            "Manifest files were created, "
            "but image training cannot start yet."
        )

        print(
            "Place PNG_train and PNG_valid "
            "inside the data directory."
        )

    else:

        print(
            "\nCopying training images..."
        )

        for _, row in train_df.iterrows():

            source = row[
                "local_image_path"
            ]

            if not source:
                continue

            class_name = row[
                "class_name"
            ]

            destination_dir = os.path.join(
                PREPARED_DIR,
                "train",
                class_name
            )

            os.makedirs(
                destination_dir,
                exist_ok=True
            )

            destination = os.path.join(
                destination_dir,
                os.path.basename(source)
            )

            if not os.path.exists(
                destination
            ):

                shutil.copy2(
                    source,
                    destination
                )


        print(
            "Copying validation images..."
        )

        for _, row in val_df.iterrows():

            source = row[
                "local_image_path"
            ]

            if not source:
                continue

            class_name = row[
                "class_name"
            ]

            destination_dir = os.path.join(
                PREPARED_DIR,
                "val",
                class_name
            )

            os.makedirs(
                destination_dir,
                exist_ok=True
            )

            destination = os.path.join(
                destination_dir,
                os.path.basename(source)
            )

            if not os.path.exists(
                destination
            ):

                shutil.copy2(
                    source,
                    destination
                )


# ============================================================
# FINAL STATISTICS
# ============================================================

print("\n========================================")
print("FINAL DATASET")
print("========================================")


print("\nTRAIN:")

print(
    train_df["class_name"]
    .value_counts()
)


print("\nVALIDATION:")

print(
    val_df["class_name"]
    .value_counts()
)


print(
    "\nUnique training patients:",
    train_df["deid_patient_id"].nunique()
)

print(
    "Unique validation patients:",
    val_df["deid_patient_id"].nunique()
)


# ============================================================
# VERIFY PATIENT LEAKAGE
# ============================================================

train_patient_set = set(
    train_df["deid_patient_id"]
)

val_patient_set = set(
    val_df["deid_patient_id"]
)


overlap = (
    train_patient_set
    &
    val_patient_set
)


print(
    "\nPatient overlap:",
    len(overlap)
)


if len(overlap) != 0:

    raise RuntimeError(
        "PATIENT LEAKAGE DETECTED!"
    )


print(
    "Patient-level split verified."
)


# ============================================================
# FINISHED
# ============================================================

print("\n========================================")
print("DATASET PREPARATION COMPLETE")
print("========================================")


print(
    "\nTrain manifest:"
)

print(
    train_manifest_path
)


print(
    "\nValidation manifest:"
)

print(
    val_manifest_path
)


print(
    "\nComplete manifest:"
)

print(
    complete_manifest_path
)


print(
    "\nPrepared image directory:"
)

print(
    PREPARED_DIR
)


print(
    "\nTarget:",
    TARGET_LABEL
)


print(
    "\nNext step:"
)

print(
    "Run verify_dataset.py"
)

print(
    "Then run train_quick.py"
)


print(
    "\n========================================\n"
)