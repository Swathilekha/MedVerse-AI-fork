import os
import pandas as pd


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

MANIFEST_DIR = os.path.join(
    BASE_DIR,
    "data",
    "manifests"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "manifests"
)

TRAIN_MANIFEST = os.path.join(
    MANIFEST_DIR,
    "chexpert_train.csv"
)

VAL_MANIFEST = os.path.join(
    MANIFEST_DIR,
    "chexpert_val.csv"
)

OUTPUT_CSV = os.path.join(
    OUTPUT_DIR,
    "chexpert_png_download_list.csv"
)

OUTPUT_TXT = os.path.join(
    OUTPUT_DIR,
    "chexpert_png_download_list.txt"
)


# ============================================================
# START
# ============================================================

print("\n========================================")
print("CHEXPERT PLUS PNG DOWNLOAD LIST")
print("========================================")


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(TRAIN_MANIFEST):
    raise FileNotFoundError(
        f"Training manifest not found:\n{TRAIN_MANIFEST}"
    )

if not os.path.exists(VAL_MANIFEST):
    raise FileNotFoundError(
        f"Validation manifest not found:\n{VAL_MANIFEST}"
    )


# ============================================================
# LOAD MANIFESTS
# ============================================================

print("\nLoading training manifest...")

train_df = pd.read_csv(TRAIN_MANIFEST)

print(
    "Training rows:",
    len(train_df)
)

print("\nLoading validation manifest...")

val_df = pd.read_csv(VAL_MANIFEST)

print(
    "Validation rows:",
    len(val_df)
)


# ============================================================
# CREATE DOWNLOAD LIST
# ============================================================

download_rows = []


# ------------------------------------------------------------
# TRAINING IMAGES
# ------------------------------------------------------------

for _, row in train_df.iterrows():

    original_path = str(row["path_to_image"])

    # Example:
    # train/patient00403/study2/view1_frontal.jpg

    png_path = original_path.replace(
        ".jpg",
        ".png"
    ).replace(
        ".jpeg",
        ".png"
    )

    # CheXpert Plus Files page uses:
    #
    # patient00403/study2/view1_frontal.png
    #
    # without the train/ prefix.

    png_path = png_path.replace(
        "train/",
        "",
        1
    )

    download_rows.append({
        "split": "train",
        "class_name": row["class_name"],
        "png_path": png_path
    })


# ------------------------------------------------------------
# VALIDATION IMAGES
# ------------------------------------------------------------

for _, row in val_df.iterrows():

    original_path = str(row["path_to_image"])

    png_path = original_path.replace(
        ".jpg",
        ".png"
    ).replace(
        ".jpeg",
        ".png"
    )

    png_path = png_path.replace(
        "valid/",
        "",
        1
    )

    download_rows.append({
        "split": "valid",
        "class_name": row["class_name"],
        "png_path": png_path
    })


# ============================================================
# CREATE DATAFRAME
# ============================================================

download_df = pd.DataFrame(
    download_rows
)


# ============================================================
# REMOVE ACCIDENTAL DUPLICATES
# ============================================================

before = len(download_df)

download_df = download_df.drop_duplicates(
    subset=["png_path"]
).reset_index(drop=True)

after = len(download_df)


if before != after:

    print(
        f"\nRemoved {before - after} duplicate images."
    )


# ============================================================
# SORT
# ============================================================

download_df = download_df.sort_values(
    by=[
        "split",
        "class_name",
        "png_path"
    ]
).reset_index(drop=True)


# ============================================================
# SAVE CSV
# ============================================================

download_df.to_csv(
    OUTPUT_CSV,
    index=False
)


# ============================================================
# SAVE TXT
# ============================================================

with open(
    OUTPUT_TXT,
    "w",
    encoding="utf-8"
) as f:

    for path in download_df["png_path"]:

        f.write(
            path + "\n"
        )


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("DOWNLOAD LIST CREATED")
print("========================================")

print(
    "\nTotal PNG files:",
    len(download_df)
)

print("\nBy split:")

print(
    download_df["split"].value_counts()
)


print("\nBy class:")

print(
    download_df["class_name"].value_counts()
)


print("\nTraining classes:")

print(
    download_df[
        download_df["split"] == "train"
    ]["class_name"].value_counts()
)


print("\nValidation classes:")

print(
    download_df[
        download_df["split"] == "valid"
    ]["class_name"].value_counts()
)


print("\nFirst 20 files:")

print(
    download_df.head(20).to_string(
        index=False
    )
)


print("\n----------------------------------------")

print(
    "CSV:",
    OUTPUT_CSV
)

print(
    "TXT:",
    OUTPUT_TXT
)

print("----------------------------------------")

print(
    "\nExample CheXpert Plus search path:"
)

print(
    download_df.iloc[0]["png_path"]
)

print(
    "\n========================================"
)
print("COMPLETE")
print("========================================")