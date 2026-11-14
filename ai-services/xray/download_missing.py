import os
import redivis
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

MANIFEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "manifests",
    "chexpert_missing_images.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "chexpert_prepared"
)

TRAIN_TABLE = redivis.table(
    "aimi.chexpert_plus:5yyj:v1_0.png_train:s6cj"
)

VALID_TABLE = redivis.table(
    "aimi.chexpert_plus:5yyj:v1_0.png_valid:41v9"
)


# ============================================================
# START
# ============================================================

print("\n========================================")
print("MEDVERSE AI - DOWNLOAD MISSING IMAGES")
print("========================================")

df = pd.read_csv(MANIFEST_PATH)

print("Missing images:", len(df))

successful = 0
failed = 0


# ============================================================
# DOWNLOAD
# ============================================================

for _, row in df.iterrows():

    split = row["split"]
    class_name = row["class_name"]
    png_path = row["png_path"]

    print("\nDownloading:")
    print(png_path)

    if split == "train":
        table = TRAIN_TABLE
    else:
        table = VALID_TABLE

    destination = os.path.join(
        OUTPUT_DIR,
        split,
        class_name,
        png_path
    )

    os.makedirs(
        os.path.dirname(destination),
        exist_ok=True
    )

    try:

        file = table.file(png_path)

        file.download(destination)

        if os.path.exists(destination):
            print("SUCCESS")
            successful += 1
        else:
            print("FAILED - file not created")
            failed += 1

    except Exception as e:

        print("FAILED:")
        print(e)

        failed += 1


# ============================================================
# SUMMARY
# ============================================================

print("\n========================================")
print("DOWNLOAD COMPLETE")
print("========================================")

print("Successful:", successful)
print("Failed:", failed)

print("========================================")