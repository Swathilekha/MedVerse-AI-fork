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

MANIFEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "manifests",
    "chexpert_png_download_list.csv"
)

IMAGE_DIR = os.path.join(
    BASE_DIR,
    "data",
    "chexpert_prepared"
)


# ============================================================
# HEADER
# ============================================================

print("\n========================================")
print("MEDVERSE AI - CHEXPERT DATASET VERIFY")
print("========================================")


# ============================================================
# CHECK FILES
# ============================================================

if not os.path.exists(MANIFEST_PATH):
    raise FileNotFoundError(
        f"Manifest not found:\n{MANIFEST_PATH}"
    )

if not os.path.exists(IMAGE_DIR):
    raise FileNotFoundError(
        f"Image directory not found:\n{IMAGE_DIR}"
    )


# ============================================================
# LOAD MANIFEST
# ============================================================

print("\nLoading download manifest...")

df = pd.read_csv(MANIFEST_PATH)

print("Manifest rows:", len(df))


# ============================================================
# CHECK DOWNLOADED IMAGES
# ============================================================

print("\nChecking downloaded images...")

found = []
missing = []

for _, row in df.iterrows():

    split = row["split"]
    class_name = row["class_name"]
    png_path = row["png_path"]

    # --------------------------------------------------------
    # Images are stored as:
    #
    # chexpert_prepared/
    #   train/
    #       NORMAL/
    #           patientXXXX/
    #               studyX/
    #                   image.png
    #
    # --------------------------------------------------------

    local_path = os.path.join(
        IMAGE_DIR,
        split,
        class_name,
        png_path
    )

    if os.path.exists(local_path):
        found.append(row)
    else:
        missing.append(row)


# ============================================================
# RESULTS
# ============================================================

print("\n========================================")
print("VERIFICATION RESULTS")
print("========================================")

print("Expected images:", len(df))
print("Found images:", len(found))
print("Missing images:", len(missing))


# ============================================================
# MISSING FILES
# ============================================================

if missing:

    print("\nFirst missing files:")

    for row in missing[:20]:

        print(
            f"  {row['split']} / "
            f"{row['class_name']} / "
            f"{row['png_path']}"
        )


# ============================================================
# DOWNLOADED CLASS COUNTS
# ============================================================

if found:

    found_df = pd.DataFrame(found)

    print("\n========================================")
    print("DOWNLOADED CLASS COUNTS")
    print("========================================")

    print(
        found_df.groupby(
            ["split", "class_name"]
        ).size()
    )


# ============================================================
# DOWNLOAD PERCENTAGE
# ============================================================

percentage = (
    len(found) / len(df) * 100
)

print("\nDownload completion:")
print(f"{percentage:.2f}%")


# ============================================================
# FINAL STATUS
# ============================================================

print("\n========================================")

if len(missing) == 0:

    print("DATASET VERIFICATION SUCCESSFUL")
    print("All images are available locally.")

else:

    print(
        f"DATASET VERIFICATION: "
        f"{len(missing)} IMAGE(S) MISSING"
    )

print("========================================")


# ============================================================
# SAVE MISSING LIST
# ============================================================

if missing:

    missing_df = pd.DataFrame(missing)

    missing_path = os.path.join(
        BASE_DIR,
        "data",
        "manifests",
        "chexpert_missing_images.csv"
    )

    missing_df.to_csv(
        missing_path,
        index=False
    )

    print(
        "\nMissing-image list saved to:"
    )
    print(missing_path)


print("\n========================================")
print("VERIFICATION COMPLETE")
print("========================================")