import os
import time
import pandas as pd
import redivis


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
    "chexpert_prepared"
)

DOWNLOAD_LIST = os.path.join(
    MANIFEST_DIR,
    "chexpert_png_download_list.csv"
)


# ============================================================
# REDIVIS TABLES
# ============================================================

TRAIN_TABLE_ID = (
    "aimi.chexpert_plus:5yyj:v1_0.png_train:s6cj"
)

VALID_TABLE_ID = (
    "aimi.chexpert_plus:5yyj:v1_0.png_valid:41v9"
)


# ============================================================
# START
# ============================================================

print("\n========================================")
print("CHEXPERT PLUS IMAGE DOWNLOADER")
print("========================================")


# ============================================================
# LOAD MANIFEST
# ============================================================

if not os.path.exists(DOWNLOAD_LIST):

    raise FileNotFoundError(
        f"Download list not found:\n{DOWNLOAD_LIST}"
    )

print("\nLoading download list...")

df = pd.read_csv(DOWNLOAD_LIST)

print("Total files:", len(df))

print("\nBy split:")
print(df["split"].value_counts())

print("\nBy class:")
print(df["class_name"].value_counts())


# ============================================================
# CREATE OUTPUT DIRECTORIES
# ============================================================

for split in ["train", "valid"]:

    for class_name in ["NORMAL", "PNEUMONIA"]:

        os.makedirs(
            os.path.join(
                OUTPUT_DIR,
                split,
                class_name
            ),
            exist_ok=True
        )


# ============================================================
# CONNECT TO REDIVIS
# ============================================================

print("\nConnecting to Redivis...")

train_table = redivis.table(
    TRAIN_TABLE_ID
)

valid_table = redivis.table(
    VALID_TABLE_ID
)

print("Redivis tables connected.")


# ============================================================
# DOWNLOAD
# ============================================================

successful = 0
skipped = 0
failed = 0

total = len(df)

start_time = time.time()


for index, row in df.iterrows():

    split = row["split"]
    class_name = row["class_name"]
    png_path = row["png_path"]

    # --------------------------------------------------------
    # Select correct Redivis table
    # --------------------------------------------------------

    if split == "train":
        table = train_table

    elif split == "valid":
        table = valid_table

    else:
        print(
            f"[{index + 1}/{total}] "
            f"Unknown split: {split}"
        )

        failed += 1
        continue


    # --------------------------------------------------------
    # IMPORTANT:
    # Preserve the complete patient/study/view path.
    #
    # Example:
    #
    # patient00011/study8/view1_frontal.png
    #
    # becomes:
    #
    # train/NORMAL/patient00011/study8/view1_frontal.png
    #
    # This prevents filename collisions.
    # --------------------------------------------------------

    relative_path = png_path.replace(
        "/",
        os.sep
    )

    destination = os.path.join(
        OUTPUT_DIR,
        split,
        class_name,
        relative_path
    )


    # --------------------------------------------------------
    # Create patient/study directory
    # --------------------------------------------------------

    os.makedirs(
        os.path.dirname(destination),
        exist_ok=True
    )


    # --------------------------------------------------------
    # Skip if already downloaded correctly
    # --------------------------------------------------------

    if os.path.exists(destination):

        skipped += 1

        print(
            f"[{index + 1}/{total}] "
            f"EXISTS: {png_path}"
        )

        continue


    # --------------------------------------------------------
    # Download
    # --------------------------------------------------------

    print(
        f"[{index + 1}/{total}] "
        f"DOWNLOADING: {png_path}"
    )

    try:

        file = table.file(
            png_path
        )

        file.download(
            destination
        )

        if os.path.exists(destination):

            successful += 1

            print("  OK")

        else:

            failed += 1

            print(
                "  FAILED: file was not created"
            )

    except Exception as e:

        failed += 1

        print(
            f"  FAILED: {e}"
        )

    time.sleep(0.1)


# ============================================================
# FINAL REPORT
# ============================================================

elapsed = time.time() - start_time


print("\n========================================")
print("DOWNLOAD COMPLETE")
print("========================================")

print(
    f"\nSuccessful: {successful}"
)

print(
    f"Already existed: {skipped}"
)

print(
    f"Failed: {failed}"
)

print(
    f"Total: {total}"
)

print(
    f"Time: {elapsed / 60:.2f} minutes"
)


# ============================================================
# VERIFY ACTUAL IMAGE COUNTS
# ============================================================

print("\n========================================")
print("DOWNLOADED IMAGE COUNTS")
print("========================================")


for split in ["train", "valid"]:

    print(f"\n{split.upper()}")

    split_total = 0

    for class_name in ["NORMAL", "PNEUMONIA"]:

        directory = os.path.join(
            OUTPUT_DIR,
            split,
            class_name
        )

        count = 0

        if os.path.exists(directory):

            for root, dirs, files in os.walk(directory):

                count += sum(
                    1
                    for f in files
                    if f.lower().endswith(
                        (".png", ".jpg", ".jpeg")
                    )
                )

        print(
            f"  {class_name}: {count}"
        )

        split_total += count

    print(
        f"  Total: {split_total}"
    )


print("\n========================================")
print("COMPLETE")
print("========================================")