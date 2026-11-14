import pandas as pd

original = pd.read_csv(
    "data/clinical_text/multimodal_dataset.csv"
)

fusion = pd.read_csv(
    "data/clinical_text/fusion_experiment_dataset.csv"
)

original_val = original[
    original["final_split"].str.lower() == "val"
].copy()

fusion_val = fusion[
    fusion["final_split"].str.lower() == "val"
].copy()

original_ids = set(original_val["patient_id"])
fusion_ids = set(fusion_val["patient_id"])

missing = original_ids - fusion_ids

print("MISSING PATIENTS:", missing)

for patient in missing:
    print("\nMISSING PATIENT DETAILS:")
    print(
        original_val[
            original_val["patient_id"] == patient
        ].to_string(index=False)
    )

# Compare image paths as well
original_paths = set(original_val["png_path"].dropna())
fusion_paths = set(fusion_val["image_path"].dropna())

print("\nMISSING IMAGE PATHS:", original_paths - fusion_paths)