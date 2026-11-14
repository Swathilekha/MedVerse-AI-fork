import pandas as pd

original_path = "data/clinical_text/multimodal_dataset.csv"
fusion_path = "data/clinical_text/fusion_experiment_dataset.csv"

original = pd.read_csv(original_path)
fusion = pd.read_csv(fusion_path)

print("ORIGINAL DATASET:", original.shape)
print("FUSION DATASET:", fusion.shape)

print("\nORIGINAL COLUMNS:")
print(original.columns.tolist())

print("\nFUSION COLUMNS:")
print(fusion.columns.tolist())

# Compare patient IDs
original_val = original[
    original["final_split"].astype(str).str.lower() == "val"
].copy()

fusion_val = fusion[
    fusion["final_split"].astype(str).str.lower() == "val"
].copy()

original_patients = set(original_val["patient_id"].dropna())
fusion_patients = set(fusion_val["patient_id"].dropna())

print("\nVALIDATION COUNTS")
print("Original:", len(original_val))
print("Fusion:", len(fusion_val))

print("\nPATIENT ID COMPARISON")
print("Original unique patients:", len(original_patients))
print("Fusion unique patients:", len(fusion_patients))
print("Only in original:", len(original_patients - fusion_patients))
print("Only in fusion:", len(fusion_patients - original_patients))

# Compare image paths
for col in ["png_path", "image_path"]:
    if col in original.columns and col in fusion.columns:
        a = set(original_val[col].dropna().astype(str))
        b = set(fusion_val[col].dropna().astype(str))

        print(f"\n{col} COMPARISON")
        print("Only original:", len(a - b))
        print("Only fusion:", len(b - a))

print("\nFUSION VALIDATION LABEL DISTRIBUTION")
print(fusion_val["label"].value_counts(dropna=False))

print("\nORIGINAL VALIDATION LABEL DISTRIBUTION")
print(original_val["class_name"].value_counts(dropna=False))