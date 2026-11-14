import pandas as pd

path = "data/clinical_text/multimodal_dataset.csv"
df = pd.read_csv(path)

df["final_split"] = df["final_split"].astype(str).str.lower().str.strip()
df["clinical_text"] = df["clinical_text"].fillna("").astype(str)

train = df[df["final_split"] == "train"].copy()
val = df[df["final_split"] == "val"].copy()

print("TRAIN RECORDS:", len(train))
print("VALIDATION RECORDS:", len(val))

# 1. Patient overlap
train_patients = set(train["patient_id"].dropna())
val_patients = set(val["patient_id"].dropna())

overlap = train_patients & val_patients

print("\nPATIENT OVERLAP:", len(overlap))
if overlap:
    print("Examples:", list(overlap)[:15])

# 2. Exact report duplication across splits
train_reports = set(train["clinical_text"]) - {""}
val_reports = set(val["clinical_text"]) - {""}

common_reports = train_reports & val_reports

print("\nEXACT REPORT OVERLAP:", len(common_reports))

# 3. Image path overlap
for column in ["png_path", "local_path", "image_path"]:
    if column in df.columns:
        train_paths = set(train[column].dropna().astype(str))
        val_paths = set(val[column].dropna().astype(str))
        common_paths = train_paths & val_paths

        print(f"\n{column} OVERLAP:", len(common_paths))

# 4. Pneumonia mentions by label and split
mentions = df["clinical_text"].str.contains(
    r"\bpneumonia\b",
    case=False,
    regex=True
)

print("\nPNEUMONIA MENTIONS BY SPLIT AND LABEL:")
print(
    df[mentions]
    .groupby(["final_split", "class_name"])
    .size()
)

# 5. Multiple records per patient
print("\nREPEATED PATIENTS:")
print("Train:", (train["patient_id"].value_counts() > 1).sum())
print("Validation:", (val["patient_id"].value_counts() > 1).sum())