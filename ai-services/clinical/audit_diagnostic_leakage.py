import pandas as pd
import re

path = "data/clinical_text/multimodal_dataset.csv"
df = pd.read_csv(path)

val = df[df["final_split"].str.lower() == "val"].copy()

patterns = {
    "explicit_pneumonia": r"\bpneumonia\b",
    "no_pneumonia": r"\bno evidence of pneumonia\b|\bwithout pneumonia\b",
    "positive_pneumonia": r"\b(pneumonia|pneumonic)\b",
    "diagnosis_section": r"\bdiagnosis\s*:",
    "normal_statement": r"\b(normal chest|lungs are clear|no focal consolidation)\b",
    "negative_infection": r"\b(no acute cardiopulmonary|no focal airspace disease)\b"
}

for name, pattern in patterns.items():
    matches = val["clinical_text"].fillna("").str.contains(
        pattern, flags=re.IGNORECASE, regex=True
    )
    print(f"{name:25s}: {matches.sum()}")

print("\nPNEUMONIA WORD OCCURRENCES BY LABEL")
mask = val["clinical_text"].fillna("").str.contains(
    r"\bpneumonia\b", case=False, regex=True
)

print(val.loc[mask, "class_name"].value_counts())

print("\nSAMPLE REPORTS CONTAINING PNEUMONIA")
for _, row in val.loc[mask].head(15).iterrows():
    print("\nPatient:", row["patient_id"])
    print("Label:", row["class_name"])
    print("Report:", str(row["clinical_text"])[:1000])
    print("-" * 60)