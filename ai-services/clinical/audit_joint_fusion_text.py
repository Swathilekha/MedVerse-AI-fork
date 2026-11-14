import pandas as pd
import re

path = "data/clinical_text/multimodal_dataset.csv"
df = pd.read_csv(path)

val = df[df["final_split"].str.lower() == "val"].copy()

print("VALIDATION RECORDS:", len(val))
print("\nLABEL DISTRIBUTION:")
print(val["class_name"].value_counts(dropna=False))

keywords = [
    "pneumonia",
    "no pneumonia",
    "without pneumonia",
    "pneumonia is present",
    "diagnosis",
    "impression",
    "consolidation",
    "infiltrate"
]

for keyword in keywords:
    count = val["clinical_text"].fillna("").str.contains(
        re.escape(keyword), case=False, regex=True
    ).sum()

    print(f"{keyword:30s}: {count}")

print("\n" + "=" * 65)
print("SAMPLE VALIDATION RECORDS")
print("=" * 65)

for _, row in val.head(10).iterrows():
    print("\nPatient:", row.get("patient_id"))
    print("Label:", row.get("class_name"))
    print("Clinical text:")
    print(str(row.get("clinical_text", ""))[:700])
    print("-" * 65)