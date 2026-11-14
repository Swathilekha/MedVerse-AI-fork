import pandas as pd

path = "data/clinical_text/multimodal_dataset.csv"
df = pd.read_csv(path)

df["final_split"] = df["final_split"].astype(str).str.lower().str.strip()
df["clinical_text"] = df["clinical_text"].fillna("").astype(str)

train = df[df["final_split"] == "train"]
val = df[df["final_split"] == "val"]

train_reports = set(train["clinical_text"]) - {""}
val_reports = set(val["clinical_text"]) - {""}

common = train_reports & val_reports

print("DUPLICATED REPORTS:", len(common))

for i, report in enumerate(common, 1):
    print(f"\n{'=' * 70}")
    print(f"DUPLICATE REPORT {i}")
    print(report[:1000])

    matches = df[df["clinical_text"] == report]

    columns = [
        c for c in [
            "patient_id", "class_name", "final_split",
            "png_path", "local_path"
        ] if c in matches.columns
    ]

    print(matches[columns].to_string(index=False))