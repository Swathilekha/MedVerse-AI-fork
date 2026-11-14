import pandas as pd

path = "data/clinical_text/fusion_conflict_analysis.csv"
df = pd.read_csv(path)

print("VALIDATION RECORDS:", len(df))

print("\nDEFERRAL REASONS:")
print(df["defer_reason"].value_counts())

deferred = df[df["defer_reason"] != "NONE"]
accepted = df[df["defer_reason"] == "NONE"]

print("\nDeferred cases:", len(deferred))
print("Accepted cases:", len(accepted))

print("Coverage:", round(len(accepted) / len(df), 4))

for name, subset in [
    ("Deferred", deferred),
    ("Accepted", accepted)
]:
    if len(subset):
        predictions = (
            subset["aarf_v3_probability"] >= 0.5
        ).astype(int)

        actual = (
            subset["true_class"] == "PNEUMONIA"
        ).astype(int)

        error_rate = (predictions != actual).mean()

        print(f"\n{name} error rate: {error_rate:.4f}")

print("\nEVIDENCE INTERPRETATION:")
print(df["evidence_interpretation"].value_counts())

print("\nSEMANTIC CONFLICTS:")
print(df["semantic_conflict"].value_counts())