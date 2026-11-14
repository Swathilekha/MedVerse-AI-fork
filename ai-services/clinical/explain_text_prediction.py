import joblib
import pandas as pd
import numpy as np

MODEL_PATH = "ai-services/clinical/outputs/text_only/text_only_logistic_regression.joblib"
VECTORIZER_PATH = "ai-services/clinical/outputs/text_only/tfidf_vectorizer.joblib"
DATA_PATH = "data/clinical_text/multimodal_dataset.csv"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

df = pd.read_csv(DATA_PATH)

# Select one validation example containing clinical text
sample = df[
    (df["final_split"] == "val") &
    (df["clinical_text"].notna()) &
    (df["clinical_text"].str.strip() != "")
].iloc[0]

report = sample["clinical_text"]
actual_label = int(sample["class_name"] == "PNEUMONIA")

X = vectorizer.transform([report])
probability = model.predict_proba(X)[0]
prediction = int(model.predict(X)[0])

# Positive coefficient contribution indicates PNEUMONIA.
feature_names = vectorizer.get_feature_names_out()
contributions = X.toarray()[0] * model.coef_[0]

active = np.where(X.toarray()[0] != 0)[0]
ranked = sorted(active, key=lambda i: contributions[i], reverse=True)

print("\n========== TEXT EXPLAINABILITY ==========")
print("Patient ID:", sample["patient_id"])
print("Actual:", "PNEUMONIA" if actual_label else "NORMAL")
print("Predicted:", "PNEUMONIA" if prediction else "NORMAL")
print(f"NORMAL probability: {probability[0]:.4f}")
print(f"PNEUMONIA probability: {probability[1]:.4f}")

print("\nTop terms supporting PNEUMONIA:")
for i in ranked[:10]:
    print(f"{feature_names[i]:35s} {contributions[i]:+.5f}")

print("\nTop terms supporting NORMAL:")
for i in sorted(active, key=lambda i: contributions[i])[:10]:
    print(f"{feature_names[i]:35s} {contributions[i]:+.5f}")

print("\n=========================================")
