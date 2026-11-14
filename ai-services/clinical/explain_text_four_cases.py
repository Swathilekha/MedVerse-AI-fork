import os
import numpy as np
import pandas as pd
import joblib

BASE = "ai-services/clinical"
MODEL_PATH = f"{BASE}/outputs/text_only/text_only_logistic_regression.joblib"
VECTORIZER_PATH = f"{BASE}/outputs/text_only/tfidf_vectorizer.joblib"
PREDICTIONS_PATH = f"{BASE}/outputs/common_723/text_only_common_predictions.csv"
DATA_PATH = "data/clinical_text/multimodal_dataset.csv"

model = joblib.load(MODEL_PATH)
vectorizer = joblib.load(VECTORIZER_PATH)

predictions = pd.read_csv(PREDICTIONS_PATH)
data = pd.read_csv(DATA_PATH)

cases = {
    "TN": ("patient00040", 0, 0),
    "TP": ("patient01369", 1, 1),
    "FP": ("patient05750", 0, 1),
    "FN": ("patient08478", 1, 0)
}

features = vectorizer.get_feature_names_out()
results = []

for case, (patient_id, actual, predicted) in cases.items():

    rows = data[data["patient_id"] == patient_id]

    if rows.empty:
        print("Missing patient:", patient_id)
        continue

    report = rows.iloc[0]["clinical_text"]

    if pd.isna(report) or not str(report).strip():
        print("Missing clinical text:", patient_id)
        continue

    X = vectorizer.transform([str(report)])
    contributions = X.toarray()[0] * model.coef_[0]
    active = np.where(X.toarray()[0] != 0)[0]

    positive = sorted(active, key=lambda i: contributions[i], reverse=True)[:10]
    negative = sorted(active, key=lambda i: contributions[i])[:10]

    probability = model.predict_proba(X)[0][1]

    def format_terms(indices):
        return "; ".join(
            f"{features[i]} ({contributions[i]:+.5f})"
            for i in indices
        )

    results.append({
        "case": case,
        "patient_id": patient_id,
        "actual_label": actual,
        "predicted_label": predicted,
        "pneumonia_probability": probability,
        "top_pneumonia_supporting_terms": format_terms(positive),
        "top_normal_supporting_terms": format_terms(negative)
    })

    print(f"\n{case} - {patient_id}")
    print("Pneumonia probability:", round(probability, 4))
    print("Pneumonia-supporting:", format_terms(positive))
    print("Normal-supporting:", format_terms(negative))

output = f"{BASE}/outputs/text_only/text_explainability_four_cases.csv"
pd.DataFrame(results).to_csv(output, index=False)

print("\nSaved:", output)
print("Cases generated:", len(results))
