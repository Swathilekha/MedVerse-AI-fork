
import os
import json
import joblib
import pandas as pd

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# CONFIG
# --------------------------------------------------

MANIFEST_PATH = "data/clinical_text/fusion_experiment_dataset.csv"

OUTPUT_DIR = "ai-services/clinical/outputs/text_only"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --------------------------------------------------
# LOAD DATA
# --------------------------------------------------

df = pd.read_csv(MANIFEST_PATH)

required = [
    "patient_id",
    "clinical_text",
    "label",
    "final_split"
]

for col in required:
    if col not in df.columns:
        raise ValueError(f"Missing column: {col}")

df = df.dropna(
    subset=["patient_id", "clinical_text", "label", "final_split"]
).copy()

df["clinical_text"] = df["clinical_text"].astype(str)
df["label"] = df["label"].astype(int)

train_df = df[df["final_split"] == "train"].copy()
val_df = df[df["final_split"] == "val"].copy()

if train_df.empty or val_df.empty:
    raise ValueError("Training or validation data is empty.")

# Verify patient-level separation
overlap = set(train_df["patient_id"]) & set(val_df["patient_id"])

if overlap:
    raise ValueError(
        f"Patient leakage detected: {len(overlap)} overlapping patients."
    )

print("Training records:", len(train_df))
print("Validation records:", len(val_df))
print("Patient overlap:", len(overlap))

print("\nTraining label distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation label distribution:")
print(val_df["label"].value_counts().sort_index())

# --------------------------------------------------
# TF-IDF
# Fit ONLY on training text
# --------------------------------------------------

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(
    train_df["clinical_text"]
)

X_val = vectorizer.transform(
    val_df["clinical_text"]
)

y_train = train_df["label"].values
y_val = val_df["label"].values

print("\nTraining feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)

# --------------------------------------------------
# TRAIN LOGISTIC REGRESSION
# --------------------------------------------------

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# --------------------------------------------------
# EVALUATION
# --------------------------------------------------

probabilities = model.predict_proba(X_val)[:, 1]
predictions = (probabilities >= 0.5).astype(int)

metrics = {
    "model": "Clinical Text-Only TF-IDF + Logistic Regression",
    "validation_records": int(len(y_val)),
    "accuracy": float(accuracy_score(y_val, predictions)),
    "precision": float(
        precision_score(y_val, predictions, zero_division=0)
    ),
    "recall": float(
        recall_score(y_val, predictions, zero_division=0)
    ),
    "f1_score": float(
        f1_score(y_val, predictions, zero_division=0)
    ),
    "roc_auc": float(
        roc_auc_score(y_val, probabilities)
    ),
    "pr_auc": float(
        average_precision_score(y_val, probabilities)
    ),
    "balanced_accuracy": float(
        balanced_accuracy_score(y_val, predictions)
    ),
    "confusion_matrix": confusion_matrix(
        y_val, predictions
    ).tolist()
}

print("\n========== TEXT-ONLY RESULTS ==========")

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nClassification Report:")
print(classification_report(y_val, predictions, zero_division=0))

# --------------------------------------------------
# SAVE MODEL AND RESULTS
# --------------------------------------------------

joblib.dump(
    vectorizer,
    os.path.join(OUTPUT_DIR, "tfidf_vectorizer.joblib")
)

joblib.dump(
    model,
    os.path.join(OUTPUT_DIR, "text_only_logistic_regression.joblib")
)

with open(
    os.path.join(OUTPUT_DIR, "text_only_results.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4)

# Save record-level predictions for later comparison
predictions_df = val_df[
    ["patient_id", "image_path", "label"]
].copy()

predictions_df["predicted_probability"] = probabilities
predictions_df["predicted_label"] = predictions

predictions_df.to_csv(
    os.path.join(OUTPUT_DIR, "text_only_predictions.csv"),
    index=False
)

print("\nSaved all outputs to:", OUTPUT_DIR)
print("Text-only experiment completed.")
