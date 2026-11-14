
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

# ==================================================
# CONFIGURATION
# ==================================================

MANIFEST_PATH = "data/clinical_text/multimodal_dataset.csv"

OUTPUT_DIR = "ai-services/clinical/outputs/history_only"

os.makedirs(OUTPUT_DIR, exist_ok=True)

# ==================================================
# LOAD DATA
# ==================================================

df = pd.read_csv(MANIFEST_PATH)

required_columns = [
    "patient_id",
    "section_clinical_history",
    "class_name",
    "final_split"
]

for column in required_columns:
    if column not in df.columns:
        raise ValueError(f"Missing column: {column}")

# Use the same label mapping as previous experiments
df["label"] = df["class_name"].map({
    "NORMAL": 0,
    "PNEUMONIA": 1
})

df = df.dropna(
    subset=[
        "patient_id",
        "section_clinical_history",
        "label",
        "final_split"
    ]
).copy()

df["label"] = df["label"].astype(int)
df["section_clinical_history"] = (
    df["section_clinical_history"].astype(str)
)

# Remove empty clinical history
df["section_clinical_history"] = (
    df["section_clinical_history"].str.strip()
)

df = df[
    df["section_clinical_history"] != ""
].copy()

# ==================================================
# PATIENT-LEVEL TRAIN / VALIDATION SPLIT
# ==================================================

train_df = df[df["final_split"] == "train"].copy()
val_df = df[df["final_split"] == "val"].copy()

if train_df.empty or val_df.empty:
    raise ValueError("Training or validation dataset is empty.")

# Verify no patient overlap
overlap = (
    set(train_df["patient_id"])
    & set(val_df["patient_id"])
)

if overlap:
    raise ValueError(
        f"Patient overlap detected: {len(overlap)}"
    )

print("========== DATASET INFORMATION ==========")

print("Total usable records:", len(df))
print("Training records:", len(train_df))
print("Validation records:", len(val_df))
print("Patient overlap:", len(overlap))

print("\nTraining label distribution:")
print(train_df["label"].value_counts().sort_index())

print("\nValidation label distribution:")
print(val_df["label"].value_counts().sort_index())

# ==================================================
# TF-IDF FEATURE EXTRACTION
# Fit only on training data
# ==================================================

vectorizer = TfidfVectorizer(
    max_features=10000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_train = vectorizer.fit_transform(
    train_df["section_clinical_history"]
)

X_val = vectorizer.transform(
    val_df["section_clinical_history"]
)

y_train = train_df["label"].values
y_val = val_df["label"].values

print("\nTraining feature shape:", X_train.shape)
print("Validation feature shape:", X_val.shape)

# ==================================================
# LOGISTIC REGRESSION
# ==================================================

model = LogisticRegression(
    max_iter=2000,
    class_weight="balanced",
    random_state=42
)

model.fit(X_train, y_train)

# ==================================================
# EVALUATION
# ==================================================

probabilities = model.predict_proba(X_val)[:, 1]

predictions = (
    probabilities >= 0.5
).astype(int)

metrics = {
    "model": "Clinical History-Only TF-IDF + Logistic Regression",
    "validation_records": int(len(y_val)),

    "accuracy": float(
        accuracy_score(y_val, predictions)
    ),

    "precision": float(
        precision_score(
            y_val, predictions, zero_division=0
        )
    ),

    "recall": float(
        recall_score(
            y_val, predictions, zero_division=0
        )
    ),

    "f1_score": float(
        f1_score(
            y_val, predictions, zero_division=0
        )
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

print("\n========== HISTORY-ONLY RESULTS ==========")

for key, value in metrics.items():
    print(f"{key}: {value}")

print("\nClassification Report:")
print(
    classification_report(
        y_val,
        predictions,
        zero_division=0
    )
)

# ==================================================
# SAVE OUTPUTS
# ==================================================

joblib.dump(
    vectorizer,
    os.path.join(
        OUTPUT_DIR,
        "history_tfidf_vectorizer.joblib"
    )
)

joblib.dump(
    model,
    os.path.join(
        OUTPUT_DIR,
        "history_only_logistic_regression.joblib"
    )
)

with open(
    os.path.join(OUTPUT_DIR, "history_only_results.json"),
    "w",
    encoding="utf-8"
) as f:
    json.dump(metrics, f, indent=4)

# Save record-level predictions
predictions_df = val_df[
    ["patient_id", "png_path", "label"]
].copy()

predictions_df["predicted_probability"] = probabilities
predictions_df["predicted_label"] = predictions

predictions_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "history_only_predictions.csv"
    ),
    index=False
)

print("\nAll outputs saved to:", OUTPUT_DIR)
print("History-only experiment completed successfully.")
