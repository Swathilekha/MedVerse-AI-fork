from pathlib import Path

import pandas as pd
import joblib

from sklearn.feature_extraction.text import TfidfVectorizer


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "clinical_text"
    / "fusion_experiment_dataset.csv"
)

OUTPUT_FEATURES = (
    PROJECT_ROOT
    / "data"
    / "clinical_text"
    / "early_fusion_text_features.joblib"
)

OUTPUT_VECTORIZER = (
    PROJECT_ROOT
    / "data"
    / "clinical_text"
    / "early_fusion_tfidf_vectorizer.joblib"
)


# ============================================================
# LOAD DATA
# ============================================================

print("=" * 70)
print("EARLY FUSION — CLINICAL TEXT FEATURE EXTRACTION")
print("=" * 70)

df = pd.read_csv(MANIFEST)

print(f"\nTotal records: {len(df)}")


# ============================================================
# TEXT
# ============================================================

texts = (
    df["clinical_text"]
    .fillna("")
    .astype(str)
)


# ============================================================
# TF-IDF
# ============================================================

print("\nBuilding TF-IDF representation...")

vectorizer = TfidfVectorizer(
    max_features=5000,
    ngram_range=(1, 2),
    min_df=2,
    sublinear_tf=True
)

X_text = vectorizer.fit_transform(texts)


print("\nTF-IDF created.")

print(
    f"TF-IDF shape: {X_text.shape}"
)

print(
    f"Vocabulary size: {len(vectorizer.vocabulary_)}"
)


# ============================================================
# SAVE
# ============================================================

joblib.dump(
    X_text,
    OUTPUT_FEATURES
)

joblib.dump(
    vectorizer,
    OUTPUT_VECTORIZER
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("TEXT FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"\nText feature matrix: {X_text.shape}"
)

print(
    f"Features saved to:\n{OUTPUT_FEATURES}"
)

print(
    f"\nVectorizer saved to:\n{OUTPUT_VECTORIZER}"
)

print("\nDone.")