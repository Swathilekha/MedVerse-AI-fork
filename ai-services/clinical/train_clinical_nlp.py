import os
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix
)


INPUT_PATH = "data/clinical_text/nlp_training_data.csv"

MODEL_DIR = "ai-services/clinical/models"
MODEL_PATH = os.path.join(
    MODEL_DIR,
    "clinical_evidence_classifier.joblib"
)


def main():

    print("=" * 70)
    print("TRAINING CLINICAL NLP MODEL")
    print("=" * 70)

    df = pd.read_csv(INPUT_PATH)

    print("\nExamples:", len(df))

    # Remove invalid rows.
    df = df.dropna(
        subset=["text", "label"]
    )

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    print("\nLabel distribution:")
    print(y.value_counts())

    # Stratified split.
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print("\nTrain:", len(X_train))
    print("Test :", len(X_test))

    # ========================================================
    # TF-IDF + Logistic Regression
    # ========================================================

    model = Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                lowercase=True,
                ngram_range=(1, 2),
                min_df=2,
                max_df=0.98,
                sublinear_tf=True,
                max_features=50000
            )
        ),
        (
            "classifier",
            LogisticRegression(
                max_iter=1000,
                class_weight="balanced"
            )
        )
    ])

    print("\nTraining...")

    model.fit(
        X_train,
        y_train
    )

    print("Training complete.")

    # ========================================================
    # EVALUATION
    # ========================================================

    predictions = model.predict(X_test)

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test,
        predictions,
        average="weighted",
        zero_division=0
    )

    print("\n" + "=" * 70)
    print("NLP MODEL RESULTS")
    print("=" * 70)

    print(
        f"\nAccuracy : {accuracy:.4f}"
    )

    print(
        f"Precision: {precision:.4f}"
    )

    print(
        f"Recall   : {recall:.4f}"
    )

    print(
        f"F1-score : {f1:.4f}"
    )

    print("\nClassification Report:")
    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    print("Confusion Matrix:")
    print(
        confusion_matrix(
            y_test,
            predictions
        )
    )

    # ========================================================
    # SAVE MODEL
    # ========================================================

    os.makedirs(
        MODEL_DIR,
        exist_ok=True
    )

    joblib.dump(
        model,
        MODEL_PATH
    )

    print("\nModel saved:")
    print(MODEL_PATH)

    # ========================================================
    # SAMPLE PREDICTIONS
    # ========================================================

    samples = [
        "NO EVIDENCE OF PNEUMOTHORAX.",
        "SMALL LEFT APICAL PNEUMOTHORAX.",
        "POSSIBLE SMALL LEFT PLEURAL EFFUSION.",
        "ATELECTASIS VERSUS PNEUMONIA.",
        "AIRSPACE OPACITIES CONCERNING FOR PNEUMONIA.",
        "NO FOCAL CONSOLIDATION OR PLEURAL EFFUSION."
    ]

    print("\n" + "=" * 70)
    print("SAMPLE PREDICTIONS")
    print("=" * 70)

    probabilities = model.predict_proba(samples)
    classes = model.classes_

    for text, prediction, probs in zip(
        samples,
        model.predict(samples),
        probabilities
    ):

        confidence = max(probs)

        print("\nText:")
        print(text)

        print(
            f"Prediction: {prediction}"
        )

        print(
            f"Confidence: {confidence:.4f}"
        )

        print(
            "Probabilities:",
            {
                cls: round(float(prob), 4)
                for cls, prob
                in zip(classes, probs)
            }
        )


if __name__ == "__main__":
    main()