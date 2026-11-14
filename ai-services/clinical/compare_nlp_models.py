import os
import time
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.calibration import CalibratedClassifierCV
from sklearn.naive_bayes import MultinomialNB

from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report
)


INPUT_PATH = (
    "data/clinical_text/"
    "nlp_training_data.csv"
)


def evaluate_model(name, model, X_train, X_test, y_train, y_test):

    print("\n" + "=" * 70)
    print(name)
    print("=" * 70)

    start = time.time()

    model.fit(
        X_train,
        y_train
    )

    training_time = time.time() - start

    predictions = model.predict(
        X_test
    )

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            y_test,
            predictions,
            average="macro",
            zero_division=0
        )
    )

    print(
        f"\nTraining time: "
        f"{training_time:.2f} seconds"
    )

    print(
        f"Accuracy: {accuracy:.4f}"
    )

    print(
        f"Macro Precision: {precision:.4f}"
    )

    print(
        f"Macro Recall: {recall:.4f}"
    )

    print(
        f"Macro F1: {f1:.4f}"
    )

    print("\nClassification Report:")

    print(
        classification_report(
            y_test,
            predictions,
            zero_division=0
        )
    )

    return {
        "model": name,
        "accuracy": accuracy,
        "macro_precision": precision,
        "macro_recall": recall,
        "macro_f1": f1,
        "training_time_sec": training_time,
        "pipeline": model
    }


def main():

    print("=" * 70)
    print("CLINICAL NLP MODEL COMPARISON")
    print("=" * 70)

    df = pd.read_csv(
        INPUT_PATH
    )

    df = df.dropna(
        subset=[
            "text",
            "label"
        ]
    )

    X = df["text"].astype(str)
    y = df["label"].astype(str)

    print(
        "\nTotal examples:",
        len(df)
    )

    print("\nLabels:")
    print(
        y.value_counts()
    )

    # --------------------------------------------------------
    # SAME SPLIT FOR EVERY MODEL
    # --------------------------------------------------------

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    print(
        "\nTraining examples:",
        len(X_train)
    )

    print(
        "Testing examples:",
        len(X_test)
    )

    models = {

        "TF-IDF + Logistic Regression":

            Pipeline([
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
            ]),

        "TF-IDF + Calibrated Linear SVM":

    Pipeline([
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
            CalibratedClassifierCV(
                LinearSVC(
                    class_weight="balanced"
                ),
                method="sigmoid",
                cv=3
            )
        )
    ]),

        "TF-IDF + Multinomial Naive Bayes":

            Pipeline([
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
                    MultinomialNB()
                )
            ])
    }

    results = []

    for name, model in models.items():

        result = evaluate_model(
            name,
            model,
            X_train,
            X_test,
            y_train,
            y_test
        )

        results.append(
            result
        )

    # --------------------------------------------------------
    # COMPARISON
    # --------------------------------------------------------

    results_df = pd.DataFrame([
        {
            "Model": r["model"],
            "Accuracy": r["accuracy"],
            "Macro Precision": r["macro_precision"],
            "Macro Recall": r["macro_recall"],
            "Macro F1": r["macro_f1"],
            "Training Time (sec)": r["training_time_sec"]
        }
        for r in results
    ])

    print("\n" + "=" * 70)
    print("MODEL COMPARISON")
    print("=" * 70)

    print(
        results_df.to_string(
            index=False
        )
    )

    # --------------------------------------------------------
    # SELECT BY MACRO F1
    # --------------------------------------------------------

    best_index = results_df[
        "Macro F1"
    ].idxmax()

    best_name = results_df.loc[
        best_index,
        "Model"
    ]

    best_result = results[
        best_index
    ]

    print("\n" + "=" * 70)
    print("SELECTED MODEL")
    print("=" * 70)

    print(
        "\nBest Macro-F1 model:",
        best_name
    )

    print(
        f"Macro F1: "
        f"{results_df.loc[best_index, 'Macro F1']:.4f}"
    )

    # --------------------------------------------------------
    # SAVE COMPARISON
    # --------------------------------------------------------

    output_path = (
        "data/clinical_text/"
        "nlp_model_comparison.csv"
    )

    results_df.to_csv(
        output_path,
        index=False
    )

    print(
        "\nComparison saved:",
        output_path
    )

    # --------------------------------------------------------
    # SAVE BEST MODEL
    # --------------------------------------------------------

    model_path = (
        "ai-services/clinical/models/"
        "best_clinical_nlp_classifier.joblib"
    )

    os.makedirs(
        os.path.dirname(model_path),
        exist_ok=True
    )

    joblib.dump(
        best_result["pipeline"],
        model_path
    )

    print(
        "Best model saved:",
        model_path
    )


if __name__ == "__main__":
    main()