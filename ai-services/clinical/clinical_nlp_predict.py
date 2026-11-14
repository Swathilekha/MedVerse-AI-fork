import os
import json
import joblib
import pandas as pd


MODEL_PATH = (
    "ai-services/clinical/models/"
    "clinical_evidence_classifier.joblib"
)

INPUT_PATH = (
    "data/clinical_text/"
    "multimodal_dataset.csv"
)

OUTPUT_PATH = (
    "data/clinical_text/"
    "nlp_evidence_predictions.csv"
)


def main():

    print("=" * 70)
    print("CLINICAL NLP EVIDENCE PREDICTION")
    print("=" * 70)

    print("\nLoading model...")

    model = joblib.load(
        MODEL_PATH
    )

    df = pd.read_csv(
        INPUT_PATH,
        low_memory=False
    )

    print(
        "Records:",
        len(df)
    )

    records = []

    for _, row in df.iterrows():

        text = str(
            row.get(
                "clinical_text",
                ""
            )
        )

        if not text or text.lower() == "nan":
            continue

        prediction = model.predict(
            [text]
        )[0]

        probabilities = model.predict_proba(
            [text]
        )[0]

        classes = model.classes_

        probability_dict = {
            cls: float(prob)
            for cls, prob
            in zip(
                classes,
                probabilities
            )
        }

        confidence = max(
            probability_dict.values()
        )

        records.append({
            "patient_id": row["patient_id"],
            "png_path": row["png_path"],
            "class_name": row["class_name"],
            "final_split": row["final_split"],
            "source": "clinical_nlp",
            "status": prediction,
            "confidence": round(
                confidence,
                4
            ),
            "prob_present": round(
                probability_dict.get(
                    "present",
                    0
                ),
                4
            ),
            "prob_absent": round(
                probability_dict.get(
                    "absent",
                    0
                ),
                4
            ),
            "prob_uncertain": round(
                probability_dict.get(
                    "uncertain",
                    0
                ),
                4
            ),
            "clinical_text": text
        })

    result = pd.DataFrame(records)

    result.to_csv(
        OUTPUT_PATH,
        index=False
    )

    print(
        "\nGenerated:",
        len(result),
        "clinical NLP evidence records"
    )

    print("\nStatus distribution:")
    print(
        result["status"]
        .value_counts()
        .to_string()
    )

    print("\nSample:")
    print(
        result[
            [
                "patient_id",
                "status",
                "confidence",
                "prob_present",
                "prob_absent",
                "prob_uncertain"
            ]
        ].head(20).to_string(
            index=False
        )
    )

    print("\nSaved:")
    print(OUTPUT_PATH)


if __name__ == "__main__":
    main()