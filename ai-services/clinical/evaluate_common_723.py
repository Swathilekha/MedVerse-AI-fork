
from pathlib import Path
import json
import pandas as pd

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix,
)

ROOT = Path(__file__).resolve().parents[2]

MULTIMODAL = ROOT / "data/clinical_text/multimodal_dataset.csv"
FUSION_MANIFEST = ROOT / "data/clinical_text/fusion_experiment_dataset.csv"

OUTPUT_DIR = ROOT / "ai-services/clinical/outputs"
COMMON_DIR = OUTPUT_DIR / "common_723"
COMMON_DIR.mkdir(parents=True, exist_ok=True)

# Update filenames here only if your actual filenames differ.
FILES = {
    "X-ray-only": ROOT / "ai-services/xray/architecture_comparison/densenet121_common_val_predictions.csv",

    "Text-only": OUTPUT_DIR / "text_only/text_only_predictions.csv",

    "History-only": OUTPUT_DIR / "history_only/history_only_predictions.csv",

    "Early Fusion Best": OUTPUT_DIR / "evaluation_predictions/EarlyFusion_Best_predictions.csv",

    "Early Fusion Continued": OUTPUT_DIR / "evaluation_predictions/EarlyFusion_Continued_predictions.csv",

    "Joint Fusion": OUTPUT_DIR / "joint_fusion_text_predictions_20261103_151843.csv",
}


def normalize_path(value):
    """Convert absolute and relative image paths to a common format."""
    if pd.isna(value):
        return ""

    value = str(value).replace("\\", "/").strip().lower()

    # Extract path beginning with patient ID.
    index = value.rfind("patient")

    if index == -1:
        return value

    return value[index:]


def find_column(df, candidates):
    for col in candidates:
        if col in df.columns:
            return col
    return None


def load_common_cohort():
    multi = pd.read_csv(MULTIMODAL)
    manifest = pd.read_csv(FUSION_MANIFEST)

    multi["history_clean"] = (
        multi["section_clinical_history"]
        .fillna("")
        .astype(str)
        .str.strip()
    )

    multi["split_clean"] = (
        multi["final_split"].astype(str).str.strip().str.lower()
    )

    common = multi[
        (multi["split_clean"] == "val")
        & (multi["history_clean"] != "")
    ].copy()

    common["path_key"] = common["png_path"].apply(normalize_path)
    manifest["path_key"] = manifest["image_path"].apply(normalize_path)

    common = common.merge(
        manifest[["path_key", "label"]],
        on="path_key",
        how="inner",
        validate="one_to_one",
    )

    common = common.drop_duplicates("path_key").reset_index(drop=True)

    print("\nCOMMON VALIDATION COHORT")
    print("Matched records:", len(common))
    print("Unique patients:", common["patient_id"].nunique())
    print("\nLabel distribution:")
    print(common["label"].value_counts().sort_index())

    if len(common) == 0:
        raise ValueError(
            "No common records matched. Check the image paths."
        )

    common[["patient_id", "path_key", "label"]].to_csv(
        COMMON_DIR / "common_723_manifest.csv",
        index=False,
    )

    return common


def load_predictions(name, file_path, common):

    if not file_path.exists():
        print(f"\n[SKIP] Missing file: {file_path}")
        return None

    pred = pd.read_csv(file_path)

    print(f"\nChecking {name}")
    print("File:", file_path.name)
    print("Rows:", len(pred))
    print("Columns:", pred.columns.tolist())

    path_col = find_column(
        pred,
        ["image_path", "png_path", "local_path", "path_key"],
    )

    prob_col = find_column(
        pred,
        [
            "predicted_probability",
            "probability",
            "probability_pneumonia",
            "pneumonia_probability",
            "positive_probability",
            "original_probability",
        ],
    )

    if prob_col is None:
        print(f"[SKIP] No probability column found for {name}.")
        return None

    # If image path exists, use it for exact alignment.
    if path_col is not None:

        pred["path_key"] = pred[path_col].apply(normalize_path)

    else:
        # Only use row-order alignment for a full 1,188-record
        # validation prediction file.
        if len(pred) != 1188:
            print(
                f"[SKIP] {name}: no image path and row count is "
                f"{len(pred)}, not 1188."
            )
            return None

        ordered_val = pd.read_csv(FUSION_MANIFEST)

        ordered_val = ordered_val[
            ordered_val["final_split"].astype(str).str.lower() == "val"
        ].reset_index(drop=True)

        if len(ordered_val) != len(pred):
            print(f"[SKIP] {name}: validation row count mismatch.")
            return None

        pred["path_key"] = ordered_val["image_path"].apply(normalize_path)

    if pred["path_key"].duplicated().any():
        print(f"[SKIP] {name}: duplicate image paths in predictions.")
        return None

    # Restrict predictions to the same common validation records.
    merged = common[
        ["patient_id", "path_key", "label"]
    ].merge(
        pred[["path_key", prob_col]],
        on="path_key",
        how="left",
        validate="one_to_one",
    )

    if merged[prob_col].isna().any():
        missing = int(merged[prob_col].isna().sum())
        print(f"[SKIP] {name}: missing predictions for {missing} records.")
        return None

    y_true = merged["label"].astype(int).to_numpy()
    y_prob = merged[prob_col].astype(float).to_numpy()

    if not ((y_prob >= 0) & (y_prob <= 1)).all():
        print(f"[SKIP] {name}: probabilities are outside [0, 1].")
        return None

    y_pred = (y_prob >= 0.5).astype(int)

    results = {
        "Model": name,
        "Records": len(y_true),
        "Accuracy": float(accuracy_score(y_true, y_pred)),
        "Precision": float(precision_score(y_true, y_pred, zero_division=0)),
        "Recall": float(recall_score(y_true, y_pred, zero_division=0)),
        "F1": float(f1_score(y_true, y_pred, zero_division=0)),
        "ROC-AUC": float(roc_auc_score(y_true, y_prob)),
        "PR-AUC": float(average_precision_score(y_true, y_prob)),
        "Balanced Accuracy": float(balanced_accuracy_score(y_true, y_pred)),
        "Confusion Matrix": confusion_matrix(y_true, y_pred).tolist(),
    }

    print(f"\n{name} RESULTS")
    print(json.dumps(results, indent=2))

    merged["actual_label"] = y_true
    merged["predicted_probability"] = y_prob
    merged["predicted_label"] = y_pred

    output_name = name.lower().replace(" ", "_").replace("-", "_")

    merged.to_csv(
        COMMON_DIR / f"{output_name}_common_predictions.csv",
        index=False,
    )

    return results


def main():

    common = load_common_cohort()

    results = []

    for name, file_path in FILES.items():

        result = load_predictions(name, file_path, common)

        if result is not None:
            results.append(result)

    if not results:
        print("\nNo models were successfully evaluated.")
        return

    # Save complete results, including confusion matrices.
    with open(COMMON_DIR / "common_723_full_results.json", "w") as f:
        json.dump(results, f, indent=2)

    # Create percentage comparison table.
    comparison = pd.DataFrame(results)

    comparison = comparison.drop(columns=["Confusion Matrix"])

    metric_cols = [
        "Accuracy",
        "Precision",
        "Recall",
        "F1",
        "ROC-AUC",
        "PR-AUC",
        "Balanced Accuracy",
    ]

    comparison[metric_cols] = (
        comparison[metric_cols] * 100
    ).round(2)

    comparison.to_csv(
        COMMON_DIR / "common_723_comparison.csv",
        index=False,
    )

    print("\n" + "=" * 75)
    print("FINAL COMMON-COHORT MODEL COMPARISON (%)")
    print("=" * 75)
    print(comparison.to_string(index=False))

    print("\nResults saved in:")
    print(COMMON_DIR)


if __name__ == "__main__":
    main()
