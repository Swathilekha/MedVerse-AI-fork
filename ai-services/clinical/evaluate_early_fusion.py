
import os
import json
import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    balanced_accuracy_score,
    confusion_matrix
)

# Import the EXACT architecture and dataset used during training
from train_early_fusion_full import (
    EarlyFusionDataset,
    FullEarlyFusionModel,
    DEVICE,
    BATCH_SIZE,
    MANIFEST_PATH,
    VOCAB_PATH,
    DENSENET_CHECKPOINT
)


# ============================================================
# CONFIGURATION
# ============================================================

OUTPUT_DIR = "ai-services/clinical/outputs"

CHECKPOINTS = {
    "EarlyFusion_Best": os.path.join(
        OUTPUT_DIR,
        "early_fusion_full_best.pt"
    ),

    "EarlyFusion_Continued": os.path.join(
        OUTPUT_DIR,
        "early_fusion_full_continued_best.pt"
    )
}

PREDICTIONS_DIR = os.path.join(
    OUTPUT_DIR,
    "evaluation_predictions"
)

os.makedirs(PREDICTIONS_DIR, exist_ok=True)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(labels, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {
        "accuracy": float(
            accuracy_score(labels, predictions)
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(labels, probabilities)
        ),

        "pr_auc": float(
            average_precision_score(
                labels,
                probabilities
            )
        ),

        "balanced_accuracy": float(
            balanced_accuracy_score(
                labels,
                predictions
            )
        ),

        "confusion_matrix": confusion_matrix(
            labels,
            predictions
        ).tolist()
    }

    return metrics, predictions


# ============================================================
# EVALUATE ONE CHECKPOINT
# ============================================================

def evaluate_checkpoint(
    checkpoint_name,
    checkpoint_path,
    val_loader,
    vocab_size,
    pad_idx
):

    print("\n" + "=" * 65)
    print("EVALUATING:", checkpoint_name)
    print("=" * 65)

    if not os.path.exists(checkpoint_path):
        print("Checkpoint not found:", checkpoint_path)
        return None

    # Build the same model architecture
    model = FullEarlyFusionModel(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        checkpoint_path=DENSENET_CHECKPOINT
    ).to(DEVICE)

    # Load trained Early Fusion checkpoint
    checkpoint = torch.load(
        checkpoint_path,
        map_location=DEVICE
    )

    if "model_state_dict" not in checkpoint:
        raise RuntimeError(
            f"Invalid checkpoint format: {checkpoint_path}"
        )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    all_labels = []
    all_probabilities = []
    all_patient_ids = []

    # --------------------------------------------------------
    # INFERENCE
    # --------------------------------------------------------

    with torch.no_grad():

        for batch_index, batch in enumerate(val_loader):

            images = batch["image"].to(DEVICE)

            input_ids = batch["input_ids"].to(DEVICE)

            attention_mask = batch[
                "attention_mask"
            ].to(DEVICE)

            labels = batch["label"].to(DEVICE)

            logits = model(
                images,
                input_ids,
                attention_mask
            )

            probabilities = torch.sigmoid(logits)

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

            all_patient_ids.extend(
                batch["patient_id"]
            )

            if (batch_index + 1) % 20 == 0:
                print(
                    f"Processed batches: {batch_index + 1}"
                )

    # --------------------------------------------------------
    # CALCULATE RESULTS
    # --------------------------------------------------------

    labels = np.asarray(all_labels).astype(int)

    probabilities = np.asarray(
        all_probabilities
    )

    metrics, predictions = calculate_metrics(
        labels,
        probabilities
    )

    print("\nRESULTS")

    for key, value in metrics.items():
        print(f"{key}: {value}")

    # --------------------------------------------------------
    # SAVE PREDICTIONS
    # --------------------------------------------------------

    predictions_df = pd.DataFrame({
        "patient_id": all_patient_ids,
        "actual_label": labels,
        "predicted_probability": probabilities,
        "predicted_label": predictions
    })

    prediction_path = os.path.join(
        PREDICTIONS_DIR,
        f"{checkpoint_name}_predictions.csv"
    )

    predictions_df.to_csv(
        prediction_path,
        index=False
    )

    print("\nPredictions saved:", prediction_path)

    # Add checkpoint metadata if available
    result = {
        "checkpoint": checkpoint_path,
        "validation_records": int(len(labels)),
        "metrics": metrics
    }

    if "epoch" in checkpoint:
        result["checkpoint_epoch"] = int(
            checkpoint["epoch"]
        )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("EARLY FUSION CHECKPOINT EVALUATION")
    print("=" * 65)

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    if not os.path.exists(MANIFEST_PATH):
        raise FileNotFoundError(
            f"Manifest not found: {MANIFEST_PATH}"
        )

    df = pd.read_csv(MANIFEST_PATH)

    required_columns = [
        "image_path",
        "clinical_text",
        "label",
        "patient_id",
        "final_split"
    ]

    missing = [
        column for column in required_columns
        if column not in df.columns
    ]

    if missing:
        raise ValueError(
            f"Missing manifest columns: {missing}"
        )

    val_df = df[
        df["final_split"] == "val"
    ].copy()

    if val_df.empty:
        raise ValueError(
            "No validation records found."
        )

    print("\nTotal records:", len(df))
    print("Validation records:", len(val_df))

    print(
        "Validation class distribution:\n",
        val_df["label"].value_counts().sort_index()
    )

    # --------------------------------------------------------
    # VERIFY IMAGE PATHS
    # --------------------------------------------------------

    missing_images = [
        path for path in val_df["image_path"].astype(str)
        if not os.path.exists(path)
    ]

    if missing_images:
        raise FileNotFoundError(
            f"{len(missing_images)} validation images are missing. "
            f"First missing path: {missing_images[0]}"
        )

    # --------------------------------------------------------
    # LOAD VOCABULARY
    # --------------------------------------------------------

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as file:

        vocab_data = json.load(file)

    vocab = vocab_data["vocabulary"]

    vocab_size = len(vocab)
    pad_idx = vocab["<PAD>"]

    print("Vocabulary size:", vocab_size)

    # --------------------------------------------------------
    # VALIDATION DATASET
    # --------------------------------------------------------

    val_dataset = EarlyFusionDataset(
        val_df,
        vocab
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # EVALUATE CHECKPOINTS
    # --------------------------------------------------------

    all_results = {}

    for name, checkpoint_path in CHECKPOINTS.items():

        result = evaluate_checkpoint(
            checkpoint_name=name,
            checkpoint_path=checkpoint_path,
            val_loader=val_loader,
            vocab_size=vocab_size,
            pad_idx=pad_idx
        )

        if result is not None:
            all_results[name] = result

    # --------------------------------------------------------
    # SAVE FINAL RESULTS
    # --------------------------------------------------------

    if not all_results:
        raise FileNotFoundError(
            "No Early Fusion checkpoints were found."
        )

    results_path = os.path.join(
        OUTPUT_DIR,
        "early_fusion_evaluation_results.json"
    )

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            all_results,
            file,
            indent=4
        )

    # --------------------------------------------------------
    # SUMMARY
    # --------------------------------------------------------

    print("\n" + "=" * 65)
    print("EARLY FUSION EVALUATION SUMMARY")
    print("=" * 65)

    for name, result in all_results.items():

        print("\n", name)

        for metric, value in result["metrics"].items():
            print(f"{metric}: {value}")

    print("\nResults saved to:")
    print(results_path)

    print("\nEvaluation completed successfully.")


if __name__ == "__main__":
    main()
