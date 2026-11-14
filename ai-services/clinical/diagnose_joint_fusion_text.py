
"""
Joint Fusion — Text Sensitivity Diagnostic

Evaluation only. Does not train or modify model parameters.

Experiments:
1. Original clinical text
2. Impression section removed
3. Clinical text replaced with text from another patient

Preserves:
- Original checkpoint
- Original vocabulary
- Original validation image-label pairs
- Original validation ordering

Saves separate timestamped JSON and CSV results.
"""

import os
import re
import json
from datetime import datetime

import numpy as np
import pandas as pd
import torch

from torch.utils.data import DataLoader

from train_joint_fusion import (
    JointFusionDataset,
    JointFusionModel,
    calculate_metrics,
    DEVICE,
    MANIFEST_PATH,
    VOCAB_PATH,
    CHECKPOINT_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    BATCH_SIZE,
    MAX_SEQUENCE_LENGTH,
)


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

IMPRESSION_PATTERN = re.compile(
    r"\bimpression\s*:",
    flags=re.IGNORECASE
)


# ============================================================
# TEXT PERTURBATION
# ============================================================

def remove_impression(text):

    text = str(text)

    match = IMPRESSION_PATTERN.search(text)

    if match is None:
        return text, False

    # Keep everything before the impression section.
    masked_text = text[:match.start()].strip()

    return masked_text, True


def create_shuffled_text(df):

    """
    Assign each patient another patient's clinical text.

    All records belonging to one patient receive text
    from a different patient.

    A deterministic derangement is used.
    """

    patient_ids = df["patient_id"].astype(str).unique()

    if len(patient_ids) < 2:
        raise ValueError(
            "At least two unique patients are required."
        )

    rng = np.random.default_rng(RANDOM_SEED)

    shuffled_ids = rng.permutation(patient_ids)

    # Ensure nobody receives their own patient's text.
    while np.any(patient_ids == shuffled_ids):
        shuffled_ids = rng.permutation(patient_ids)

    # Use one representative text per donor patient.
    patient_text = (
        df.assign(
            patient_id=df["patient_id"].astype(str)
        )
        .groupby("patient_id", sort=False)["clinical_text"]
        .first()
        .to_dict()
    )

    donor_mapping = dict(
        zip(patient_ids, shuffled_ids)
    )

    shuffled = df.copy()

    shuffled["clinical_text"] = [
        patient_text[donor_mapping[str(pid)]]
        for pid in df["patient_id"]
    ]

    shuffled["text_donor_patient_id"] = [
        donor_mapping[str(pid)]
        for pid in df["patient_id"]
    ]

    return shuffled, donor_mapping


# ============================================================
# EVALUATION
# ============================================================

@torch.inference_mode()
def evaluate_model(model, dataframe, vocab):

    dataset = JointFusionDataset(
        dataframe,
        vocab
    )

    loader = DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
        pin_memory=torch.cuda.is_available()
    )

    model.eval()

    all_labels = []
    all_probabilities = []

    for batch in loader:

        images = batch["image"].to(
            DEVICE,
            non_blocking=True
        )

        input_ids = batch["input_ids"].to(
            DEVICE,
            non_blocking=True
        )

        attention_mask = batch["attention_mask"].to(
            DEVICE,
            non_blocking=True
        )

        labels = batch["label"].to(
            DEVICE,
            non_blocking=True
        )

        logits = model(
            images,
            input_ids,
            attention_mask
        )

        probabilities = torch.sigmoid(logits)

        all_labels.extend(
            labels.cpu().numpy().reshape(-1)
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().reshape(-1)
        )

    labels = np.asarray(
        all_labels,
        dtype=int
    )

    probabilities = np.asarray(
        all_probabilities,
        dtype=float
    )

    metrics = calculate_metrics(
        labels,
        probabilities
    )

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return metrics, labels, probabilities, predictions


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("JOINT FUSION — TEXT SENSITIVITY DIAGNOSTIC")
    print("=" * 70)

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # CHECK CHECKPOINT
    # --------------------------------------------------------

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(MODEL_PATH)

    if not os.path.exists(VOCAB_PATH):
        raise FileNotFoundError(VOCAB_PATH)

    print("\nCheckpoint:", MODEL_PATH)

    # --------------------------------------------------------
    # LOAD VALIDATION DATA
    # --------------------------------------------------------

    print("\nLoading validation manifest...")

    df = pd.read_csv(MANIFEST_PATH)

    val_df = df[
        df["final_split"] == "val"
    ].copy().reset_index(drop=True)

    print("Validation records:", len(val_df))
    print(
        "Unique validation patients:",
        val_df["patient_id"].nunique()
    )

    # Preserve original identity and order.
    original_metadata = val_df[
        ["patient_id", "image_path", "label"]
    ].copy()

    # --------------------------------------------------------
    # LOAD VOCABULARY
    # --------------------------------------------------------

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        vocab_data = json.load(f)

    vocab = vocab_data["vocabulary"]

    vocab_size = len(vocab)
    pad_idx = vocab["<PAD>"]

    print("Vocabulary size:", vocab_size)
    print("Maximum sequence length:", MAX_SEQUENCE_LENGTH)

    # --------------------------------------------------------
    # BUILD AND LOAD MODEL
    # --------------------------------------------------------

    print("\nBuilding original Joint Fusion architecture...")

    model = JointFusionModel(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        checkpoint_path=CHECKPOINT_PATH
    ).to(DEVICE)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    model.eval()

    print("Checkpoint loaded successfully.")
    print("Best epoch:", checkpoint.get("best_epoch"))
    print(
        "Checkpoint validation ROC-AUC:",
        checkpoint.get("best_val_roc_auc")
    )

    # --------------------------------------------------------
    # EXPERIMENT 1: ORIGINAL TEXT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT 1 — ORIGINAL TEXT")
    print("=" * 70)

    original_metrics, labels, original_probs, original_preds = (
        evaluate_model(
            model,
            val_df,
            vocab
        )
    )

    # --------------------------------------------------------
    # EXPERIMENT 2: IMPRESSION REMOVED
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT 2 — IMPRESSION REMOVED")
    print("=" * 70)

    masked_df = val_df.copy()

    masked_results = masked_df["clinical_text"].apply(
        remove_impression
    )

    masked_df["clinical_text"] = [
        result[0] for result in masked_results
    ]

    marker_found = sum(
        result[1] for result in masked_results
    )

    empty_after_mask = (
        masked_df["clinical_text"]
        .fillna("")
        .str.strip()
        .eq("")
        .sum()
    )

    print("Impression sections removed:", marker_found)
    print("Records without impression marker:",
          len(masked_df) - marker_found)
    print("Records with empty text after masking:",
          int(empty_after_mask))

    masked_metrics, _, masked_probs, masked_preds = (
        evaluate_model(
            model,
            masked_df,
            vocab
        )
    )

    # --------------------------------------------------------
    # EXPERIMENT 3: DIFFERENT PATIENT'S TEXT
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("EXPERIMENT 3 — SHUFFLED PATIENT TEXT")
    print("=" * 70)

    shuffled_df, donor_mapping = create_shuffled_text(
        val_df
    )

    shuffled_metrics, _, shuffled_probs, shuffled_preds = (
        evaluate_model(
            model,
            shuffled_df,
            vocab
        )
    )

    print(
        "Unique patient-to-patient mappings:",
        len(donor_mapping)
    )

    # --------------------------------------------------------
    # VERIFY IMAGE-LABEL ALIGNMENT
    # --------------------------------------------------------

    for altered_df in [masked_df, shuffled_df]:

        assert (
            altered_df["image_path"].tolist()
            == original_metadata["image_path"].tolist()
        )

        assert (
            altered_df["label"].tolist()
            == original_metadata["label"].tolist()
        )

        assert (
            altered_df["patient_id"].astype(str).tolist()
            == original_metadata["patient_id"].astype(str).tolist()
        )

    # --------------------------------------------------------
    # PRINT COMPARISON
    # --------------------------------------------------------

    experiments = {
        "original": original_metrics,
        "impression_removed": masked_metrics,
        "shuffled_patient_text": shuffled_metrics
    }

    print("\n" + "=" * 70)
    print("DIAGNOSTIC RESULTS")
    print("=" * 70)

    metric_names = [
        "accuracy",
        "precision",
        "recall",
        "f1_score",
        "roc_auc"
    ]

    print(
        f"\n{'Metric':<15}"
        f"{'Original':>14}"
        f"{'Impression Removed':>22}"
        f"{'Shuffled Text':>18}"
    )

    for metric in metric_names:

        print(
            f"{metric:<15}"
            f"{original_metrics[metric]:>14.6f}"
            f"{masked_metrics[metric]:>22.6f}"
            f"{shuffled_metrics[metric]:>18.6f}"
        )

    print("\nOriginal confusion matrix:")
    print(original_metrics["confusion_matrix"])

    print("\nImpression-removed confusion matrix:")
    print(masked_metrics["confusion_matrix"])

    print("\nShuffled-text confusion matrix:")
    print(shuffled_metrics["confusion_matrix"])

    # --------------------------------------------------------
    # PER-RECORD RESULTS
    # --------------------------------------------------------

    prediction_df = original_metadata.copy()

    prediction_df["original_probability"] = original_probs
    prediction_df["original_prediction"] = original_preds

    prediction_df["masked_probability"] = masked_probs
    prediction_df["masked_prediction"] = masked_preds

    prediction_df["shuffled_probability"] = shuffled_probs
    prediction_df["shuffled_prediction"] = shuffled_preds

    prediction_df["shuffled_donor_patient_id"] = (
        shuffled_df["text_donor_patient_id"].values
    )

    # --------------------------------------------------------
    # SAVE RESULTS WITHOUT OVERWRITING
    # --------------------------------------------------------

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    timestamp = datetime.now().strftime(
        "%Y%m%d_%H%M%S"
    )

    json_path = os.path.join(
        OUTPUT_DIR,
        f"joint_fusion_text_diagnostic_{timestamp}.json"
    )

    csv_path = os.path.join(
        OUTPUT_DIR,
        f"joint_fusion_text_predictions_{timestamp}.csv"
    )

    results = {

        "experiment":
            "Joint Fusion text sensitivity diagnostic",

        "checkpoint":
            MODEL_PATH,

        "checkpoint_best_epoch":
            checkpoint.get("best_epoch"),

        "checkpoint_best_validation_roc_auc":
            checkpoint.get("best_val_roc_auc"),

        "validation_records":
            len(val_df),

        "unique_validation_patients":
            int(val_df["patient_id"].nunique()),

        "vocabulary_size":
            vocab_size,

        "max_sequence_length":
            MAX_SEQUENCE_LENGTH,

        "random_seed":
            RANDOM_SEED,

        "impression_masking": {

            "records_with_impression_marker":
                int(marker_found),

            "records_without_impression_marker":
                int(len(val_df) - marker_found),

            "records_with_empty_text_after_masking":
                int(empty_after_mask)
        },

        "experiments": experiments,

        "interpretation_note":
            "Inference-only sensitivity tests. "
            "Altered text conditions may cause input-distribution "
            "shift and do not establish causal leakage or replace "
            "separately trained ablation experiments."
    }

    with open(
        json_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    prediction_df.to_csv(
        csv_path,
        index=False
    )

    print("\n" + "=" * 70)
    print("DIAGNOSTIC COMPLETE")
    print("=" * 70)

    print("\nJSON results:")
    print(json_path)

    print("\nPer-record predictions:")
    print(csv_path)

    print("\nNo training performed.")
    print("Original checkpoint was not modified.")
    print("Original manifest was not modified.")


if __name__ == "__main__":
    main()
