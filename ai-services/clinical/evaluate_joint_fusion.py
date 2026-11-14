"""
Evaluate the saved Joint Fusion checkpoint.

This script DOES NOT train the model.
It only:
1. Loads joint_fusion_best.pt
2. Uses the same validation dataset
3. Calculates Accuracy, Precision, Recall, F1, ROC-AUC
4. Prints the confusion matrix
5. Saves joint_fusion_results.json
"""

import os
import json
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

# Import the EXACT architecture and utilities
# from the existing Joint Fusion training script.
from train_joint_fusion import (
    JointFusionDataset,
    JointFusionModel,
    run_epoch,
    DEVICE,
    MANIFEST_PATH,
    VOCAB_PATH,
    CHECKPOINT_PATH,
    MODEL_PATH,
    OUTPUT_DIR,
    BATCH_SIZE,
    MAX_SEQUENCE_LENGTH,
    IMAGE_SIZE,
    IMAGE_TOKEN_DIM,
    NUM_HEADS,
)


def main():

    print("=" * 70)
    print("JOINT FUSION — CHECKPOINT EVALUATION")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    # ============================================================
    # CHECKPOINT
    # ============================================================

    if not os.path.exists(MODEL_PATH):

        raise FileNotFoundError(
            f"\nJoint Fusion checkpoint not found:\n{MODEL_PATH}"
        )

    print()
    print("Checkpoint found:")
    print(MODEL_PATH)

    # ============================================================
    # LOAD MANIFEST
    # ============================================================

    print()
    print("Loading fusion manifest...")

    df = pd.read_csv(
        MANIFEST_PATH
    )

    train_df = df[
        df["final_split"] == "train"
    ].copy()

    val_df = df[
        df["final_split"] == "val"
    ].copy()

    print()
    print("Total records      :", len(df))
    print("Training records   :", len(train_df))
    print("Validation records :", len(val_df))

    # ============================================================
    # LOAD VOCABULARY
    # ============================================================

    print()
    print("Loading training vocabulary...")

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        vocab_data = json.load(f)

    vocab = vocab_data["vocabulary"]

    vocab_size = len(vocab)

    pad_idx = vocab["<PAD>"]

    print()
    print("Vocabulary size:", vocab_size)
    print("PAD index      :", pad_idx)

    # ============================================================
    # VALIDATION DATASET
    # ============================================================

    print()
    print("Creating validation dataset...")

    val_dataset = JointFusionDataset(
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

    print(
        "Validation batches:",
        len(val_loader)
    )

    # ============================================================
    # BUILD EXACT SAME MODEL
    # ============================================================

    print()
    print("Building Joint Fusion model...")

    model = JointFusionModel(
        vocab_size=vocab_size,
        pad_idx=pad_idx,
        checkpoint_path=CHECKPOINT_PATH
    ).to(DEVICE)

    # ============================================================
    # LOAD SAVED BEST CHECKPOINT
    # ============================================================

    print()
    print("=" * 70)
    print("LOADING SAVED BEST CHECKPOINT")
    print("=" * 70)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    print()
    print("Checkpoint best epoch:",
          checkpoint.get("best_epoch"))

    print(
        "Checkpoint best ROC-AUC:",
        checkpoint.get("best_val_roc_auc")
    )

    model.load_state_dict(
        checkpoint["model_state_dict"],
        strict=True
    )

    print()
    print("Checkpoint loaded successfully.")
    print("No training will be performed.")

    # ============================================================
    # SAME CLASS WEIGHT AS TRAINING
    # ============================================================

    positive_count = int(
        (train_df["label"] == 1).sum()
    )

    negative_count = int(
        (train_df["label"] == 0).sum()
    )

    pos_weight_value = (
        negative_count /
        positive_count
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=DEVICE
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    print()
    print("Positive samples:", positive_count)
    print("Negative samples:", negative_count)
    print(
        "Positive class weight:",
        pos_weight_value
    )

    # ============================================================
    # FINAL VALIDATION
    # ============================================================

    print()
    print("=" * 70)
    print("RUNNING VALIDATION")
    print("=" * 70)

    final_metrics = run_epoch(
        model,
        val_loader,
        criterion
    )

    # ============================================================
    # PRINT RESULTS
    # ============================================================

    print()
    print("=" * 70)
    print("JOINT FUSION — EPOCH 11 CHECKPOINT RESULTS")
    print("=" * 70)

    print()
    print(
        f"Accuracy : "
        f"{final_metrics['accuracy']:.6f}"
    )

    print(
        f"Precision: "
        f"{final_metrics['precision']:.6f}"
    )

    print(
        f"Recall   : "
        f"{final_metrics['recall']:.6f}"
    )

    print(
        f"F1 Score : "
        f"{final_metrics['f1_score']:.6f}"
    )

    print(
        f"ROC-AUC  : "
        f"{final_metrics['roc_auc']:.6f}"
    )

    print()
    print("Confusion Matrix:")

    print(
        final_metrics["confusion_matrix"]
    )

    # ============================================================
    # SAVE RESULTS
    # ============================================================

    results_path = os.path.join(
        OUTPUT_DIR,
        "joint_fusion_results.json"
    )

    results = {

        "model":
            "Joint Fusion",

        "fusion_type":
            "Joint multimodal cross-attention fusion",

        "architecture": {

            "image_encoder":
                "DenseNet121 spatial feature encoder",

            "text_encoder":
                "2-layer bidirectional LSTM",

            "cross_modal_fusion":
                "bidirectional image-text cross-attention",

            "image_tokens":
                "DenseNet spatial regions; "
                "no global pooling before attention",

            "interaction_features": [

                "pooled image representation",

                "masked pooled text representation",

                "element-wise product",

                "absolute difference"
            ]
        },

        "device":
            str(DEVICE),

        "total_records":
            len(df),

        "train_records":
            len(train_df),

        "validation_records":
            len(val_df),

        "vocabulary_size":
            vocab_size,

        "max_sequence_length":
            MAX_SEQUENCE_LENGTH,

        "image_size":
            IMAGE_SIZE,

        "image_token_dim":
            IMAGE_TOKEN_DIM,

        "attention_heads":
            NUM_HEADS,

        "checkpoint":
            MODEL_PATH,

        "checkpoint_best_epoch":
            checkpoint.get("best_epoch"),

        "checkpoint_best_val_roc_auc":
            checkpoint.get("best_val_roc_auc"),

        "final_metrics":
            final_metrics,

        "note":
            "Evaluation-only run using the saved best "
            "Joint Fusion checkpoint. "
            "No model parameters were updated."
    }

    with open(
        results_path,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    # ============================================================
    # DONE
    # ============================================================

    print()
    print("=" * 70)
    print("RESULTS SAVED")
    print("=" * 70)

    print()
    print(results_path)

    print()
    print("=" * 70)
    print("JOINT FUSION EVALUATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()