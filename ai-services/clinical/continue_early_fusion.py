import os
import json
import random
import re

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CONFIG
# ============================================================

SEED = 42

MANIFEST_PATH = (
    "data/clinical_text/fusion_experiment_dataset.csv"
)

VOCAB_PATH = (
    "data/clinical_text/joint_fusion_vocab.json"
)

DENSENET_CHECKPOINT = (
    "ai-services/xray/outputs/"
    "best_densenet121_finetuned_cpu.pth"
)

# Existing Epoch-2 Early Fusion checkpoint
START_CHECKPOINT = (
    "ai-services/clinical/outputs/"
    "early_fusion_full_best.pt"
)

OUTPUT_DIR = (
    "ai-services/clinical/outputs"
)

# Separate output so the original checkpoint is NOT overwritten
CONTINUED_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "early_fusion_full_continued_best.pt"
)

CONTINUED_RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "early_fusion_full_continued_results.json"
)


# ============================================================
# CONTINUATION CONFIG
# ============================================================

IMAGE_SIZE = 160

MAX_SEQUENCE_LENGTH = 256

BATCH_SIZE = 8

# Only 3 additional epochs
CONTINUATION_EPOCHS = 3

# Stop if there is no improvement for 2 epochs
EARLY_STOPPING_PATIENCE = 2

# Same learning rates as original experiment
FUSION_LEARNING_RATE = 1e-4

IMAGE_LEARNING_RATE = 1e-5

WEIGHT_DECAY = 1e-4


# ============================================================
# TEXT ENCODER
# ============================================================

EMBEDDING_DIM = 256

LSTM_HIDDEN_DIM = 128

TEXT_OUTPUT_DIM = 256


# ============================================================
# IMAGE ENCODER
# ============================================================

IMAGE_FEATURE_DIM = 1024

IMAGE_OUTPUT_DIM = 256


DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)

    np.random.seed(seed)

    torch.manual_seed(seed)

    if torch.cuda.is_available():

        torch.cuda.manual_seed_all(seed)


# ============================================================
# IMAGE PREPROCESSING
# ============================================================

IMAGE_TRANSFORM = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASET
# ============================================================

class EarlyFusionDataset(Dataset):

    def __init__(
        self,
        dataframe,
        vocabulary
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.vocab = vocabulary

        self.pad_idx = vocabulary[
            "<PAD>"
        ]

        self.unk_idx = vocabulary[
            "<UNK>"
        ]

    def tokenize(self, text):

        text = str(text).lower()

        tokens = re.findall(

            r"[a-z]+(?:[-'][a-z]+)*|\d+(?:\.\d+)?",

            text
        )

        return tokens

    def encode_text(self, text):

        tokens = self.tokenize(text)

        ids = [

            self.vocab.get(
                token,
                self.unk_idx
            )

            for token in tokens
        ]

        ids = ids[
            :MAX_SEQUENCE_LENGTH
        ]

        mask = [
            1
        ] * len(ids)

        padding = (
            MAX_SEQUENCE_LENGTH -
            len(ids)
        )

        if padding > 0:

            ids += [
                self.pad_idx
            ] * padding

            mask += [
                0
            ] * padding

        return (

            torch.tensor(
                ids,
                dtype=torch.long
            ),

            torch.tensor(
                mask,
                dtype=torch.bool
            )
        )

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = str(
            row["image_path"]
        )

        image = Image.open(
            image_path
        ).convert("L")

        image = IMAGE_TRANSFORM(
            image
        )

        input_ids, attention_mask = (
            self.encode_text(
                row["clinical_text"]
            )
        )

        label = torch.tensor(

            float(row["label"]),

            dtype=torch.float32
        )

        return {

            "image": image,

            "input_ids": input_ids,

            "attention_mask":
                attention_mask,

            "label": label,

            "patient_id":
                str(row["patient_id"])
        }


# ============================================================
# IMAGE ENCODER
# ============================================================

class DenseNetEncoder(nn.Module):

    def __init__(self):

        super().__init__()

        # Build the same DenseNet121 feature architecture.
        # The complete Early Fusion checkpoint will be loaded
        # later with strict=True.

        densenet = models.densenet121(
            weights=None,
            num_classes=2
        )

        self.features = densenet.features

        self.output_dim = IMAGE_FEATURE_DIM

    def forward(self, x):

        x = self.features(x)

        x = torch.relu(x)

        x = torch.nn.functional.adaptive_avg_pool2d(
            x,
            (1, 1)
        )

        x = torch.flatten(
            x,
            1
        )

        return x


# ============================================================
# TEXT ENCODER
# ============================================================

class ClinicalTextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        pad_idx
    ):

        super().__init__()

        self.embedding = nn.Embedding(

            vocab_size,

            EMBEDDING_DIM,

            padding_idx=pad_idx
        )

        self.lstm = nn.LSTM(

            input_size=EMBEDDING_DIM,

            hidden_size=LSTM_HIDDEN_DIM,

            num_layers=2,

            batch_first=True,

            bidirectional=True,

            dropout=0.15
        )

        self.projection = nn.Sequential(

            nn.Linear(

                LSTM_HIDDEN_DIM * 2,

                TEXT_OUTPUT_DIM
            ),

            nn.LayerNorm(
                TEXT_OUTPUT_DIM
            ),

            nn.GELU(),

            nn.Dropout(0.15)
        )

    def forward(
        self,
        input_ids,
        attention_mask
    ):

        x = self.embedding(
            input_ids
        )

        x, _ = self.lstm(x)

        x = self.projection(x)

        mask = attention_mask.unsqueeze(
            -1
        ).float()

        x = x * mask

        denominator = mask.sum(
            dim=1
        ).clamp(
            min=1.0
        )

        x = x.sum(
            dim=1
        ) / denominator

        return x


# ============================================================
# FULL EARLY FUSION MODEL
# ============================================================

class FullEarlyFusionModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        pad_idx
    ):

        super().__init__()

        # ----------------------------------------------------
        # Image encoder
        # ----------------------------------------------------

        self.image_encoder = DenseNetEncoder()

        self.image_projection = nn.Sequential(

            nn.Linear(
                IMAGE_FEATURE_DIM,
                IMAGE_OUTPUT_DIM
            ),

            nn.LayerNorm(
                IMAGE_OUTPUT_DIM
            ),

            nn.GELU(),

            nn.Dropout(0.15)
        )

        # ----------------------------------------------------
        # Clinical text encoder
        # ----------------------------------------------------

        self.text_encoder = ClinicalTextEncoder(
            vocab_size,
            pad_idx
        )

        # ----------------------------------------------------
        # Feature-level early fusion
        # ----------------------------------------------------

        fused_dim = (
            IMAGE_OUTPUT_DIM
            + TEXT_OUTPUT_DIM
        )

        self.fusion_head = nn.Sequential(

            nn.Linear(
                fused_dim,
                512
            ),

            nn.LayerNorm(
                512
            ),

            nn.GELU(),

            nn.Dropout(0.30),

            nn.Linear(
                512,
                128
            ),

            nn.GELU(),

            nn.Dropout(0.20),

            nn.Linear(
                128,
                1
            )
        )

    def forward(
        self,
        image,
        input_ids,
        attention_mask
    ):

        # Image representation
        image_features = self.image_encoder(
            image
        )

        image_features = self.image_projection(
            image_features
        )

        # Clinical text representation
        text_features = self.text_encoder(
            input_ids,
            attention_mask
        )

        # Early feature-level fusion
        fused_features = torch.cat(
            [
                image_features,
                text_features
            ],
            dim=1
        )

        logits = self.fusion_head(
            fused_features
        )

        return logits.squeeze(1)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    return {

        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
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
            roc_auc_score(
                labels,
                probabilities
            )
        ),

        "confusion_matrix":
            confusion_matrix(
                labels,
                predictions
            ).tolist()
    }


# ============================================================
# TRAIN / VALIDATION
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None
):

    training = optimizer is not None

    if training:
        model.train()
    else:
        model.eval()

    total_loss = 0.0

    all_labels = []

    all_probabilities = []

    for batch in loader:

        images = batch[
            "image"
        ].to(DEVICE)

        input_ids = batch[
            "input_ids"
        ].to(DEVICE)

        attention_mask = batch[
            "attention_mask"
        ].to(DEVICE)

        labels = batch[
            "label"
        ].to(DEVICE)

        if training:

            optimizer.zero_grad(
                set_to_none=True
            )

        with torch.set_grad_enabled(
            training
        ):

            logits = model(

                images,

                input_ids,

                attention_mask
            )

            loss = criterion(
                logits,
                labels
            )

            if training:

                loss.backward()

                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    max_norm=1.0
                )

                optimizer.step()

        total_loss += (

            loss.item() *

            labels.size(0)
        )

        probabilities = torch.sigmoid(
            logits
        )

        all_labels.extend(

            labels.detach()
            .cpu()
            .numpy()
        )

        all_probabilities.extend(

            probabilities.detach()
            .cpu()
            .numpy()
        )

    all_labels = np.array(
        all_labels
    )

    all_probabilities = np.array(
        all_probabilities
    )

    average_loss = (

        total_loss /

        len(loader.dataset)
    )

    metrics = calculate_metrics(

        all_labels,

        all_probabilities
    )

    metrics["loss"] = float(
        average_loss
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print(
        "FULL EARLY FUSION — WARM START CONTINUATION"
    )
    print("=" * 70)

    print()
    print(
        "Device:",
        DEVICE
    )

    # --------------------------------------------------------
    # Check starting checkpoint
    # --------------------------------------------------------

    if not os.path.exists(
        START_CHECKPOINT
    ):

        raise FileNotFoundError(
            f"Starting checkpoint not found:\n"
            f"{START_CHECKPOINT}"
        )

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

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
    print(
        "Total records:",
        len(df)
    )

    print(
        "Training records:",
        len(train_df)
    )

    print(
        "Validation records:",
        len(val_df)
    )

    # --------------------------------------------------------
    # Vocabulary
    # --------------------------------------------------------

    with open(
        VOCAB_PATH,
        "r",
        encoding="utf-8"
    ) as f:

        vocab_data = json.load(f)

    vocab = vocab_data[
        "vocabulary"
    ]

    vocab_size = len(vocab)

    pad_idx = vocab[
        "<PAD>"
    ]

    print()
    print(
        "Vocabulary size:",
        vocab_size
    )

    # --------------------------------------------------------
    # Dataset / loaders
    # --------------------------------------------------------

    train_dataset = EarlyFusionDataset(
        train_df,
        vocab
    )

    val_dataset = EarlyFusionDataset(
        val_df,
        vocab
    )

    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=torch.cuda.is_available()
    )

    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available()
    )

    # --------------------------------------------------------
    # Build exact same architecture
    # --------------------------------------------------------

    print()
    print(
        "Building Full Early Fusion model..."
    )

    model = FullEarlyFusionModel(
        vocab_size=vocab_size,
        pad_idx=pad_idx
    ).to(DEVICE)

    # --------------------------------------------------------
    # LOAD EXISTING EARLY FUSION CHECKPOINT
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LOADING EXISTING EPOCH-2 EARLY FUSION CHECKPOINT"
    )
    print("=" * 70)

    start_checkpoint = torch.load(

        START_CHECKPOINT,

        map_location=DEVICE
    )

    print()
    print(
        "Saved best epoch:",
        start_checkpoint.get(
            "best_epoch"
        )
    )

    print(
        "Saved best ROC-AUC:",
        start_checkpoint.get(
            "best_val_roc_auc"
        )
    )

    checkpoint_state = (
        start_checkpoint[
            "model_state_dict"
        ]
    )

    # Strict loading is intentional.
    # If the architecture does not match,
    # STOP instead of silently continuing.

    load_result = model.load_state_dict(

        checkpoint_state,

        strict=True
    )

    print()
    print(
        "✓ Existing Early Fusion weights loaded successfully."
    )

    print(
        load_result
    )

    # --------------------------------------------------------
    # Parameter count
    # --------------------------------------------------------

    total_params = sum(

        p.numel()

        for p in model.parameters()
    )

    trainable_params = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )

    print()
    print(
        "Total parameters:",
        f"{total_params:,}"
    )

    print(
        "Trainable parameters:",
        f"{trainable_params:,}"
    )

    # --------------------------------------------------------
    # Class imbalance
    # --------------------------------------------------------

    positive_count = (
        train_df["label"] == 1
    ).sum()

    negative_count = (
        train_df["label"] == 0
    ).sum()

    pos_weight_value = (

        negative_count /

        positive_count
    )

    pos_weight = torch.tensor(

        [pos_weight_value],

        dtype=torch.float32,

        device=DEVICE
    )

    print()
    print(
        "Positive samples:",
        int(positive_count)
    )

    print(
        "Negative samples:",
        int(negative_count)
    )

    print(
        "Positive class weight:",
        float(pos_weight_value)
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    # --------------------------------------------------------
    # Fresh optimizer
    # --------------------------------------------------------

    image_parameters = list(

        model.image_encoder.parameters()
    )

    other_parameters = [

        parameter

        for name, parameter
        in model.named_parameters()

        if not name.startswith(
            "image_encoder."
        )
    ]

    optimizer = torch.optim.AdamW(

        [

            {
                "params":
                    image_parameters,

                "lr":
                    IMAGE_LEARNING_RATE
            },

            {
                "params":
                    other_parameters,

                "lr":
                    FUSION_LEARNING_RATE
            }

        ],

        weight_decay=WEIGHT_DECAY
    )

    scheduler = (
        torch.optim.lr_scheduler.ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=1
        )
    )

    # --------------------------------------------------------
    # Existing best result
    # --------------------------------------------------------

    best_auc = float(

        start_checkpoint[
            "best_val_roc_auc"
        ]
    )

    previous_best_epoch = int(

        start_checkpoint[
            "best_epoch"
        ]
    )

    best_epoch = previous_best_epoch

    patience_counter = 0

    history = []

    print()
    print("=" * 70)
    print(
        "STARTING WARM-START CONTINUATION"
    )
    print("=" * 70)

    print()
    print(
        "Previous best epoch:",
        previous_best_epoch
    )

    print(
        "Previous best ROC-AUC:",
        f"{best_auc:.6f}"
    )

    print(
        "Additional epochs:",
        CONTINUATION_EPOCHS
    )

    # --------------------------------------------------------
    # Continue training
    # --------------------------------------------------------

    for continuation_epoch in range(

        1,

        CONTINUATION_EPOCHS + 1
    ):

        train_metrics = run_epoch(

            model,

            train_loader,

            criterion,

            optimizer
        )

        val_metrics = run_epoch(

            model,

            val_loader,

            criterion
        )

        scheduler.step(
            val_metrics["roc_auc"]
        )

        # Actual epoch number in the overall experiment
        actual_epoch = (

            previous_best_epoch +

            continuation_epoch
        )

        print()
        print(
            f"Continuation Epoch "
            f"{continuation_epoch:02d}/"
            f"{CONTINUATION_EPOCHS}"
        )

        print(
            f"  Overall Epoch : "
            f"{actual_epoch}"
        )

        print(
            f"  Train Loss    : "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"  Val Loss      : "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"  Val Accuracy  : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"  Val Precision : "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"  Val Recall    : "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"  Val F1        : "
            f"{val_metrics['f1_score']:.4f}"
        )

        print(
            f"  Val ROC-AUC   : "
            f"{val_metrics['roc_auc']:.4f}"
        )

        history.append({

            "continuation_epoch":
                continuation_epoch,

            "overall_epoch":
                actual_epoch,

            "train":
                train_metrics,

            "validation":
                val_metrics
        })

        current_auc = (
            val_metrics["roc_auc"]
        )

        # ----------------------------------------------------
        # New best
        # ----------------------------------------------------

        if current_auc > best_auc:

            best_auc = current_auc

            best_epoch = actual_epoch

            patience_counter = 0

            torch.save(

                {

                    "model_state_dict":
                        model.state_dict(),

                    "vocab_size":
                        vocab_size,

                    "pad_idx":
                        pad_idx,

                    "best_val_roc_auc":
                        best_auc,

                    "best_epoch":
                        best_epoch,

                    "warm_started_from_epoch":
                        previous_best_epoch,

                    "warm_start_checkpoint":
                        START_CHECKPOINT
                },

                CONTINUED_MODEL_PATH
            )

            print()
            print(
                "✓ New best continuation checkpoint saved."
            )

        else:

            patience_counter += 1

            print()
            print(
                "No ROC-AUC improvement."
            )

        if patience_counter >= (
            EARLY_STOPPING_PATIENCE
        ):

            print()
            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Determine final checkpoint
    # --------------------------------------------------------

    if os.path.exists(
        CONTINUED_MODEL_PATH
    ):

        final_checkpoint_path = (
            CONTINUED_MODEL_PATH
        )

        final_checkpoint = torch.load(

            final_checkpoint_path,

            map_location=DEVICE
        )

        model.load_state_dict(

            final_checkpoint[
                "model_state_dict"
            ],

            strict=True
        )

        final_best_auc = float(

            final_checkpoint[
                "best_val_roc_auc"
            ]
        )

        final_best_epoch = int(

            final_checkpoint[
                "best_epoch"
            ]
        )

    else:

        # No improvement over Epoch 2.
        # Use the original checkpoint.

        final_checkpoint_path = (
            START_CHECKPOINT
        )

        final_best_auc = best_auc

        final_best_epoch = best_epoch

        model.load_state_dict(

            start_checkpoint[
                "model_state_dict"
            ],

            strict=True
        )

    # --------------------------------------------------------
    # Final validation
    # --------------------------------------------------------

    final_metrics = run_epoch(

        model,

        val_loader,

        criterion
    )

    # --------------------------------------------------------
    # Print final result
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FULL EARLY FUSION — FINAL RESULTS"
    )
    print("=" * 70)

    print()
    print(
        "Best overall epoch:",
        final_best_epoch
    )

    print(
        "Best ROC-AUC:",
        f"{final_best_auc:.6f}"
    )

    print()

    print(
        f"Accuracy : "
        f"{final_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: "
        f"{final_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : "
        f"{final_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : "
        f"{final_metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC  : "
        f"{final_metrics['roc_auc']:.4f}"
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        np.array(
            final_metrics[
                "confusion_matrix"
            ]
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {

        "model":
            "Full Early Fusion",

        "fusion_type":
            "Trainable feature-level fusion",

        "image_encoder":
            "DenseNet121",

        "text_encoder":
            "2-layer bidirectional LSTM",

        "image_encoder_trainable":
            True,

        "text_encoder_trainable":
            True,

        "total_records":
            len(df),

        "train_records":
            len(train_df),

        "validation_records":
            len(val_df),

        "vocabulary_size":
            vocab_size,

        "warm_started":
            True,

        "warm_started_from_epoch":
            previous_best_epoch,

        "initial_best_val_roc_auc":
            float(
                start_checkpoint[
                    "best_val_roc_auc"
                ]
            ),

        "best_epoch":
            final_best_epoch,

        "best_val_roc_auc":
            final_best_auc,

        "final_metrics":
            final_metrics,

        "training_history":
            history,

        "note":
            "Warm-start continuation from the saved "
            "Early Fusion model. Optimizer and scheduler "
            "states were not restored because they were "
            "not present in the original checkpoint."
    }

    with open(

        CONTINUED_RESULTS_PATH,

        "w",

        encoding="utf-8"

    ) as f:

        json.dump(

            results,

            f,

            indent=4
        )

    print()
    print(
        "Final checkpoint:"
    )

    print(
        final_checkpoint_path
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        CONTINUED_RESULTS_PATH
    )

    print()
    print("=" * 70)
    print(
        "WARM-START CONTINUATION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":

    main()