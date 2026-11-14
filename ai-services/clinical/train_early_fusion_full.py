import os
import json
import random
import re

import numpy as np
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

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

OUTPUT_DIR = (
    "ai-services/clinical/outputs"
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "early_fusion_full_best.pt"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "early_fusion_full_results.json"
)


# ============================================================
# TRAINING CONFIGURATION
# ============================================================

IMAGE_SIZE = 160

MAX_SEQUENCE_LENGTH = 256

BATCH_SIZE = 8

# Deliberately limited for CPU training.
# Early stopping can terminate before this.
EPOCHS = 8

EARLY_STOPPING_PATIENCE = 2

# Newly initialized layers
FUSION_LEARNING_RATE = 1e-4

# Fine-tuned DenseNet
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

    # --------------------------------------------------------
    # Tokenization
    # --------------------------------------------------------

    def tokenize(self, text):

        text = str(text).lower()

        tokens = re.findall(
            r"[a-z]+(?:[-'][a-z]+)*|\d+(?:\.\d+)?",
            text
        )

        return tokens

    # --------------------------------------------------------
    # Text encoding
    # --------------------------------------------------------

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

    # --------------------------------------------------------
    # Dataset
    # --------------------------------------------------------

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        # -------------------------------
        # Image
        # -------------------------------

        image_path = str(
            row["image_path"]
        )

        image = Image.open(
            image_path
        ).convert("L")

        image = IMAGE_TRANSFORM(
            image
        )

        # -------------------------------
        # Text
        # -------------------------------

        input_ids, attention_mask = (
            self.encode_text(
                row["clinical_text"]
            )
        )

        # -------------------------------
        # Label
        # -------------------------------

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

    def __init__(
        self,
        checkpoint_path
    ):

        super().__init__()

        print()
        print(
            "Building DenseNet121..."
        )

        # IMPORTANT:
        # The checkpoint was trained for 2 classes.
        densenet = models.densenet121(
            weights=None,
            num_classes=2
        )

        # ----------------------------------------------------
        # Load the COMPLETE checkpoint
        # ----------------------------------------------------

        print(
            "Loading DenseNet121 checkpoint..."
        )

        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu"
        )

        if (
            not isinstance(
                checkpoint,
                dict
            )
            or
            "model_state_dict"
            not in checkpoint
        ):

            raise RuntimeError(
                "Expected checkpoint to contain "
                "'model_state_dict'."
            )

        state_dict = checkpoint[
            "model_state_dict"
        ]

        # Remove DataParallel prefix if present
        cleaned_state_dict = {}

        for key, value in state_dict.items():

            key = key.replace(
                "module.",
                ""
            )

            cleaned_state_dict[
                key
            ] = value

        # STRICT loading is intentional.
        # We want to guarantee that the exact
        # fine-tuned DenseNet121 checkpoint is used.
        result = densenet.load_state_dict(
            cleaned_state_dict,
            strict=True
        )

        print(
            "DenseNet checkpoint loaded successfully."
        )

        print(result)

        # ----------------------------------------------------
        # Keep only the feature extractor.
        #
        # The 2-class classifier has already served its
        # purpose in X-ray-only training. Early Fusion needs
        # the 1024-D representation before classification.
        # ----------------------------------------------------

        self.features = densenet.features

        self.output_dim = IMAGE_FEATURE_DIM

    def forward(self, x):

        x = self.features(x)

        x = torch.relu(x)

        # Global average pooling
        # for feature-level fusion.

        x = F.adaptive_avg_pool2d(
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

        x, _ = self.lstm(
            x
        )

        x = self.projection(
            x
        )

        # Mask padding tokens

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
        pad_idx,
        checkpoint_path
    ):

        super().__init__()

        # ----------------------------------------------------
        # Image branch
        # ----------------------------------------------------

        self.image_encoder = (
            DenseNetEncoder(
                checkpoint_path
            )
        )

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
        # Text branch
        # ----------------------------------------------------

        self.text_encoder = (
            ClinicalTextEncoder(
                vocab_size,
                pad_idx
            )
        )

        # ----------------------------------------------------
        # Feature-level Early Fusion
        # ----------------------------------------------------

        fused_dim = (
            IMAGE_OUTPUT_DIM +
            TEXT_OUTPUT_DIM
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

        # ----------------------------------------------------
        # IMAGE REPRESENTATION
        # ----------------------------------------------------

        image_features = (
            self.image_encoder(
                image
            )
        )

        image_features = (
            self.image_projection(
                image_features
            )
        )

        # ----------------------------------------------------
        # CLINICAL TEXT REPRESENTATION
        # ----------------------------------------------------

        text_features = (
            self.text_encoder(
                input_ids,
                attention_mask
            )
        )

        # ----------------------------------------------------
        # EARLY / FEATURE-LEVEL FUSION
        # ----------------------------------------------------

        fused_features = torch.cat(
            [
                image_features,
                text_features
            ],
            dim=1
        )

        # ----------------------------------------------------
        # CLASSIFICATION
        # ----------------------------------------------------

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
# TRAIN / VALIDATION EPOCH
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
        "FULL EARLY FUSION — FINAL TRAINING"
    )
    print("=" * 70)

    print()
    print(
        "Device:",
        DEVICE
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
    # Datasets
    # --------------------------------------------------------

    train_dataset = (
        EarlyFusionDataset(
            train_df,
            vocab
        )
    )

    val_dataset = (
        EarlyFusionDataset(
            val_df,
            vocab
        )
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
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Building Full Early Fusion model..."
    )

    model = FullEarlyFusionModel(

        vocab_size=vocab_size,

        pad_idx=pad_idx,

        checkpoint_path=DENSENET_CHECKPOINT

    ).to(DEVICE)

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

    criterion = (
        nn.BCEWithLogitsLoss(
            pos_weight=pos_weight
        )
    )

    # --------------------------------------------------------
    # Optimizer
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
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "STARTING FULL EARLY FUSION TRAINING"
    )
    print("=" * 70)

    print()
    print(
        "Image encoder learning rate:",
        IMAGE_LEARNING_RATE
    )

    print(
        "Text/fusion learning rate:",
        FUSION_LEARNING_RATE
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Maximum epochs:",
        EPOCHS
    )

    print(
        "Early stopping patience:",
        EARLY_STOPPING_PATIENCE
    )

    best_auc = -1.0

    best_epoch = 0

    patience_counter = 0

    history = []

    for epoch in range(
        1,
        EPOCHS + 1
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

        print()
        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )

        print(
            f"  Train Loss   : "
            f"{train_metrics['loss']:.4f}"
        )

        print(
            f"  Val Loss     : "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"  Val Accuracy : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"  Val Precision: "
            f"{val_metrics['precision']:.4f}"
        )

        print(
            f"  Val Recall   : "
            f"{val_metrics['recall']:.4f}"
        )

        print(
            f"  Val F1       : "
            f"{val_metrics['f1_score']:.4f}"
        )

        print(
            f"  Val ROC-AUC  : "
            f"{val_metrics['roc_auc']:.4f}"
        )

        history.append({

            "epoch": epoch,

            "train": train_metrics,

            "validation": val_metrics

        })

        current_auc = (
            val_metrics["roc_auc"]
        )

        if current_auc > best_auc:

            best_auc = current_auc

            best_epoch = epoch

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
                        best_epoch

                },

                MODEL_PATH
            )

            print(
                "  ✓ New best Early Fusion model saved."
            )

        else:

            patience_counter += 1

            print(
                f"  No improvement "
                f"({patience_counter}/"
                f"{EARLY_STOPPING_PATIENCE})"
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
    # Best model
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "LOADING BEST EARLY FUSION MODEL"
    )
    print("=" * 70)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint[
            "model_state_dict"
        ],
        strict=True
    )

    final_metrics = run_epoch(

        model,

        val_loader,

        criterion
    )

    # --------------------------------------------------------
    # Final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print(
        "FULL EARLY FUSION — FINAL RESULTS"
    )
    print("=" * 70)

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

        "image_representation_dim":
            IMAGE_FEATURE_DIM,

        "image_projection_dim":
            IMAGE_OUTPUT_DIM,

        "text_representation_dim":
            TEXT_OUTPUT_DIM,

        "fusion_dimension":
            IMAGE_OUTPUT_DIM +
            TEXT_OUTPUT_DIM,

        "total_records":
            len(df),

        "train_records":
            len(train_df),

        "validation_records":
            len(val_df),

        "vocabulary_size":
            vocab_size,

        "image_size":
            IMAGE_SIZE,

        "max_sequence_length":
            MAX_SEQUENCE_LENGTH,

        "batch_size":
            BATCH_SIZE,

        "maximum_epochs":
            EPOCHS,

        "early_stopping_patience":
            EARLY_STOPPING_PATIENCE,

        "best_epoch":
            best_epoch,

        "best_val_roc_auc":
            float(best_auc),

        "final_metrics":
            final_metrics,

        "training_history":
            history
    }

    with open(
        RESULTS_PATH,
        "w"
    ) as f:

        json.dump(
            results,
            f,
            indent=4
        )

    print()
    print(
        "Model saved to:"
    )

    print(
        MODEL_PATH
    )

    print()

    print(
        "Results saved to:"
    )

    print(
        RESULTS_PATH
    )

    print()
    print("=" * 70)
    print(
        "FULL EARLY FUSION COMPLETE"
    )
    print("=" * 70)


if __name__ == "__main__":
    main()