import os
import json
import random
import copy

import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from PIL import Image

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
# CONFIGURATION
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MANIFEST_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clinical_text",
    "fusion_experiment_dataset.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ai-services",
    "xray",
    "architecture_comparison"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# EXPERIMENT SETTINGS
# ============================================================

SEED = 42

IMAGE_SIZE = 160
BATCH_SIZE = 8

MAX_EPOCHS = 10
PATIENCE = 3

IMAGE_LR = 1e-5
WEIGHT_DECAY = 1e-4

NUM_WORKERS = 0

DEVICE = torch.device("cpu")


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
# DATASET
# ============================================================

class XRayDataset(Dataset):

    def __init__(
        self,
        dataframe,
        transform=None
    ):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):

        return len(self.df)

    def __getitem__(self, index):

        row = self.df.iloc[index]

        image_path = row["image_path"]

        if not os.path.isabs(image_path):
            image_path = os.path.join(
                BASE_DIR,
                image_path
            )

        image_path = os.path.normpath(image_path)

        if not os.path.exists(image_path):

            raise FileNotFoundError(
                f"Image not found:\n{image_path}"
            )

        image = Image.open(
            image_path
        ).convert("L")

        label = int(row["label"])

        if self.transform is not None:

            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.RandomRotation(
        degrees=5
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL BUILDERS
# ============================================================

def build_densenet():

    model = models.densenet121(
        weights=models.DenseNet121_Weights.DEFAULT
    )

    in_features = model.classifier.in_features

    model.classifier = nn.Linear(
        in_features,
        2
    )

    return model


def build_resnet():

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        2
    )

    return model


def build_efficientnet():

    model = models.efficientnet_b0(
        weights=models.EfficientNet_B0_Weights.DEFAULT
    )

    in_features = model.classifier[-1].in_features

    model.classifier[-1] = nn.Linear(
        in_features,
        2
    )

    return model


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities,
    threshold=0.5
):

    predictions = (
        probabilities >= threshold
    ).astype(int)

    accuracy = accuracy_score(
        labels,
        predictions
    )

    precision = precision_score(
        labels,
        predictions,
        zero_division=0
    )

    recall = recall_score(
        labels,
        predictions,
        zero_division=0
    )

    f1 = f1_score(
        labels,
        predictions,
        zero_division=0
    )

    try:

        auc = roc_auc_score(
            labels,
            probabilities
        )

    except ValueError:

        auc = float("nan")

    cm = confusion_matrix(
        labels,
        predictions,
        labels=[0, 1]
    )

    return {
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": float(auc),
        "confusion_matrix": cm.tolist()
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            probabilities = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    average_loss = (
        total_loss
        / len(loader.dataset)
    )

    metrics = calculate_metrics(
        np.asarray(all_labels),
        np.asarray(all_probabilities)
    )

    metrics["loss"] = float(
        average_loss
    )

    return metrics


# ============================================================
# TRAINING
# ============================================================

def train_model(
    model_name,
    model,
    train_loader,
    val_loader,
    class_weights
):

    print()
    print("=" * 70)
    print(f"TRAINING: {model_name}")
    print("=" * 70)

    model = model.to(
        DEVICE
    )

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=IMAGE_LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )

    best_auc = -1.0
    best_epoch = 0

    best_state = None

    patience_counter = 0

    history = []

    for epoch in range(
        1,
        MAX_EPOCHS + 1
    ):

        model.train()

        total_loss = 0.0

        all_labels = []
        all_probabilities = []

        for images, labels in train_loader:

            images = images.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(
                images
            )

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                max_norm=1.0
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            probabilities = torch.softmax(
                logits.detach(),
                dim=1
            )[:, 1]

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

        train_loss = (
            total_loss
            / len(train_loader.dataset)
        )

        train_metrics = calculate_metrics(
            np.asarray(all_labels),
            np.asarray(all_probabilities)
        )

        val_metrics = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_metrics["roc_auc"]
        )

        history_entry = {

            "epoch": epoch,

            "train_loss": train_loss,

            "train_accuracy":
                train_metrics["accuracy"],

            "train_precision":
                train_metrics["precision"],

            "train_recall":
                train_metrics["recall"],

            "train_f1":
                train_metrics["f1"],

            "train_roc_auc":
                train_metrics["roc_auc"],

            "val_loss":
                val_metrics["loss"],

            "val_accuracy":
                val_metrics["accuracy"],

            "val_precision":
                val_metrics["precision"],

            "val_recall":
                val_metrics["recall"],

            "val_f1":
                val_metrics["f1"],

            "val_roc_auc":
                val_metrics["roc_auc"]
        }

        history.append(
            history_entry
        )

        print(
            f"Epoch {epoch:02d}/{MAX_EPOCHS}"
        )

        print(
            f"  Train Loss : {train_loss:.4f}"
        )

        print(
            f"  Train Acc  : "
            f"{train_metrics['accuracy']:.4f}"
        )

        print(
            f"  Val Loss   : "
            f"{val_metrics['loss']:.4f}"
        )

        print(
            f"  Val Acc    : "
            f"{val_metrics['accuracy']:.4f}"
        )

        print(
            f"  Val F1     : "
            f"{val_metrics['f1']:.4f}"
        )

        print(
            f"  Val AUC    : "
            f"{val_metrics['roc_auc']:.4f}"
        )

        # ----------------------------------------------------
        # SAVE BEST MODEL
        # ----------------------------------------------------

        if (
            val_metrics["roc_auc"]
            > best_auc
        ):

            best_auc = (
                val_metrics["roc_auc"]
            )

            best_epoch = epoch

            best_state = copy.deepcopy(
                model.state_dict()
            )

            patience_counter = 0

            checkpoint_path = os.path.join(
                OUTPUT_DIR,
                f"{model_name}_best.pt"
            )

            torch.save(
                {
                    "model_state_dict":
                        best_state,

                    "model":
                        model_name,

                    "best_epoch":
                        best_epoch,

                    "best_val_roc_auc":
                        best_auc,

                    "image_size":
                        IMAGE_SIZE
                },
                checkpoint_path
            )

            print(
                f"  ★ New best model saved"
            )

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print(
                "  Early stopping."
            )

            break

    # --------------------------------------------------------
    # RESTORE BEST MODEL
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    final_metrics = evaluate(
        model,
        val_loader,
        criterion
    )

    result = {

        "model":
            model_name,

        "best_epoch":
            best_epoch,

        "best_val_roc_auc":
            best_auc,

        "final_metrics":
            final_metrics,

        "history":
            history
    }

    result_path = os.path.join(
        OUTPUT_DIR,
        f"{model_name}_results.json"
    )

    with open(
        result_path,
        "w"
    ) as f:

        json.dump(
            result,
            f,
            indent=4
        )

    print()
    print(
        f"{model_name} FINAL RESULTS"
    )

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
        f"F1       : "
        f"{final_metrics['f1']:.6f}"
    )

    print(
        f"ROC-AUC  : "
        f"{final_metrics['roc_auc']:.6f}"
    )

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

    print(
        f"Results saved to:"
    )

    print(
        result_path
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(
        SEED
    )

    print()
    print("=" * 70)
    print("MEDVERSE AI")
    print("EXPERIMENT 3 — X-RAY ARCHITECTURE COMPARISON")
    print("=" * 70)

    print(
        "Device:",
        DEVICE
    )

    print(
        "Image size:",
        IMAGE_SIZE
    )

    print(
        "Batch size:",
        BATCH_SIZE
    )

    print(
        "Maximum epochs:",
        MAX_EPOCHS
    )

    print(
        "Manifest:",
        MANIFEST_PATH
    )

    # ========================================================
    # LOAD MANIFEST
    # ========================================================

    if not os.path.exists(
        MANIFEST_PATH
    ):

        raise FileNotFoundError(
            f"Manifest not found:\n{MANIFEST_PATH}"
        )

    df = pd.read_csv(
        MANIFEST_PATH
    )

    required_columns = {
        "patient_id",
        "image_path",
        "label",
        "final_split"
    }

    missing = (
        required_columns
        - set(df.columns)
    )

    if missing:

        raise ValueError(
            f"Missing columns: {missing}"
        )

    # ========================================================
    # BASIC DATA CHECKS
    # ========================================================

    print()
    print("DATASET CHECK")
    print("-" * 70)

    print(
        "Total records:",
        len(df)
    )

    print(
        "Train records:",
        (df["final_split"] == "train").sum()
    )

    print(
        "Validation records:",
        (df["final_split"] == "val").sum()
    )

    print(
        "Labels:"
    )

    print(
        df["label"].value_counts()
        .sort_index()
    )

    # --------------------------------------------------------
    # PATIENT OVERLAP CHECK
    # --------------------------------------------------------

    train_patients = set(
        df.loc[
            df["final_split"] == "train",
            "patient_id"
        ]
    )

    val_patients = set(
        df.loc[
            df["final_split"] == "val",
            "patient_id"
        ]
    )

    overlap = (
        train_patients
        & val_patients
    )

    print(
        "Patient overlap:",
        len(overlap)
    )

    if len(overlap) != 0:

        raise ValueError(
            "Patient leakage detected!"
        )

    # ========================================================
    # SPLIT DATA
    # ========================================================

    train_df = df[
        df["final_split"] == "train"
    ].copy()

    val_df = df[
        df["final_split"] == "val"
    ].copy()

    # ========================================================
    # DATASETS
    # ========================================================

    train_dataset = XRayDataset(
        train_df,
        transform=train_transform
    )

    val_dataset = XRayDataset(
        val_df,
        transform=val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS
    )

    # ========================================================
    # CLASS WEIGHTS
    # ========================================================

    class_counts = (
        train_df["label"]
        .value_counts()
        .sort_index()
    )

    negative_count = int(
        class_counts.get(
            0,
            0
        )
    )

    positive_count = int(
        class_counts.get(
            1,
            0
        )
    )

    class_weights = torch.tensor(
        [
            1.0,
            negative_count / positive_count
        ],
        dtype=torch.float32,
        device=DEVICE
    )

    print()
    print(
        "Class counts:"
    )

    print(
        "NORMAL    :",
        negative_count
    )

    print(
        "PNEUMONIA :",
        positive_count
    )

    print(
        "Class weights:",
        class_weights.tolist()
    )

    # ========================================================
    # BUILD MODELS
    # ========================================================

    model_builders = {

        "densenet121":
            build_densenet,

        "resnet50":
            build_resnet,

        "efficientnet_b0":
            build_efficientnet
    }

    all_results = {}

    # ========================================================
    # TRAIN MODELS
    # ========================================================

    for model_name, builder in model_builders.items():

        # Reset seed so each architecture gets
        # deterministic data-loader/model initialization
        set_seed(SEED)

        print()
        print(
            f"Building {model_name}..."
        )

        model = builder()

        result = train_model(
            model_name,
            model,
            train_loader,
            val_loader,
            class_weights
        )

        all_results[
            model_name
        ] = result

    # ========================================================
    # FINAL COMPARISON
    # ========================================================

    comparison = []

    for model_name, result in all_results.items():

        metrics = result[
            "final_metrics"
        ]

        comparison.append({

            "model":
                model_name,

            "accuracy":
                metrics["accuracy"],

            "precision":
                metrics["precision"],

            "recall":
                metrics["recall"],

            "f1":
                metrics["f1"],

            "roc_auc":
                metrics["roc_auc"],

            "best_epoch":
                result["best_epoch"]
        })

    comparison_df = pd.DataFrame(
        comparison
    )

    comparison_path = os.path.join(
        OUTPUT_DIR,
        "xray_architecture_comparison.csv"
    )

    comparison_df.to_csv(
        comparison_path,
        index=False
    )

    all_results_path = os.path.join(
        OUTPUT_DIR,
        "xray_architecture_comparison.json"
    )

    with open(
        all_results_path,
        "w"
    ) as f:

        json.dump(
            all_results,
            f,
            indent=4
        )

    print()
    print("=" * 70)
    print("FINAL ARCHITECTURE COMPARISON")
    print("=" * 70)

    print(
        comparison_df.to_string(
            index=False
        )
    )

    print()
    print(
        "CSV saved:"
    )

    print(
        comparison_path
    )

    print()
    print(
        "JSON saved:"
    )

    print(
        all_results_path
    )


if __name__ == "__main__":

    main()