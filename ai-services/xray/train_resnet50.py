import os
import json
import random
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms
from PIL import Image

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

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(os.path.abspath(__file__))
    )
)

MANIFEST = os.path.join(
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

DEVICE = torch.device("cpu")

IMAGE_SIZE = 160
BATCH_SIZE = 8
EPOCHS = 10
PATIENCE = 3

LR = 1e-5
WEIGHT_DECAY = 1e-4
SEED = 42


# ============================================================
# SEED
# ============================================================

def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


# ============================================================
# DATASET
# ============================================================

class XRayDataset(Dataset):

    def __init__(self, dataframe, transform):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        path = row["image_path"]

        if not os.path.isabs(path):
            path = os.path.join(BASE_DIR, path)

        path = os.path.normpath(path)

        image = Image.open(path).convert("L")

        label = int(row["label"])

        image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomRotation(degrees=5),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        [0.485, 0.456, 0.406],
        [0.229, 0.224, 0.225]
    )
])


# ============================================================
# MODEL
# ============================================================

def build_model():

    model = models.resnet50(
        weights=models.ResNet50_Weights.DEFAULT
    )

    in_features = model.fc.in_features

    model.fc = nn.Linear(
        in_features,
        2
    )

    return model


# ============================================================
# METRICS
# ============================================================

def metrics(labels, probs):

    preds = (probs >= 0.5).astype(int)

    return {
        "accuracy": float(
            accuracy_score(labels, preds)
        ),

        "precision": float(
            precision_score(
                labels,
                preds,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                labels,
                preds,
                zero_division=0
            )
        ),

        "f1": float(
            f1_score(
                labels,
                preds,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                labels,
                probs
            )
        ),

        "confusion_matrix":
            confusion_matrix(
                labels,
                preds,
                labels=[0, 1]
            ).tolist()
    }


# ============================================================
# EVALUATION
# ============================================================

def evaluate(model, loader, criterion):

    model.eval()

    total_loss = 0

    labels_all = []
    probs_all = []

    with torch.no_grad():

        for images, labels in loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            probs = torch.softmax(
                logits,
                dim=1
            )[:, 1]

            labels_all.extend(
                labels.cpu().numpy()
            )

            probs_all.extend(
                probs.cpu().numpy()
            )

    result = metrics(
        np.array(labels_all),
        np.array(probs_all)
    )

    result["loss"] = (
        total_loss / len(loader.dataset)
    )

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    print("=" * 70)
    print("MEDVERSE AI — RESNET50 X-RAY EXPERIMENT")
    print("=" * 70)

    print("Device:", DEVICE)

    # --------------------------------------------------------
    # LOAD MANIFEST
    # --------------------------------------------------------

    df = pd.read_csv(MANIFEST)

    train_df = df[
        df["final_split"] == "train"
    ].copy()

    val_df = df[
        df["final_split"] == "val"
    ].copy()

    print("Total:", len(df))
    print("Train:", len(train_df))
    print("Validation:", len(val_df))

    # --------------------------------------------------------
    # PATIENT LEAKAGE CHECK
    # --------------------------------------------------------

    train_patients = set(
        train_df["patient_id"]
    )

    val_patients = set(
        val_df["patient_id"]
    )

    overlap = train_patients & val_patients

    print("Patient overlap:", len(overlap))

    if overlap:
        raise RuntimeError(
            "Patient leakage detected!"
        )

    # --------------------------------------------------------
    # DATA
    # --------------------------------------------------------

    train_dataset = XRayDataset(
        train_df,
        train_transform
    )

    val_dataset = XRayDataset(
        val_df,
        val_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # CLASS WEIGHT
    # --------------------------------------------------------

    negative = int(
        (train_df["label"] == 0).sum()
    )

    positive = int(
        (train_df["label"] == 1).sum()
    )

    pos_weight = negative / positive

    class_weights = torch.tensor(
        [1.0, pos_weight],
        dtype=torch.float32
    ).to(DEVICE)

    print(
        "NORMAL:",
        negative,
        "PNEUMONIA:",
        positive
    )

    print(
        "Class weights:",
        class_weights.tolist()
    )

    # --------------------------------------------------------
    # MODEL
    # --------------------------------------------------------

    model = build_model().to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        weight=class_weights
    )

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY
    )

    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="max",
        factor=0.5,
        patience=1
    )

    best_auc = -1
    best_epoch = 0
    best_state = None
    patience_counter = 0

    history = []

    # --------------------------------------------------------
    # TRAIN
    # --------------------------------------------------------

    for epoch in range(1, EPOCHS + 1):

        model.train()

        total_loss = 0

        for images, labels in train_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            optimizer.zero_grad(
                set_to_none=True
            )

            logits = model(images)

            loss = criterion(
                logits,
                labels
            )

            loss.backward()

            torch.nn.utils.clip_grad_norm_(
                model.parameters(),
                1.0
            )

            optimizer.step()

            total_loss += (
                loss.item()
                * labels.size(0)
            )

        train_loss = (
            total_loss
            / len(train_loader.dataset)
        )

        val_result = evaluate(
            model,
            val_loader,
            criterion
        )

        scheduler.step(
            val_result["roc_auc"]
        )

        print(
            f"\nEpoch {epoch}/{EPOCHS}"
        )

        print(
            f"Train Loss : {train_loss:.4f}"
        )

        print(
            f"Val Acc    : "
            f"{val_result['accuracy']:.4f}"
        )

        print(
            f"Val F1     : "
            f"{val_result['f1']:.4f}"
        )

        print(
            f"Val ROC-AUC: "
            f"{val_result['roc_auc']:.4f}"
        )

        history.append({
            "epoch": epoch,
            "train_loss": train_loss,
            "val": val_result
        })

        # ----------------------------------------------------
        # BEST MODEL
        # ----------------------------------------------------

        if val_result["roc_auc"] > best_auc:

            best_auc = val_result["roc_auc"]
            best_epoch = epoch

            best_state = {
                k: v.cpu().clone()
                for k, v in model.state_dict().items()
            }

            patience_counter = 0

            print("★ New best model")

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print("Early stopping.")

            break

    # --------------------------------------------------------
    # FINAL BEST MODEL
    # --------------------------------------------------------

    model.load_state_dict(
        best_state
    )

    final_result = evaluate(
        model,
        val_loader,
        criterion
    )

    print("\n" + "=" * 70)
    print("RESNET50 FINAL RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {final_result['accuracy']:.6f}"
    )

    print(
        f"Precision: {final_result['precision']:.6f}"
    )

    print(
        f"Recall   : {final_result['recall']:.6f}"
    )

    print(
        f"F1       : {final_result['f1']:.6f}"
    )

    print(
        f"ROC-AUC  : {final_result['roc_auc']:.6f}"
    )

    print("Confusion Matrix:")
    print(
        np.array(
            final_result["confusion_matrix"]
        )
    )

    # --------------------------------------------------------
    # SAVE
    # --------------------------------------------------------

    checkpoint = os.path.join(
        OUTPUT_DIR,
        "resnet50_best.pt"
    )

    torch.save(
        {
            "model_state_dict": best_state,
            "model": "resnet50",
            "best_epoch": best_epoch,
            "best_val_roc_auc": best_auc,
            "image_size": IMAGE_SIZE
        },
        checkpoint
    )

    result_path = os.path.join(
        OUTPUT_DIR,
        "resnet50_results.json"
    )

    with open(
        result_path,
        "w"
    ) as f:

        json.dump(
            {
                "model": "ResNet50",
                "best_epoch": best_epoch,
                "final_metrics": final_result,
                "history": history
            },
            f,
            indent=4
        )

    print("\nSaved:")
    print(checkpoint)
    print(result_path)


if __name__ == "__main__":
    main()