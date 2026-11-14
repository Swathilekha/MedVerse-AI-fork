import os
import json
import numpy as np
import pandas as pd
import torch
import torch.nn as nn

from PIL import Image
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
# PATHS
# ============================================================

MANIFEST = "data/clinical_text/fusion_experiment_dataset.csv"

CHECKPOINT = "ai-services/xray/outputs/best_densenet121_finetuned_cpu.pth"

OUTPUT_DIR = "ai-services/xray/architecture_comparison"
RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "densenet121_common_val_results.json"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ============================================================
# CONFIG
# ============================================================

IMAGE_SIZE = 160
BATCH_SIZE = 8

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


print("=" * 70)
print("MEDVERSE AI — DENSENET121 COMMON VALIDATION EVALUATION")
print("=" * 70)

print("Device:", DEVICE)


# ============================================================
# DATASET
# ============================================================

class XRayDataset(Dataset):

    def __init__(self, dataframe, transform=None):

        self.df = dataframe.reset_index(drop=True)
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        image_path = row["image_path"]
        label = int(row["label"])

        image = Image.open(image_path).convert("L")

        if self.transform:
            image = self.transform(image)

        return image, label


# ============================================================
# TRANSFORM
# ============================================================

val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD MANIFEST
# ============================================================

df = pd.read_csv(MANIFEST)

print("\nManifest records:", len(df))

train_df = df[df["final_split"] == "train"].copy()
val_df = df[df["final_split"] == "val"].copy()

print("Train:", len(train_df))
print("Validation:", len(val_df))


# ============================================================
# PATIENT OVERLAP CHECK
# ============================================================

train_patients = set(train_df["patient_id"])
val_patients = set(val_df["patient_id"])

overlap = train_patients.intersection(val_patients)

print("Patient overlap:", len(overlap))

if len(overlap) > 0:
    raise RuntimeError(
        f"Patient leakage detected: {len(overlap)} overlapping patients"
    )


# ============================================================
# LABEL DISTRIBUTION
# ============================================================

print(
    "Validation NORMAL:",
    int((val_df["label"] == 0).sum())
)

print(
    "Validation PNEUMONIA:",
    int((val_df["label"] == 1).sum())
)


# ============================================================
# DATA LOADER
# ============================================================

val_dataset = XRayDataset(
    val_df,
    transform=val_transform
)

val_loader = DataLoader(
    val_dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)

print("Validation batches:", len(val_loader))


# ============================================================
# LOAD DENSENET121
# ============================================================

print("\nLoading DenseNet121 checkpoint...")

checkpoint = torch.load(
    CHECKPOINT,
    map_location=DEVICE
)

model = models.densenet121(
    weights=None,
    num_classes=2
)

model.load_state_dict(
    checkpoint["model_state_dict"],
    strict=True
)

model = model.to(DEVICE)
model.eval()

print("Checkpoint loaded successfully.")

if isinstance(checkpoint, dict):

    print("\nCheckpoint information:")

    for key in [
        "classes",
        "image_size",
        "experiment",
        "best_val_f1",
        "val_accuracy",
        "val_precision",
        "val_recall",
        "val_auc",
        "epoch"
    ]:

        if key in checkpoint:
            print(f"{key}: {checkpoint[key]}")


# ============================================================
# EVALUATION
# ============================================================

all_labels = []
all_predictions = []
all_probabilities = []

print("\nRunning evaluation...")

with torch.no_grad():

    for batch_idx, (images, labels) in enumerate(val_loader):

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )[:, 1]

        predictions = (
            probabilities >= 0.5
        ).long()

        all_labels.extend(
            labels.numpy().tolist()
        )

        all_predictions.extend(
            predictions.cpu().numpy().tolist()
        )

        all_probabilities.extend(
            probabilities.cpu().numpy().tolist()
        )

        if (batch_idx + 1) % 20 == 0:
            print(
                f"Processed batches: "
                f"{batch_idx + 1}/{len(val_loader)}"
            )


# ============================================================
# METRICS
# ============================================================

y_true = np.array(all_labels)
y_pred = np.array(all_predictions)
y_prob = np.array(all_probabilities)

accuracy = accuracy_score(
    y_true,
    y_pred
)

precision = precision_score(
    y_true,
    y_pred,
    zero_division=0
)

recall = recall_score(
    y_true,
    y_pred,
    zero_division=0
)

f1 = f1_score(
    y_true,
    y_pred,
    zero_division=0
)

roc_auc = roc_auc_score(
    y_true,
    y_prob
)

cm = confusion_matrix(
    y_true,
    y_pred
)

# ============================================================
# SAVE PER-RECORD PREDICTIONS
# ============================================================

prediction_df = val_df[
    ["patient_id", "image_path", "label"]
].copy()

prediction_df["prediction"] = y_pred
prediction_df["probability_pneumonia"] = y_prob

PREDICTIONS_PATH = os.path.join(
    OUTPUT_DIR,
    "densenet121_common_val_predictions.csv"
)

prediction_df.to_csv(
    PREDICTIONS_PATH,
    index=False
)

print("\nPer-record predictions saved:")
print(os.path.abspath(PREDICTIONS_PATH))

# ============================================================
# RESULTS
# ============================================================

print("\n")
print("=" * 70)
print("DENSENET121 COMMON VALIDATION RESULTS")
print("=" * 70)

print(f"Accuracy : {accuracy:.6f}")
print(f"Precision: {precision:.6f}")
print(f"Recall   : {recall:.6f}")
print(f"F1       : {f1:.6f}")
print(f"ROC-AUC  : {roc_auc:.6f}")

print("\nConfusion Matrix:")
print(cm)


# ============================================================
# SAVE RESULTS
# ============================================================

results = {

    "model": "DenseNet121",

    "evaluation_type":
        "Existing checkpoint evaluated on final clean validation split",

    "manifest":
        MANIFEST,

    "checkpoint":
        CHECKPOINT,

    "device":
        str(DEVICE),

    "image_size":
        IMAGE_SIZE,

    "validation_records":
        int(len(val_df)),

    "validation_normal":
        int((val_df["label"] == 0).sum()),

    "validation_pneumonia":
        int((val_df["label"] == 1).sum()),

    "patient_overlap":
        int(len(overlap)),

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1":
        float(f1),

    "roc_auc":
        float(roc_auc),

    "confusion_matrix":
        cm.tolist()
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


print("\nSaved:")
print(
    os.path.abspath(RESULTS_PATH)
)

print("=" * 70)