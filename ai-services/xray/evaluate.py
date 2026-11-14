import os
import json
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report
)

import matplotlib.pyplot as plt


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "chexpert_final"
)

VAL_DIR = os.path.join(
    DATA_DIR,
    "val"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ai-services",
    "xray",
    "outputs"
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_densenet121_finetuned_cpu.pth"
)

IMAGE_SIZE = 160
BATCH_SIZE = 8

DEVICE = torch.device("cpu")

CLASSES = [
    "NORMAL",
    "PNEUMONIA"
]


# ============================================================
# HEADER
# ============================================================

print("\n========================================")
print("MEDVERSE AI - CHEXPERT PLUS EVALUATION")
print("========================================")

print("Device:", DEVICE)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)

print("\nModel:")
print(MODEL_PATH)


# ============================================================
# CHECK MODEL
# ============================================================

if not os.path.exists(MODEL_PATH):

    raise FileNotFoundError(
        f"Model not found:\n{MODEL_PATH}"
    )


# ============================================================
# TRANSFORM
# ============================================================

transform = transforms.Compose([

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
# DATASET
# ============================================================

print("\nLoading validation dataset...")

dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=transform
)

print("Classes:", dataset.classes)
print("Validation images:", len(dataset))


loader = DataLoader(
    dataset,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0
)


# ============================================================
# MODEL
# ============================================================

print("\nLoading DenseNet121...")

model = models.densenet121(
    weights=None
)

model.classifier = nn.Linear(
    model.classifier.in_features,
    len(CLASSES)
)


checkpoint = torch.load(
    MODEL_PATH,
    map_location=DEVICE
)

if "model_state_dict" in checkpoint:

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

else:

    model.load_state_dict(
        checkpoint
    )


model = model.to(DEVICE)

model.eval()


# ============================================================
# PREDICTION
# ============================================================

print("\n========================================")
print("RUNNING EVALUATION")
print("========================================")

all_labels = []
all_predictions = []
all_probabilities = []

total_batches = len(loader)

with torch.no_grad():

    for batch_index, (images, labels) in enumerate(
        loader,
        start=1
    ):

        images = images.to(DEVICE)

        outputs = model(images)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        predictions = torch.argmax(
            probabilities,
            dim=1
        )

        all_labels.extend(
            labels.numpy()
        )

        all_predictions.extend(
            predictions.cpu().numpy()
        )

        # Probability of PNEUMONIA
        all_probabilities.extend(
            probabilities[:, 1]
            .cpu()
            .numpy()
        )

        if (
            batch_index % 25 == 0
            or batch_index == total_batches
        ):

            print(
                f"Processed batches: "
                f"{batch_index}/{total_batches}"
            )


# ============================================================
# METRICS
# ============================================================

accuracy = accuracy_score(
    all_labels,
    all_predictions
)

precision = precision_score(
    all_labels,
    all_predictions,
    zero_division=0
)

recall = recall_score(
    all_labels,
    all_predictions,
    zero_division=0
)

f1 = f1_score(
    all_labels,
    all_predictions,
    zero_division=0
)

roc_auc = roc_auc_score(
    all_labels,
    all_probabilities
)

cm = confusion_matrix(
    all_labels,
    all_predictions
)

report = classification_report(
    all_labels,
    all_predictions,
    target_names=CLASSES,
    zero_division=0
)


# ============================================================
# PRINT RESULTS
# ============================================================

print("\n========================================")
print("CHEXPERT PLUS RESULTS")
print("========================================")

print(
    f"Accuracy : {accuracy:.4f}"
)

print(
    f"Precision: {precision:.4f}"
)

print(
    f"Recall   : {recall:.4f}"
)

print(
    f"F1-score : {f1:.4f}"
)

print(
    f"ROC-AUC  : {roc_auc:.4f}"
)


print("\nClassification Report:")

print(report)


print("\nConfusion Matrix:")

print(cm)


# ============================================================
# SAVE METRICS
# ============================================================

metrics = {

    "dataset":
        "CheXpert Plus",

    "model":
        "DenseNet121",

    "image_size":
        IMAGE_SIZE,

    "validation_images":
        len(dataset),

    "accuracy":
        float(accuracy),

    "precision":
        float(precision),

    "recall":
        float(recall),

    "f1_score":
        float(f1),

    "roc_auc":
        float(roc_auc),

    "confusion_matrix":
        cm.tolist()
}


metrics_path = os.path.join(
    OUTPUT_DIR,
    "chexpert_metrics.json"
)

with open(
    metrics_path,
    "w"
) as f:

    json.dump(
        metrics,
        f,
        indent=4
    )


# ============================================================
# SAVE CLASSIFICATION REPORT
# ============================================================

report_path = os.path.join(
    OUTPUT_DIR,
    "chexpert_classification_report.txt"
)

with open(
    report_path,
    "w"
) as f:

    f.write(report)


# ============================================================
# CONFUSION MATRIX IMAGE
# ============================================================

plt.figure(
    figsize=(6, 5)
)

plt.imshow(cm)

plt.title(
    "CheXpert Plus - DenseNet121"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.xticks(
    range(len(CLASSES)),
    CLASSES
)

plt.yticks(
    range(len(CLASSES)),
    CLASSES
)

for i in range(len(CLASSES)):

    for j in range(len(CLASSES)):

        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.tight_layout()

cm_path = os.path.join(
    OUTPUT_DIR,
    "chexpert_confusion_matrix.png"
)

plt.savefig(
    cm_path,
    dpi=200
)

plt.close()


# ============================================================
# COMPLETE
# ============================================================

print("\n========================================")
print("EVALUATION COMPLETE")
print("========================================")

print(
    "Metrics:",
    metrics_path
)

print(
    "Classification report:",
    report_path
)

print(
    "Confusion matrix:",
    cm_path
)

print("========================================")