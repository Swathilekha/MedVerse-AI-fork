import os
import json
import time
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
    roc_auc_score
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

DATA_DIR = os.path.join(
    BASE_DIR,
    "data",
    "chexpert_final"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ai-services",
    "xray",
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ------------------------------------------------------------
# CPU-FRIENDLY SETTINGS
# ------------------------------------------------------------

DEFAULT_IMAGE_SIZE = 160
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 3

DEFAULT_LR = 1e-5


# ------------------------------------------------------------
# EXISTING MODEL FROM FEATURE-EXTRACTOR STAGE
# ------------------------------------------------------------

BASE_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_densenet121_cpu.pth"
)


# ------------------------------------------------------------
# FINE-TUNED MODEL OUTPUT
# ------------------------------------------------------------

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "best_densenet121_finetuned_cpu.pth"
)

HISTORY_PATH = os.path.join(
    OUTPUT_DIR,
    "finetune_training_history.json"
)


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser(
    description="CPU-friendly DenseNet121 fine-tuning"
)

parser.add_argument(
    "--epochs",
    type=int,
    default=DEFAULT_EPOCHS
)

parser.add_argument(
    "--batch-size",
    type=int,
    default=DEFAULT_BATCH_SIZE
)

parser.add_argument(
    "--image-size",
    type=int,
    default=DEFAULT_IMAGE_SIZE
)

parser.add_argument(
    "--lr",
    type=float,
    default=DEFAULT_LR
)

parser.add_argument(
    "--resume",
    action="store_true",
    help="Resume from existing fine-tuned checkpoint"
)

args = parser.parse_args()


# ============================================================
# FORCE CPU
# ============================================================

device = torch.device("cpu")


# ============================================================
# HEADER
# ============================================================

print("\n========================================")
print("MEDVERSE AI - DENSENET CPU FINE-TUNING")
print("========================================")

print("Device:", device)
print("Image size:", args.image_size)
print("Batch size:", args.batch_size)
print("Epochs:", args.epochs)
print("Learning rate:", args.lr)

print("\nFine-tuning strategy:")
print("DenseNet121 early layers: FROZEN")
print("DenseNet121 final dense block: TRAINABLE")
print("DenseNet121 norm5: TRAINABLE")
print("Classifier: TRAINABLE")
print("CPU-only transfer learning")


# ============================================================
# DATASET PATHS
# ============================================================

TRAIN_DIR = os.path.join(
    DATA_DIR,
    "train"
)

VAL_DIR = os.path.join(
    DATA_DIR,
    "val"
)

if not os.path.exists(TRAIN_DIR):
    raise FileNotFoundError(
        f"Training directory not found:\n{TRAIN_DIR}"
    )

if not os.path.exists(VAL_DIR):
    raise FileNotFoundError(
        f"Validation directory not found:\n{VAL_DIR}"
    )


# ============================================================
# TRANSFORMS
# ============================================================

train_transform = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (args.image_size, args.image_size)
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
        (args.image_size, args.image_size)
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DATASET
# ============================================================

print("\n========================================")
print("LOADING DATASET")
print("========================================")

train_dataset = datasets.ImageFolder(
    TRAIN_DIR,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    VAL_DIR,
    transform=val_transform
)

print("Classes:", train_dataset.classes)
print("Training images:", len(train_dataset))
print("Validation images:", len(val_dataset))


# ============================================================
# CLASS DISTRIBUTION
# ============================================================

train_targets = [
    label
    for _, label in train_dataset.samples
]

class_counts = {}

for class_index, class_name in enumerate(
    train_dataset.classes
):

    count = train_targets.count(
        class_index
    )

    class_counts[class_name] = count


print("\nTraining class distribution:")

for class_name, count in class_counts.items():

    print(
        f"{class_name}: {count}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

total_samples = len(train_targets)

num_classes = len(
    train_dataset.classes
)

class_weights = []

for class_index in range(num_classes):

    count = train_targets.count(
        class_index
    )

    weight = total_samples / (
        num_classes * count
    )

    class_weights.append(
        weight
    )


class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print("\nClass weights:")

for class_name, weight in zip(
    train_dataset.classes,
    class_weights
):

    print(
        f"{class_name}: {weight:.4f}"
    )


# ============================================================
# DATALOADERS
# ============================================================

train_loader = DataLoader(
    train_dataset,
    batch_size=args.batch_size,
    shuffle=True,
    num_workers=0
)

val_loader = DataLoader(
    val_dataset,
    batch_size=args.batch_size,
    shuffle=False,
    num_workers=0
)


# ============================================================
# LOAD DENSENET121
# ============================================================

print("\n========================================")
print("LOADING DENSENET121")
print("========================================")

print("Loading ImageNet architecture...")

model = models.densenet121(
    weights=models.DenseNet121_Weights.DEFAULT
)


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

num_features = model.classifier.in_features

model.classifier = nn.Linear(
    num_features,
    num_classes
)


# ============================================================
# LOAD PREVIOUS FEATURE-EXTRACTOR CHECKPOINT
# ============================================================

if not os.path.exists(BASE_MODEL_PATH):

    raise FileNotFoundError(
        "\nPrevious DenseNet checkpoint not found:\n"
        f"{BASE_MODEL_PATH}\n\n"
        "The feature-extractor training must be completed "
        "before fine-tuning."
    )


print("\nLoading previous trained model:")

print(
    BASE_MODEL_PATH
)


checkpoint = torch.load(
    BASE_MODEL_PATH,
    map_location="cpu"
)


# ------------------------------------------------------------
# HANDLE OUR CHECKPOINT FORMAT
# ------------------------------------------------------------

if isinstance(
    checkpoint,
    dict
) and "model_state_dict" in checkpoint:

    state_dict = checkpoint[
        "model_state_dict"
    ]

else:

    state_dict = checkpoint


# ------------------------------------------------------------
# LOAD WEIGHTS
# ------------------------------------------------------------

missing_keys, unexpected_keys = (
    model.load_state_dict(
        state_dict,
        strict=False
    )
)


if missing_keys:

    print(
        "\nMissing keys while loading checkpoint:"
    )

    for key in missing_keys[:10]:

        print(
            " ",
            key
        )


if unexpected_keys:

    print(
        "\nUnexpected keys:"
    )

    for key in unexpected_keys[:10]:

        print(
            " ",
            key
        )


print(
    "\nPrevious feature-extractor weights loaded."
)


# ============================================================
# FREEZE ENTIRE MODEL
# ============================================================

for param in model.parameters():

    param.requires_grad = False


# ============================================================
# UNFREEZE FINAL DENSE BLOCK
# ============================================================

for param in (
    model.features.denseblock4.parameters()
):

    param.requires_grad = True


# ============================================================
# UNFREEZE FINAL NORMALIZATION
# ============================================================

for param in (
    model.features.norm5.parameters()
):

    param.requires_grad = True


# ============================================================
# CLASSIFIER TRAINABLE
# ============================================================

for param in (
    model.classifier.parameters()
):

    param.requires_grad = True


# ============================================================
# RESUME FINE-TUNED MODEL IF REQUESTED
# ============================================================

if args.resume and os.path.exists(
    MODEL_PATH
):

    print(
        "\n========================================"
    )

    print(
        "RESUMING PREVIOUS FINE-TUNING"
    )

    print(
        "========================================"
    )

    resume_checkpoint = torch.load(
        MODEL_PATH,
        map_location="cpu"
    )

    if (
        isinstance(
            resume_checkpoint,
            dict
        )
        and
        "model_state_dict"
        in resume_checkpoint
    ):

        model.load_state_dict(
            resume_checkpoint[
                "model_state_dict"
            ]
        )

        print(
            "Fine-tuned checkpoint loaded."
        )

    else:

        model.load_state_dict(
            resume_checkpoint
        )

        print(
            "Model weights loaded."
        )


# ============================================================
# MOVE TO CPU
# ============================================================

model = model.to(device)


# ============================================================
# PARAMETER INFORMATION
# ============================================================

total_params = sum(
    p.numel()
    for p in model.parameters()
)

trainable_params = sum(
    p.numel()
    for p in model.parameters()
    if p.requires_grad
)

frozen_params = (
    total_params
    - trainable_params
)


print("\nModel parameters:")

print(
    f"Total parameters: "
    f"{total_params:,}"
)

print(
    f"Trainable parameters: "
    f"{trainable_params:,}"
)

print(
    f"Frozen parameters: "
    f"{frozen_params:,}"
)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(device)
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.AdamW(
    filter(
        lambda p: p.requires_grad,
        model.parameters()
    ),
    lr=args.lr,
    weight_decay=1e-4
)


# ============================================================
# LEARNING RATE SCHEDULER
# ============================================================

scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
    optimizer,
    mode="max",
    factor=0.5,
    patience=1
)


# ============================================================
# HISTORY
# ============================================================

history = []

best_val_f1 = -1.0


# If resuming, load previous history
if args.resume and os.path.exists(
    HISTORY_PATH
):

    try:

        with open(
            HISTORY_PATH,
            "r",
            encoding="utf-8"
        ) as f:

            history = json.load(f)

        if history:

            best_val_f1 = max(
                item.get(
                    "pneumonia_f1",
                    -1.0
                )
                for item in history
            )

            print(
                f"\nPrevious best "
                f"Pneumonia F1: "
                f"{best_val_f1:.4f}"
            )

    except Exception:

        print(
            "\nCould not load previous history."
        )

        history = []


# ============================================================
# TRAINING
# ============================================================

print("\n========================================")
print("STARTING CPU FINE-TUNING")
print("========================================")

training_start = time.time()


starting_epoch = len(history)


for epoch in range(
    starting_epoch,
    starting_epoch + args.epochs
):

    epoch_start = time.time()

    print("\n----------------------------------------")

    print(
        f"EPOCH {epoch + 1}"
    )

    print("----------------------------------------")


    # ========================================================
    # TRAIN MODE
    # ========================================================

    model.train()

    running_loss = 0.0

    correct = 0

    total = 0

    num_batches = len(
        train_loader
    )


    for batch_index, (
        images,
        labels
    ) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(device)

        labels = labels.to(device)


        optimizer.zero_grad()


        outputs = model(
            images
        )


        loss = criterion(
            outputs,
            labels
        )


        loss.backward()


        optimizer.step()


        running_loss += (
            loss.item()
            * images.size(0)
        )


        predictions = torch.argmax(
            outputs,
            dim=1
        )


        correct += (
            predictions == labels
        ).sum().item()


        total += labels.size(0)


        if (
            batch_index % 25 == 0
            or
            batch_index == num_batches
        ):

            current_acc = (
                correct / total
            )

            print(
                f"Batch "
                f"{batch_index}/{num_batches} "
                f"| Loss: "
                f"{loss.item():.4f} "
                f"| Acc: "
                f"{current_acc:.4f}"
            )


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_running_loss = 0.0

    val_predictions = []

    val_labels_all = []

    val_probabilities = []


    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)

            labels = labels.to(device)


            outputs = model(
                images
            )


            loss = criterion(
                outputs,
                labels
            )


            val_running_loss += (
                loss.item()
                * images.size(0)
            )


            probabilities = torch.softmax(
                outputs,
                dim=1
            )


            predictions = torch.argmax(
                probabilities,
                dim=1
            )


            val_predictions.extend(
                predictions.cpu().numpy()
            )


            val_labels_all.extend(
                labels.cpu().numpy()
            )


            val_probabilities.extend(
                probabilities[:, 1]
                .cpu()
                .numpy()
            )


    val_loss = (
        val_running_loss
        / len(val_dataset)
    )


    val_accuracy = accuracy_score(
        val_labels_all,
        val_predictions
    )


    val_precision = precision_score(
        val_labels_all,
        val_predictions,
        pos_label=1,
        zero_division=0
    )


    val_recall = recall_score(
        val_labels_all,
        val_predictions,
        pos_label=1,
        zero_division=0
    )


    val_f1 = f1_score(
        val_labels_all,
        val_predictions,
        pos_label=1,
        zero_division=0
    )


    try:

        val_auc = roc_auc_score(
            val_labels_all,
            val_probabilities
        )

    except ValueError:

        val_auc = 0.0


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler.step(
        val_f1
    )


    current_lr = optimizer.param_groups[0][
        "lr"
    ]


    epoch_time = (
        time.time()
        - epoch_start
    )


    # ========================================================
    # RESULTS
    # ========================================================

    print("\nEPOCH RESULT")

    print(
        f"Train Loss      : "
        f"{train_loss:.4f}"
    )

    print(
        f"Train Accuracy  : "
        f"{train_accuracy:.4f}"
    )

    print(
        f"Val Loss        : "
        f"{val_loss:.4f}"
    )

    print(
        f"Val Accuracy    : "
        f"{val_accuracy:.4f}"
    )

    print(
        f"Pneumonia Precision: "
        f"{val_precision:.4f}"
    )

    print(
        f"Pneumonia Recall   : "
        f"{val_recall:.4f}"
    )

    print(
        f"Pneumonia F1       : "
        f"{val_f1:.4f}"
    )

    print(
        f"ROC-AUC            : "
        f"{val_auc:.4f}"
    )

    print(
        f"Learning Rate      : "
        f"{current_lr:.8f}"
    )

    print(
        f"Time               : "
        f"{epoch_time / 60:.2f} minutes"
    )


    # ========================================================
    # SAVE HISTORY
    # ========================================================

    epoch_record = {

        "epoch":
            epoch + 1,

        "train_loss":
            train_loss,

        "train_accuracy":
            train_accuracy,

        "val_loss":
            val_loss,

        "val_accuracy":
            val_accuracy,

        "pneumonia_precision":
            val_precision,

        "pneumonia_recall":
            val_recall,

        "pneumonia_f1":
            val_f1,

        "roc_auc":
            val_auc,

        "learning_rate":
            current_lr,

        "epoch_time_minutes":
            epoch_time / 60
    }


    history.append(
        epoch_record
    )


    with open(
        HISTORY_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            history,
            f,
            indent=4
        )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if val_f1 > best_val_f1:

        best_val_f1 = val_f1

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    train_dataset.classes,

                "image_size":
                    args.image_size,

                "model":
                    "DenseNet121",

                "experiment":
                    "chexpert_plus_final_dense_block_finetuning",

                "best_val_f1":
                    best_val_f1,

                "val_accuracy":
                    val_accuracy,

                "val_precision":
                    val_precision,

                "val_recall":
                    val_recall,

                "val_auc":
                    val_auc,

                "epoch":
                    epoch + 1
            },
            MODEL_PATH
        )


        print(
            "\nBEST MODEL SAVED"
        )

        print(
            f"Pneumonia F1: "
            f"{best_val_f1:.4f}"
        )

        print(
            f"Path: "
            f"{MODEL_PATH}"
        )

    else:

        print(
            "\nNo improvement "
            "over previous best."
        )


# ============================================================
# COMPLETE
# ============================================================

total_time = (
    time.time()
    - training_start
)


print("\n========================================")
print("FINE-TUNING COMPLETE")
print("========================================")

print(
    f"Best Pneumonia F1: "
    f"{best_val_f1:.4f}"
)

print(
    f"Total training time: "
    f"{total_time / 60:.2f} minutes"
)

print("\nBest model:")

print(
    MODEL_PATH
)

print("\nTraining history:")

print(
    HISTORY_PATH
)

print("\n========================================")