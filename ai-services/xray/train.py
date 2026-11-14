import os
import json
import time
import copy
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models


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

DEVICE = torch.device("cpu")

IMAGE_SIZE = 160
DEFAULT_BATCH_SIZE = 8
DEFAULT_EPOCHS = 5
LEARNING_RATE = 0.001

NUM_WORKERS = 0

CLASSES = [
    "NORMAL",
    "PNEUMONIA"
]


# ============================================================
# ARGUMENTS
# ============================================================

parser = argparse.ArgumentParser()

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
    default=IMAGE_SIZE
)

args = parser.parse_args()

EPOCHS = args.epochs
BATCH_SIZE = args.batch_size
IMAGE_SIZE = args.image_size


# ============================================================
# HEADER
# ============================================================

print("\n========================================")
print("MEDVERSE AI - CPU TRAINING")
print("========================================")

print("Device:", DEVICE)
print("Image size:", IMAGE_SIZE)
print("Batch size:", BATCH_SIZE)
print("Epochs:", EPOCHS)
print("Learning rate:", LEARNING_RATE)

print("\nTraining mode:")
print("DenseNet121 feature extractor: FROZEN")
print("Classifier: TRAINABLE")
print("CPU-friendly transfer learning")


# ============================================================
# DATA DIRECTORIES
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
    transforms.Grayscale(num_output_channels=3),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.RandomHorizontalFlip(
        p=0.5
    ),

    transforms.ToTensor(),

    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Grayscale(num_output_channels=3),

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
# DATASETS
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

train_counts = [0] * len(train_dataset.classes)

for _, label in train_dataset.samples:
    train_counts[label] += 1

print("\nTraining class distribution:")

for i, class_name in enumerate(
    train_dataset.classes
):
    print(
        f"{class_name}: {train_counts[i]}"
    )


# ============================================================
# CLASS WEIGHTS
# ============================================================

total_train = sum(train_counts)

class_weights = []

for count in train_counts:

    weight = total_train / (
        len(train_counts) * count
    )

    class_weights.append(weight)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print("\nClass weights:")

for i, class_name in enumerate(
    train_dataset.classes
):
    print(
        f"{class_name}: "
        f"{class_weights[i].item():.4f}"
    )


# ============================================================
# DATALOADERS
# ============================================================

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


# ============================================================
# MODEL
# ============================================================

print("\n========================================")
print("LOADING DENSENET121")
print("========================================")

try:

    weights = (
        models.DenseNet121_Weights.DEFAULT
    )

    model = models.densenet121(
        weights=weights
    )

    print(
        "Pretrained ImageNet weights loaded."
    )

except Exception as e:

    print(
        "Could not load pretrained weights."
    )

    print(
        "Reason:",
        e
    )

    print(
        "Loading DenseNet121 without "
        "pretrained weights."
    )

    model = models.densenet121(
        weights=None
    )


# ============================================================
# FREEZE FEATURE EXTRACTOR
# ============================================================

for parameter in model.features.parameters():

    parameter.requires_grad = False


# ============================================================
# REPLACE CLASSIFIER
# ============================================================

input_features = (
    model.classifier.in_features
)

model.classifier = nn.Linear(
    input_features,
    len(CLASSES)
)


model = model.to(DEVICE)


# ============================================================
# LOSS
# ============================================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights.to(DEVICE)
)


# ============================================================
# OPTIMIZER
# ============================================================

optimizer = torch.optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# ============================================================
# TRAINING
# ============================================================

print("\n========================================")
print("STARTING CPU TRAINING")
print("========================================")

best_accuracy = 0.0
best_model_weights = copy.deepcopy(
    model.state_dict()
)

history = []

start_time = time.time()


for epoch in range(EPOCHS):

    epoch_start = time.time()

    print("\n----------------------------------------")
    print(
        f"EPOCH {epoch + 1}/{EPOCHS}"
    )
    print("----------------------------------------")


    # ========================================================
    # TRAIN
    # ========================================================

    model.train()

    running_loss = 0.0
    running_correct = 0
    total = 0

    total_batches = len(train_loader)

    for batch_index, (images, labels) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        running_loss += (
            loss.item() *
            images.size(0)
        )

        running_correct += (
            (predictions == labels)
            .sum()
            .item()
        )

        total += images.size(0)

        if (
            batch_index % 25 == 0
            or batch_index == total_batches
        ):

            current_loss = (
                running_loss / total
            )

            current_accuracy = (
                running_correct / total
            )

            print(
                f"Batch "
                f"{batch_index}/{total_batches} "
                f"| Loss: {current_loss:.4f} "
                f"| Acc: "
                f"{current_accuracy:.4f}"
            )


    train_loss = (
        running_loss / total
    )

    train_accuracy = (
        running_correct / total
    )


    # ========================================================
    # VALIDATION
    # ========================================================

    model.eval()

    val_loss = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(DEVICE)
            labels = labels.to(DEVICE)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_loss += (
                loss.item() *
                images.size(0)
            )

            val_correct += (
                (predictions == labels)
                .sum()
                .item()
            )

            val_total += images.size(0)


    validation_loss = (
        val_loss / val_total
    )

    validation_accuracy = (
        val_correct / val_total
    )


    # ========================================================
    # EPOCH RESULT
    # ========================================================

    epoch_time = (
        time.time() -
        epoch_start
    )

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
        f"{validation_loss:.4f}"
    )

    print(
        f"Val Accuracy    : "
        f"{validation_accuracy:.4f}"
    )

    print(
        f"Time            : "
        f"{epoch_time / 60:.2f} minutes"
    )


    # ========================================================
    # SAVE BEST MODEL
    # ========================================================

    if validation_accuracy > best_accuracy:

        best_accuracy = (
            validation_accuracy
        )

        best_model_weights = (
            copy.deepcopy(
                model.state_dict()
            )
        )

        best_model_path = os.path.join(
            OUTPUT_DIR,
            "best_densenet121_cpu.pth"
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    CLASSES,

                "image_size":
                    IMAGE_SIZE,

                "best_val_accuracy":
                    best_accuracy
            },
            best_model_path
        )

        print(
            "\nBEST MODEL SAVED"
        )

        print(
            f"Validation accuracy: "
            f"{best_accuracy:.4f}"
        )

        print(
            f"Path: {best_model_path}"
        )


    history.append(
        {
            "epoch":
                epoch + 1,

            "train_loss":
                train_loss,

            "train_accuracy":
                train_accuracy,

            "val_loss":
                validation_loss,

            "val_accuracy":
                validation_accuracy,

            "epoch_time_minutes":
                epoch_time / 60
        }
    )


# ============================================================
# LOAD BEST MODEL
# ============================================================

model.load_state_dict(
    best_model_weights
)


# ============================================================
# SAVE TRAINING HISTORY
# ============================================================

history_path = os.path.join(
    OUTPUT_DIR,
    "training_history.json"
)

with open(
    history_path,
    "w"
) as f:

    json.dump(
        history,
        f,
        indent=4
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

total_time = (
    time.time() -
    start_time
)

print("\n========================================")
print("TRAINING COMPLETE")
print("========================================")

print(
    f"Best validation accuracy: "
    f"{best_accuracy:.4f}"
)

print(
    f"Total training time: "
    f"{total_time / 60:.2f} minutes"
)

print(
    "\nBest model:"
)

print(
    os.path.join(
        OUTPUT_DIR,
        "best_densenet121_cpu.pth"
    )
)

print(
    "\nTraining history:"
)

print(history_path)

print("\n========================================")