import os
import random
import numpy as np
import torch

from torch import nn, optim
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms, models
from torchvision.models import DenseNet121_Weights
from sklearn.utils.class_weight import compute_class_weight


# ==========================================
# PATHS
# ==========================================

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
    "chest_xray",
    "chest_xray"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ai-services",
    "xray",
    "outputs"
)

os.makedirs(OUTPUT_DIR, exist_ok=True)


# ==========================================
# SETTINGS
# ==========================================

SEED = 42

random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)

IMAGE_SIZE = 160
BATCH_SIZE = 16

# Preliminary CPU experiment
SAMPLES_PER_CLASS = 250

EPOCHS = 2
LEARNING_RATE = 0.0001


# ==========================================
# DEVICE
# ==========================================

device = torch.device("cpu")

print("\n========================================")
print("MEDVERSE AI - QUICK XRAY TRAINING")
print("========================================")
print("Device:", device)
print("Image size:", IMAGE_SIZE)
print("Samples per class:", SAMPLES_PER_CLASS)
print("Epochs:", EPOCHS)
print("========================================\n")


# ==========================================
# TRANSFORMS
# ==========================================

train_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.RandomHorizontalFlip(),
    transforms.RandomRotation(5),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


val_transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ==========================================
# LOAD DATASET
# ==========================================

train_dir = os.path.join(
    DATA_DIR,
    "train"
)

val_dir = os.path.join(
    DATA_DIR,
    "val"
)

train_dataset_full = datasets.ImageFolder(
    train_dir,
    transform=train_transform
)

val_dataset = datasets.ImageFolder(
    val_dir,
    transform=val_transform
)

print("Classes:", train_dataset_full.classes)
print("Full training images:", len(train_dataset_full))
print("Validation images:", len(val_dataset))


# ==========================================
# SELECT BALANCED SUBSET
# ==========================================

targets = np.array(
    train_dataset_full.targets
)

selected_indices = []

for class_id in range(2):

    class_indices = np.where(
        targets == class_id
    )[0]

    np.random.shuffle(
        class_indices
    )

    selected = class_indices[
        :SAMPLES_PER_CLASS
    ]

    selected_indices.extend(
        selected.tolist()
    )

random.shuffle(
    selected_indices
)

train_dataset = Subset(
    train_dataset_full,
    selected_indices
)

print(
    "Quick training images:",
    len(train_dataset)
)

print(
    "Balanced subset:",
    SAMPLES_PER_CLASS,
    "NORMAL +",
    SAMPLES_PER_CLASS,
    "PNEUMONIA"
)


# ==========================================
# DATALOADERS
# ==========================================

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


# ==========================================
# CLASS WEIGHTS
# ==========================================

subset_targets = targets[
    selected_indices
]

class_weights = compute_class_weight(
    class_weight="balanced",
    classes=np.array([0, 1]),
    y=subset_targets
)

class_weights = torch.tensor(
    class_weights,
    dtype=torch.float32
)

print(
    "Class weights:",
    class_weights
)


# ==========================================
# DENSENET121
# ==========================================

print("\nLoading DenseNet121...")

model = models.densenet121(
    weights=DenseNet121_Weights.DEFAULT
)

# Freeze pretrained feature extractor
for parameter in model.features.parameters():
    parameter.requires_grad = False


input_features = model.classifier.in_features

model.classifier = nn.Linear(
    input_features,
    2
)

model = model.to(device)


# ==========================================
# LOSS + OPTIMIZER
# ==========================================

criterion = nn.CrossEntropyLoss(
    weight=class_weights
)

optimizer = optim.Adam(
    model.classifier.parameters(),
    lr=LEARNING_RATE
)


# ==========================================
# TRAINING
# ==========================================

best_val_loss = float("inf")

print("\nStarting training...\n")

for epoch in range(EPOCHS):

    model.train()

    running_loss = 0.0
    correct = 0
    total = 0

    for batch_number, (images, labels) in enumerate(
        train_loader,
        start=1
    ):

        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)

        loss = criterion(
            outputs,
            labels
        )

        loss.backward()

        optimizer.step()

        running_loss += (
            loss.item() *
            images.size(0)
        )

        predictions = torch.argmax(
            outputs,
            dim=1
        )

        total += labels.size(0)

        correct += (
            predictions == labels
        ).sum().item()

        if batch_number % 5 == 0:

            print(
                f"Epoch {epoch + 1}/{EPOCHS} "
                f"| Batch {batch_number}/{len(train_loader)}"
            )

    train_loss = running_loss / total
    train_accuracy = correct / total


    # ======================================
    # VALIDATION
    # ======================================

    model.eval()

    val_loss_total = 0.0
    val_correct = 0
    val_total = 0

    with torch.no_grad():

        for images, labels in val_loader:

            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)

            loss = criterion(
                outputs,
                labels
            )

            val_loss_total += (
                loss.item() *
                images.size(0)
            )

            predictions = torch.argmax(
                outputs,
                dim=1
            )

            val_total += labels.size(0)

            val_correct += (
                predictions == labels
            ).sum().item()

    val_loss = val_loss_total / val_total
    val_accuracy = val_correct / val_total


    print("\n----------------------------------------")
    print(
        f"Epoch {epoch + 1}/{EPOCHS}"
    )
    print(
        f"Train Loss: {train_loss:.4f}"
    )
    print(
        f"Train Accuracy: {train_accuracy:.4f}"
    )
    print(
        f"Validation Loss: {val_loss:.4f}"
    )
    print(
        f"Validation Accuracy: {val_accuracy:.4f}"
    )
    print("----------------------------------------\n")


    # ======================================
    # SAVE BEST MODEL
    # ======================================

    if val_loss < best_val_loss:

        best_val_loss = val_loss

        model_path = os.path.join(
            OUTPUT_DIR,
            "best_densenet121.pth"
        )

        torch.save(
            {
                "model_state_dict":
                    model.state_dict(),

                "classes":
                    train_dataset_full.classes,

                "image_size":
                    IMAGE_SIZE
            },
            model_path
        )

        print(
            "Best model saved to:"
        )

        print(model_path)


print("\n========================================")
print("QUICK TRAINING COMPLETE")
print("========================================")