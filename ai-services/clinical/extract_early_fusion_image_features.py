import os
from pathlib import Path

import pandas as pd
import numpy as np
from PIL import Image

import torch
import torch.nn as nn
from torchvision import models, transforms


# ============================================================
# PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

MANIFEST = (
    PROJECT_ROOT
    / "data"
    / "clinical_text"
    / "fusion_experiment_dataset.csv"
)

CHECKPOINT = (
    PROJECT_ROOT
    / "ai-services"
    / "xray"
    / "outputs"
    / "best_densenet121_finetuned_cpu.pth"
)

OUTPUT = (
    PROJECT_ROOT
    / "data"
    / "clinical_text"
    / "early_fusion_image_features.npz"
)


# ============================================================
# DEVICE
# ============================================================

device = torch.device("cpu")

print("=" * 70)
print("EARLY FUSION — IMAGE FEATURE EXTRACTION")
print("=" * 70)

print(f"\nDevice: {device}")
print(f"Manifest: {MANIFEST}")
print(f"Checkpoint: {CHECKPOINT}")


# ============================================================
# LOAD MANIFEST
# ============================================================

df = pd.read_csv(MANIFEST)

print(f"\nTotal records: {len(df)}")


# ============================================================
# IMAGE TRANSFORM
# ============================================================

transform = transforms.Compose([
    transforms.Resize((160, 160)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# ============================================================
# LOAD DENSENET121
# ============================================================

print("\nLoading DenseNet121...")

model = models.densenet121(weights=None)

# Original DenseNet classifier
model.classifier = nn.Linear(
    model.classifier.in_features,
    2
)

checkpoint = torch.load(
    CHECKPOINT,
    map_location=device
)

# Handle checkpoints that may contain a state_dict
if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
    state_dict = checkpoint["model_state_dict"]
elif isinstance(checkpoint, dict) and "state_dict" in checkpoint:
    state_dict = checkpoint["state_dict"]
else:
    state_dict = checkpoint

# Remove possible "module." prefix
state_dict = {
    k.replace("module.", "", 1): v
    for k, v in state_dict.items()
}

model.load_state_dict(
    state_dict,
    strict=True
)

model = model.to(device)
model.eval()

print("DenseNet121 loaded successfully.")


# ============================================================
# CREATE FEATURE EXTRACTOR
# ============================================================

# DenseNet121 produces a 1024-dimensional feature vector
feature_extractor = model.features


def extract_feature(image_tensor):
    """
    Extract 1024-dimensional DenseNet feature.
    """

    with torch.no_grad():

        features = feature_extractor(
            image_tensor
        )

        features = torch.relu(features)

        # Global average pooling
        features = torch.nn.functional.adaptive_avg_pool2d(
            features,
            (1, 1)
        )

        features = torch.flatten(
            features,
            start_dim=1
        )

    return features


# ============================================================
# PROCESS IMAGES
# ============================================================

features = []
patient_ids = []
labels = []
splits = []
image_paths = []

failed = []

print("\nExtracting features...")
print("This will run on CPU and may take some time.\n")


for idx, row in df.iterrows():

    image_path = Path(row["image_path"])

    try:

        if not image_path.exists():
            raise FileNotFoundError(
                f"Image not found: {image_path}"
            )

        image = Image.open(
            image_path
        ).convert("L")

        image_tensor = transform(
            image
        ).unsqueeze(0).to(device)

        feature = extract_feature(
            image_tensor
        )

        features.append(
            feature.squeeze(0).numpy()
        )

        patient_ids.append(
            row["patient_id"]
        )

        labels.append(
            int(row["label"])
        )

        splits.append(
            row["final_split"]
        )

        image_paths.append(
            str(image_path)
        )

    except Exception as e:

        failed.append({
            "index": idx,
            "patient_id": row["patient_id"],
            "image_path": str(image_path),
            "error": str(e)
        })

    if (idx + 1) % 100 == 0:

        print(
            f"Processed {idx + 1}/{len(df)}"
        )


# ============================================================
# CONVERT TO ARRAYS
# ============================================================

features = np.asarray(
    features,
    dtype=np.float32
)

labels = np.asarray(
    labels,
    dtype=np.int64
)

patient_ids = np.asarray(
    patient_ids
)

splits = np.asarray(
    splits
)

image_paths = np.asarray(
    image_paths
)


# ============================================================
# SAVE
# ============================================================

np.savez_compressed(
    OUTPUT,
    image_features=features,
    patient_ids=patient_ids,
    labels=labels,
    splits=splits,
    image_paths=image_paths
)


# ============================================================
# REPORT
# ============================================================

print("\n" + "=" * 70)
print("FEATURE EXTRACTION COMPLETE")
print("=" * 70)

print(
    f"\nFeature matrix shape: {features.shape}"
)

print(
    f"Expected feature dimension: 1024"
)

print(
    f"Successfully processed: {len(features)}"
)

print(
    f"Failed: {len(failed)}"
)

if failed:

    failed_df = pd.DataFrame(failed)

    failed_file = (
        OUTPUT.parent
        / "early_fusion_failed_images.csv"
    )

    failed_df.to_csv(
        failed_file,
        index=False
    )

    print(
        f"\nFailed-image list saved to:"
        f"\n{failed_file}"
    )

print(
    f"\nSaved features to:"
    f"\n{OUTPUT}"
)

print("\nDone.")