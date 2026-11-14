import os
import sys
import torch
from PIL import Image
from torchvision import transforms, models
from torch import nn


# --------------------------------------------------
# Project paths
# --------------------------------------------------

PROJECT_ROOT = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        ".."
    )
)

MODEL_PATH = os.path.join(
    PROJECT_ROOT,
    "ai-services",
    "xray",
    "outputs",
    "best_densenet121_finetuned_cpu.pth"
)


# --------------------------------------------------
# Configuration
# --------------------------------------------------

IMAGE_SIZE = 224

CLASS_NAMES = [
    "NORMAL",
    "PNEUMONIA"
]


# --------------------------------------------------
# Device
# --------------------------------------------------

device = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# --------------------------------------------------
# Preprocessing
# --------------------------------------------------

transform = transforms.Compose([
    transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
    transforms.Grayscale(num_output_channels=3),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225]
    )
])


# --------------------------------------------------
# Model
# --------------------------------------------------

_model = None


def load_model():

    global _model

    if _model is not None:
        return _model

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"DenseNet checkpoint not found: {MODEL_PATH}"
        )

    print("Loading MedVerse DenseNet121...")

    model = models.densenet121(weights=None)

    input_features = model.classifier.in_features

    model.classifier = nn.Linear(
        input_features,
        2
    )

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=device
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    model = model.to(device)

    model.eval()

    _model = model

    print(
        f"DenseNet121 loaded successfully on {device}"
    )

    return _model


# --------------------------------------------------
# Single X-Ray Prediction
# --------------------------------------------------

def predict_xray(image_path: str):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"X-ray image not found: {image_path}"
        )

    model = load_model()

    image = Image.open(
        image_path
    ).convert("L")

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():

        outputs = model(
            input_tensor
        )

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    normal_probability = (
        probabilities[0][0].item()
    )

    pneumonia_probability = (
        probabilities[0][1].item()
    )

    predicted_class = CLASS_NAMES[
        predicted.item()
    ]

    return {
        "prediction": predicted_class,
        "confidence": confidence.item(),

        "normal_probability":
            normal_probability,

        "pneumonia_probability":
            pneumonia_probability,

        "model": "DenseNet121",

        "checkpoint":
            os.path.basename(MODEL_PATH),

        "device":
            str(device)
    }