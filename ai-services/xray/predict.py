import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from PIL import Image
from torchvision import transforms, models
from torchvision.models import DenseNet121_Weights
from torch import nn


# --------------------------------------------------
# Configuration
# --------------------------------------------------

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.dirname(
            os.path.abspath(__file__)
        )
    )
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "ai-services",
    "xray",
    "outputs",
    "best_densenet121_finetuned_cpu.pth"
)

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
# Load model
# --------------------------------------------------

def load_model():

    print("Loading DenseNet121...")

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

    return model


# --------------------------------------------------
# Prediction
# --------------------------------------------------

def predict(image_path):

    if not os.path.exists(image_path):
        raise FileNotFoundError(
            f"Image not found: {image_path}"
        )

    model = load_model()

    image = Image.open(image_path).convert("L")

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    with torch.no_grad():

        outputs = model(input_tensor)

        probabilities = torch.softmax(
            outputs,
            dim=1
        )

        confidence, predicted = torch.max(
            probabilities,
            dim=1
        )

    predicted_class = CLASS_NAMES[
        predicted.item()
    ]

    confidence_value = confidence.item()

    print("\n========================================")
    print("MEDVERSE AI - XRAY PREDICTION")
    print("========================================")

    print("Image:", image_path)

    print(
        "Prediction:",
        predicted_class
    )

    print(
        "Confidence:",
        f"{confidence_value * 100:.2f}%"
    )

    print("\nClass probabilities:")

    for class_name, probability in zip(
        CLASS_NAMES,
        probabilities[0]
    ):

        print(
            f"{class_name}: "
            f"{probability.item() * 100:.2f}%"
        )

    print("========================================")

    return {
        "prediction": predicted_class,
        "confidence": confidence_value,
        "probabilities": {
            CLASS_NAMES[0]:
                probabilities[0][0].item(),

            CLASS_NAMES[1]:
                probabilities[0][1].item()
        }
    }

def predict_manifest(manifest_path, output_path):

    print("\n========================================")
    print("MEDVERSE AI - BATCH XRAY PREDICTION")
    print("========================================")

    print("Manifest:", manifest_path)
    print("Output:", output_path)

    df = pd.read_csv(manifest_path)

    print("Images:", len(df))

    model = load_model()

    results = []

    for _, row in tqdm(
        df.iterrows(),
        total=len(df),
        desc="Predicting"
    ):

        image_path = row["local_path"]

        if not os.path.exists(image_path):
            print(
                f"\nWARNING: Image not found: {image_path}"
            )
            continue

        try:

            image = Image.open(
                image_path
            ).convert("L")

            input_tensor = transform(image)
            input_tensor = input_tensor.unsqueeze(0)
            input_tensor = input_tensor.to(device)

            with torch.no_grad():

                outputs = model(input_tensor)

                probabilities = torch.softmax(
                    outputs,
                    dim=1
                )

            normal_probability = (
                probabilities[0][0].item()
            )

            pneumonia_probability = (
                probabilities[0][1].item()
            )

            confidence = max(
                normal_probability,
                pneumonia_probability
            )

            predicted_class = (
                "PNEUMONIA"
                if pneumonia_probability >= normal_probability
                else "NORMAL"
            )

            results.append({

                "patient_id":
                    row["patient_id"],

                "png_path":
                    row["png_path"],

                "local_path":
                    image_path,

                "true_class":
                    row["class_name"],

                "prediction":
                    predicted_class,

                "normal_probability":
                    normal_probability,

                "pneumonia_probability":
                    pneumonia_probability,

                "xray_confidence":
                    confidence
            })

        except Exception as e:

            print(
                f"\nERROR processing {image_path}: {e}"
            )

    results_df = pd.DataFrame(results)

    results_df.to_csv(
        output_path,
        index=False
    )

    print("\n========================================")
    print("BATCH PREDICTION COMPLETE")
    print("========================================")

    print(
        "Predictions generated:",
        len(results_df)
    )

    print(
        "Saved:",
        output_path
    )

    print("========================================")
# --------------------------------------------------
# Main
# --------------------------------------------------

if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--image",
        required=False,
        help="Path to a single chest X-ray image"
    )

    parser.add_argument(
        "--manifest",
        required=False,
        help="Path to X-ray manifest CSV"
    )

    parser.add_argument(
        "--output",
        default="data/clinical_text/xray_predictions.csv",
        help="Output CSV for batch predictions"
    )

    args = parser.parse_args()

    if args.manifest:

        predict_manifest(
            args.manifest,
            args.output
        )

    elif args.image:

        predict(args.image)

    else:

        parser.error(
            "Provide either --image or --manifest"
        )