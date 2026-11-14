import os
import argparse

from matplotlib import axes, image
from matplotlib import image
import torch
import numpy as np

from PIL import Image
import matplotlib.pyplot as plt

from torchvision import transforms, models
from torch import gradient, nn


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

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "ai-services",
    "xray",
    "outputs",
    "gradcam"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

IMAGE_SIZE = 160

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

    model = models.densenet121(
        weights=None
    )

    input_features = (
        model.classifier.in_features
    )

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
# Grad-CAM
# --------------------------------------------------

def generate_gradcam(image_path):

    model = load_model()

    image = Image.open(
        image_path
    ).convert("L")

    input_tensor = transform(image)

    input_tensor = input_tensor.unsqueeze(0)

    input_tensor = input_tensor.to(device)

    # DenseNet final convolutional layer
    target_layer = model.features.denseblock3

    activations = []
    gradients = []

    def forward_hook(module, input, output):

        activations.append(
            output
        )

    def backward_hook(module, grad_input, grad_output):

        gradients.append(
            grad_output[0]
        )

    target_layer.register_forward_hook(
        forward_hook
    )

    target_layer.register_full_backward_hook(
        backward_hook
    )

    # Forward pass

    output = model(
        input_tensor
    )

    probabilities = torch.softmax(
        output,
        dim=1
    )

    predicted_class = torch.argmax(
        probabilities,
        dim=1
    )

    # Backward pass

    model.zero_grad()

    target = output[
        0,
        predicted_class.item()
    ]

    target.backward()

    import torch.nn.functional as F

# Get activations and gradients
    activation = activations[0]
    gradient = gradients[0]

# Calculate channel-wise weights
    weights = gradient.mean(dim=(2, 3), keepdim=True)

# Generate Grad-CAM
    cam = (weights * activation).sum(dim=1, keepdim=True)
    cam = torch.relu(cam)

# Upsample to original image dimensions
    cam = F.interpolate(
        cam,
        size=(IMAGE_SIZE, IMAGE_SIZE),
        mode="bilinear",
        align_corners=False
    )

    cam = cam.squeeze().detach().cpu().numpy()

# Normalize
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)

# Original grayscale image
    original = np.array(
        image.resize((IMAGE_SIZE, IMAGE_SIZE))
    )

# Create publication-style visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Original image
    axes[0].imshow(original, cmap="gray")
    axes[0].set_title("Original X-ray")
    axes[0].axis("off")

# Heatmap
    axes[1].imshow(cam, cmap="jet", vmin=0, vmax=1)
    axes[1].set_title("Grad-CAM Heatmap")
    axes[1].axis("off")

# Overlay
    axes[2].imshow(original, cmap="gray")
    axes[2].imshow(
        cam,
        cmap="jet",
        alpha=0.35,
        vmin=0,
        vmax=1,
        interpolation="bilinear"
    )
    axes[2].set_title("Grad-CAM Overlay")
    axes[2].axis("off")

    fig.suptitle(
        f"Prediction: {CLASS_NAMES[predicted_class.item()]} | "
        f"Confidence: {probabilities[0][predicted_class].item() * 100:.2f}%"
    )

    plt.tight_layout()

    # Generate a unique filename using the patient ID
    patient_id = os.path.basename(
        os.path.dirname(os.path.dirname(image_path))
)

    image_name = os.path.splitext(
        os.path.basename(image_path)
    )[0]

    output_filename = f"{patient_id}_{image_name}_gradcam.png"

    output_path = os.path.join(
        OUTPUT_DIR,
        output_filename
    )

    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close()

    print("\n========================================")
    print("GRAD-CAM COMPLETE")
    print("========================================")

    print(
        "Prediction:",
        CLASS_NAMES[
            predicted_class.item()
        ]
    )

    print(
        "Confidence:",
        f"{probabilities[0][predicted_class].item() * 100:.2f}%"
    )

    print(
        "Grad-CAM saved:",
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
        required=True
    )

    args = parser.parse_args()

    generate_gradcam(
        args.image
    )