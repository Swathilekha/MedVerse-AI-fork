import os
import json
import random
import numpy as np
import pandas as pd
import joblib

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MANIFEST_PATH = (
    "data/clinical_text/fusion_experiment_dataset.csv"
)

IMAGE_FEATURES_PATH = (
    "data/clinical_text/early_fusion_image_features.npz"
)

OUTPUT_DIR = "ai-services/clinical/outputs"

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "early_fusion_best.pt"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "early_fusion_results.json"
)

VECTORIZER_PATH = (
    "data/clinical_text/early_fusion_train_tfidf_vectorizer.joblib"
)

BATCH_SIZE = 32
EPOCHS = 30
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4

IMAGE_FEATURE_DIM = 1024
TEXT_FEATURE_DIM = 5000

IMAGE_PROJECTION_DIM = 256
TEXT_PROJECTION_DIM = 256

PATIENCE = 6

DEVICE = torch.device(
    "cuda" if torch.cuda.is_available() else "cpu"
)


# ============================================================
# REPRODUCIBILITY
# ============================================================

def set_seed(seed=42):

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# ============================================================
# DATASET
# ============================================================

class EarlyFusionDataset(Dataset):

    def __init__(self, image_features, text_features, labels):

        self.image_features = torch.tensor(
            image_features,
            dtype=torch.float32
        )

        self.text_features = torch.tensor(
            text_features,
            dtype=torch.float32
        )

        self.labels = torch.tensor(
            labels,
            dtype=torch.float32
        )

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):

        return (
            self.image_features[idx],
            self.text_features[idx],
            self.labels[idx]
        )


# ============================================================
# EARLY FUSION MODEL
# ============================================================

class EarlyFusionMLP(nn.Module):

    def __init__(
        self,
        image_dim=1024,
        text_dim=5000
    ):

        super().__init__()

        # ----------------------------------------------------
        # Image branch
        # ----------------------------------------------------

        self.image_projection = nn.Sequential(

            nn.Linear(
                image_dim,
                IMAGE_PROJECTION_DIM
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                IMAGE_PROJECTION_DIM
            ),

            nn.Dropout(0.25)
        )

        # ----------------------------------------------------
        # Text branch
        # ----------------------------------------------------

        self.text_projection = nn.Sequential(

            nn.Linear(
                text_dim,
                TEXT_PROJECTION_DIM
            ),

            nn.ReLU(),

            nn.BatchNorm1d(
                TEXT_PROJECTION_DIM
            ),

            nn.Dropout(0.25)
        )

        # ----------------------------------------------------
        # Fusion layer
        # ----------------------------------------------------

        self.classifier = nn.Sequential(

            nn.Linear(
                IMAGE_PROJECTION_DIM +
                TEXT_PROJECTION_DIM,
                256
            ),

            nn.ReLU(),

            nn.Dropout(0.30),

            nn.Linear(
                256,
                64
            ),

            nn.ReLU(),

            nn.Dropout(0.20),

            nn.Linear(
                64,
                1
            )
        )

    def forward(
        self,
        image_features,
        text_features
    ):

        image_embedding = self.image_projection(
            image_features
        )

        text_embedding = self.text_projection(
            text_features
        )

        # Feature-level fusion
        fused = torch.cat(
            [
                image_embedding,
                text_embedding
            ],
            dim=1
        )

        output = self.classifier(fused)

        return output.squeeze(1)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(labels, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    metrics = {

        "accuracy": float(
            accuracy_score(
                labels,
                predictions
            )
        ),

        "precision": float(
            precision_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "recall": float(
            recall_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "f1_score": float(
            f1_score(
                labels,
                predictions,
                zero_division=0
            )
        ),

        "roc_auc": float(
            roc_auc_score(
                labels,
                probabilities
            )
        ),

        "confusion_matrix":
            confusion_matrix(
                labels,
                predictions
            ).tolist()
    }

    return metrics


# ============================================================
# EVALUATION
# ============================================================

def evaluate(
    model,
    loader,
    criterion
):

    model.eval()

    total_loss = 0.0

    all_labels = []
    all_probabilities = []

    with torch.no_grad():

        for (
            image_features,
            text_features,
            labels
        ) in loader:

            image_features = image_features.to(
                DEVICE
            )

            text_features = text_features.to(
                DEVICE
            )

            labels = labels.to(
                DEVICE
            )

            logits = model(
                image_features,
                text_features
            )

            loss = criterion(
                logits,
                labels
            )

            total_loss += (
                loss.item()
                * labels.size(0)
            )

            probabilities = torch.sigmoid(
                logits
            )

            all_labels.extend(
                labels.cpu().numpy()
            )

            all_probabilities.extend(
                probabilities.cpu().numpy()
            )

    all_labels = np.array(
        all_labels
    )

    all_probabilities = np.array(
        all_probabilities
    )

    average_loss = (
        total_loss /
        len(loader.dataset)
    )

    metrics = calculate_metrics(
        all_labels,
        all_probabilities
    )

    metrics["loss"] = float(
        average_loss
    )

    return metrics


# ============================================================
# MAIN
# ============================================================

def main():

    set_seed(SEED)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    print("=" * 70)
    print("EARLY FUSION — TRAINING")
    print("=" * 70)

    print()
    print("Device:", DEVICE)

    # --------------------------------------------------------
    # Load manifest
    # --------------------------------------------------------

    print()
    print("Loading fusion manifest...")

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        "Total records:",
        len(df)
    )

    # --------------------------------------------------------
    # Verify splits
    # --------------------------------------------------------

    train_df = df[
        df["final_split"] == "train"
    ].copy()

    val_df = df[
        df["final_split"] == "val"
    ].copy()

    print(
        "Training records:",
        len(train_df)
    )

    print(
        "Validation records:",
        len(val_df)
    )

    # --------------------------------------------------------
    # Load image features
    # --------------------------------------------------------

    print()
    print("Loading image features...")

    image_data = np.load(
        IMAGE_FEATURES_PATH
    )

    image_features = image_data[
        "image_features"
    ]

    print(
        "Image feature shape:",
        image_features.shape
    )

    if len(image_features) != len(df):

        raise ValueError(
            "Image feature count does not "
            "match manifest."
        )

    # --------------------------------------------------------
    # Fit TF-IDF ONLY on training text
    # --------------------------------------------------------

    print()
    print(
        "Fitting TF-IDF on TRAINING text only..."
    )

    vectorizer = TfidfVectorizer(

        max_features=TEXT_FEATURE_DIM,

        ngram_range=(1, 2),

        min_df=2,

        sublinear_tf=True
    )

    train_text = train_df[
        "clinical_text"
    ].fillna("").astype(str)

    val_text = val_df[
        "clinical_text"
    ].fillna("").astype(str)

    train_text_features = vectorizer.fit_transform(
        train_text
    )

    val_text_features = vectorizer.transform(
        val_text
    )

    actual_text_dim = (
        train_text_features.shape[1]
    )

    print(
        "Training TF-IDF shape:",
        train_text_features.shape
    )

    print(
        "Validation TF-IDF shape:",
        val_text_features.shape
    )

    print(
        "Vocabulary size:",
        actual_text_dim
    )

    # --------------------------------------------------------
    # Save leakage-free vectorizer
    # --------------------------------------------------------

    joblib.dump(
        vectorizer,
        VECTORIZER_PATH
    )

    print()
    print(
        "Training vectorizer saved to:"
    )

    print(
        VECTORIZER_PATH
    )

    # --------------------------------------------------------
    # IMPORTANT:
    # Align image features using manifest row positions
    # --------------------------------------------------------

    train_indices = train_df.index.to_numpy()
    val_indices = val_df.index.to_numpy()

    train_image_features = image_features[
        train_indices
    ]

    val_image_features = image_features[
        val_indices
    ]

    # --------------------------------------------------------
    # Convert sparse TF-IDF to dense
    # --------------------------------------------------------

    print()
    print(
        "Converting TF-IDF features to dense arrays..."
    )

    train_text_features = (
        train_text_features
        .astype(np.float32)
        .toarray()
    )

    val_text_features = (
        val_text_features
        .astype(np.float32)
        .toarray()
    )

    print(
        "Train text matrix:",
        train_text_features.shape
    )

    print(
        "Validation text matrix:",
        val_text_features.shape
    )

    # --------------------------------------------------------
    # Labels
    # --------------------------------------------------------

    train_labels = train_df[
        "label"
    ].values.astype(np.float32)

    val_labels = val_df[
        "label"
    ].values.astype(np.float32)

    print()
    print(
        "Train label distribution:",
        np.bincount(
            train_labels.astype(int)
        )
    )

    print(
        "Validation label distribution:",
        np.bincount(
            val_labels.astype(int)
        )
    )

    # --------------------------------------------------------
    # Create datasets
    # --------------------------------------------------------

    train_dataset = EarlyFusionDataset(

        train_image_features,

        train_text_features,

        train_labels
    )

    val_dataset = EarlyFusionDataset(

        val_image_features,

        val_text_features,

        val_labels
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
    # Model
    # --------------------------------------------------------

    print()
    print(
        "Creating Early Fusion model..."
    )

    model = EarlyFusionMLP(

        image_dim=IMAGE_FEATURE_DIM,

        text_dim=actual_text_dim
    ).to(DEVICE)

    print(model)

    # --------------------------------------------------------
    # Class imbalance handling
    # --------------------------------------------------------

    positive_count = (
        train_labels == 1
    ).sum()

    negative_count = (
        train_labels == 0
    ).sum()

    pos_weight_value = (
        negative_count /
        positive_count
    )

    pos_weight = torch.tensor(
        [pos_weight_value],
        dtype=torch.float32,
        device=DEVICE
    )

    print()
    print(
        "Positive samples:",
        int(positive_count)
    )

    print(
        "Negative samples:",
        int(negative_count)
    )

    print(
        "Positive class weight:",
        float(pos_weight_value)
    )

    criterion = nn.BCEWithLogitsLoss(
        pos_weight=pos_weight
    )

    optimizer = torch.optim.AdamW(

        model.parameters(),

        lr=LEARNING_RATE,

        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("STARTING TRAINING")
    print("=" * 70)

    best_auc = -1.0
    best_epoch = 0
    patience_counter = 0

    training_history = []

    for epoch in range(
        1,
        EPOCHS + 1
    ):

        model.train()

        running_loss = 0.0

        for (
            image_features_batch,
            text_features_batch,
            labels_batch
        ) in train_loader:

            image_features_batch = (
                image_features_batch.to(
                    DEVICE
                )
            )

            text_features_batch = (
                text_features_batch.to(
                    DEVICE
                )
            )

            labels_batch = (
                labels_batch.to(
                    DEVICE
                )
            )

            optimizer.zero_grad()

            logits = model(

                image_features_batch,

                text_features_batch
            )

            loss = criterion(
                logits,
                labels_batch
            )

            loss.backward()

            optimizer.step()

            running_loss += (
                loss.item()
                * labels_batch.size(0)
            )

        train_loss = (
            running_loss /
            len(train_loader.dataset)
        )

        # ----------------------------------------------------
        # Validation
        # ----------------------------------------------------

        val_metrics = evaluate(

            model,

            val_loader,

            criterion
        )

        print(
            f"Epoch {epoch:02d}/{EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Val Loss: {val_metrics['loss']:.4f} | "
            f"Val Accuracy: {val_metrics['accuracy']:.4f} | "
            f"Val Precision: {val_metrics['precision']:.4f} | "
            f"Val Recall: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1_score']:.4f} | "
            f"Val ROC-AUC: {val_metrics['roc_auc']:.4f}"
        )

        history_entry = {

            "epoch": epoch,

            "train_loss": float(
                train_loss
            ),

            **val_metrics
        }

        training_history.append(
            history_entry
        )

        # ----------------------------------------------------
        # Save best model based on ROC-AUC
        # ----------------------------------------------------

        current_auc = (
            val_metrics["roc_auc"]
        )

        if current_auc > best_auc:

            best_auc = current_auc

            best_epoch = epoch

            patience_counter = 0

            torch.save(

                {
                    "model_state_dict":
                        model.state_dict(),

                    "image_feature_dim":
                        IMAGE_FEATURE_DIM,

                    "text_feature_dim":
                        actual_text_dim,

                    "image_projection_dim":
                        IMAGE_PROJECTION_DIM,

                    "text_projection_dim":
                        TEXT_PROJECTION_DIM,

                    "best_val_roc_auc":
                        best_auc,

                    "epoch":
                        epoch
                },

                MODEL_PATH
            )

            print(
                "  ✓ New best model saved."
            )

        else:

            patience_counter += 1

        if patience_counter >= PATIENCE:

            print()
            print(
                "Early stopping triggered."
            )

            break

    # --------------------------------------------------------
    # Final best model evaluation
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("LOADING BEST EARLY FUSION MODEL")
    print("=" * 70)

    checkpoint = torch.load(
        MODEL_PATH,
        map_location=DEVICE
    )

    model.load_state_dict(
        checkpoint["model_state_dict"]
    )

    final_metrics = evaluate(

        model,

        val_loader,

        criterion
    )

    # --------------------------------------------------------
    # Print final results
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("EARLY FUSION — FINAL VALIDATION RESULTS")
    print("=" * 70)

    print(
        f"Accuracy : {final_metrics['accuracy']:.4f}"
    )

    print(
        f"Precision: {final_metrics['precision']:.4f}"
    )

    print(
        f"Recall   : {final_metrics['recall']:.4f}"
    )

    print(
        f"F1 Score : {final_metrics['f1_score']:.4f}"
    )

    print(
        f"ROC-AUC  : {final_metrics['roc_auc']:.4f}"
    )

    print()
    print(
        "Confusion Matrix:"
    )

    print(
        np.array(
            final_metrics["confusion_matrix"]
        )
    )

    # --------------------------------------------------------
    # Save results
    # --------------------------------------------------------

    results = {

        "model": "Early Fusion",

        "fusion_type":
            "Feature-level / Early Fusion",

        "device":
            str(DEVICE),

        "total_records":
            int(len(df)),

        "train_records":
            int(len(train_df)),

        "validation_records":
            int(len(val_df)),

        "image_feature_dimension":
            IMAGE_FEATURE_DIM,

        "text_feature_dimension":
            actual_text_dim,

        "best_epoch":
            best_epoch,

        "best_val_roc_auc":
            float(best_auc),

        "final_metrics":
            final_metrics,

        "training_history":
            training_history
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

    print()
    print(
        "Model saved to:"
    )

    print(
        MODEL_PATH
    )

    print()
    print(
        "Results saved to:"
    )

    print(
        RESULTS_PATH
    )

    print()
    print("=" * 70)
    print("EARLY FUSION TRAINING COMPLETE")
    print("=" * 70)


if __name__ == "__main__":

    main()