import os
import json
import random
import re

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)


# ============================================================
# CONFIGURATION
# ============================================================

SEED = 42

MANIFEST_PATH = (
    "data/clinical_text/fusion_experiment_dataset.csv"
)

VOCAB_PATH = (
    "data/clinical_text/joint_fusion_vocab.json"
)

CHECKPOINT_PATH = (
    "ai-services/xray/outputs/"
    "best_densenet121_finetuned_cpu.pth"
)

OUTPUT_DIR = (
    "ai-services/clinical/outputs"
)

MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "joint_fusion_best.pt"
)

RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "joint_fusion_results.json"
)


# ============================================================
# TRAINING
# ============================================================

BATCH_SIZE = 8

EPOCHS = 20

LEARNING_RATE = 1e-4

DENSENET_LEARNING_RATE = 1e-5

WEIGHT_DECAY = 1e-4

PATIENCE = 5


# ============================================================
# TEXT
# ============================================================

MAX_SEQUENCE_LENGTH = 256

EMBEDDING_DIM = 256

LSTM_HIDDEN_DIM = 128

TEXT_FEATURE_DIM = 256


# ============================================================
# IMAGE
# ============================================================

IMAGE_SIZE = 160

IMAGE_FEATURE_DIM = 1024

IMAGE_TOKEN_DIM = 256


# ============================================================
# CROSS-MODAL ATTENTION
# ============================================================

NUM_HEADS = 4

TRANSFORMER_DROPOUT = 0.15


# ============================================================
# DEVICE
# ============================================================

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "cpu"
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

    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ============================================================
# IMAGE TRANSFORM
# ============================================================

IMAGE_TRANSFORM = transforms.Compose([

    transforms.Grayscale(
        num_output_channels=3
    ),

    transforms.Resize(
        (IMAGE_SIZE, IMAGE_SIZE)
    ),

    transforms.ToTensor(),

    transforms.Normalize(

        mean=[
            0.485,
            0.456,
            0.406
        ],

        std=[
            0.229,
            0.224,
            0.225
        ]
    )
])


# ============================================================
# DATASET
# ============================================================

class JointFusionDataset(Dataset):

    def __init__(
        self,
        dataframe,
        vocab
    ):

        self.df = dataframe.reset_index(
            drop=True
        )

        self.vocab = vocab

        self.pad_idx = vocab["<PAD>"]

        self.unk_idx = vocab["<UNK>"]


    @staticmethod
    def tokenize(text):

        text = str(text).lower()

        tokens = re.findall(

            r"[a-z]+(?:[-'][a-z]+)*|\d+(?:\.\d+)?",

            text
        )

        return tokens


    def encode_text(self, text):

        tokens = self.tokenize(text)

        ids = [

            self.vocab.get(
                token,
                self.unk_idx
            )

            for token in tokens
        ]

        ids = ids[
            :MAX_SEQUENCE_LENGTH
        ]

        attention_mask = [
            1
        ] * len(ids)

        padding_length = (
            MAX_SEQUENCE_LENGTH
            - len(ids)
        )

        if padding_length > 0:

            ids += [
                self.pad_idx
            ] * padding_length

            attention_mask += [
                0
            ] * padding_length

        return (

            torch.tensor(
                ids,
                dtype=torch.long
            ),

            torch.tensor(
                attention_mask,
                dtype=torch.bool
            )
        )


    def __len__(self):

        return len(self.df)


    def __getitem__(self, idx):

        row = self.df.iloc[idx]

        # ----------------------------------------------------
        # IMAGE
        # ----------------------------------------------------

        image_path = str(
            row["image_path"]
        )

        image = Image.open(
            image_path
        ).convert("L")

        image = IMAGE_TRANSFORM(
            image
        )


        # ----------------------------------------------------
        # CLINICAL TEXT
        # ----------------------------------------------------

        token_ids, attention_mask = (
            self.encode_text(
                row["clinical_text"]
            )
        )


        # ----------------------------------------------------
        # LABEL
        # ----------------------------------------------------

        label = torch.tensor(
            float(row["label"]),
            dtype=torch.float32
        )


        return {

            "image": image,

            "input_ids": token_ids,

            "attention_mask":
                attention_mask,

            "label": label,

            "patient_id":
                str(row["patient_id"])
        }


# ============================================================
# DENSENET IMAGE ENCODER
# ============================================================

class DenseNetImageEncoder(nn.Module):

    """
    DenseNet121 image encoder.

    IMPORTANT:
    We retain the spatial feature map.

    Therefore the architecture is:

        X-ray
          ↓
        DenseNet121
          ↓
        [B, 1024, H, W]
          ↓
        [B, H*W, 1024]

    The spatial tokens are then used by cross-modal attention.
    """

    def __init__(
        self,
        checkpoint_path
    ):

        super().__init__()


        # ----------------------------------------------------
        # DenseNet121
        # ----------------------------------------------------

        base = models.densenet121(

            weights=None,

            # Matches the supplied checkpoint.
            num_classes=2
        )

        self.features = base.features

        self.num_features = 1024


        # ----------------------------------------------------
        # Load validated checkpoint
        # ----------------------------------------------------

        self.load_checkpoint(
            checkpoint_path
        )


    def load_checkpoint(
        self,
        checkpoint_path
    ):

        print()

        print(
            "Loading DenseNet121 checkpoint..."
        )

        print(
            "Checkpoint:",
            checkpoint_path
        )


        checkpoint = torch.load(

            checkpoint_path,

            map_location="cpu"
        )


        # ----------------------------------------------------
        # Extract state dictionary
        # ----------------------------------------------------

        if (

            isinstance(
                checkpoint,
                dict
            )

            and
            "model_state_dict"
            in checkpoint

        ):

            state_dict = checkpoint[
                "model_state_dict"
            ]


        elif (

            isinstance(
                checkpoint,
                dict
            )

            and
            "state_dict"
            in checkpoint

        ):

            state_dict = checkpoint[
                "state_dict"
            ]


        else:

            state_dict = checkpoint


        # ----------------------------------------------------
        # Convert:
        #
        # features.conv0.weight
        #
        # into:
        #
        # conv0.weight
        #
        # because self.features expects the latter.
        # ----------------------------------------------------

        cleaned_state_dict = {}


        for key, value in state_dict.items():

            new_key = key


            if new_key.startswith(
                "module."
            ):

                new_key = new_key[
                    len("module.") :
                ]


            if new_key.startswith(
                "features."
            ):

                new_key = new_key[
                    len("features.") :
                ]

                cleaned_state_dict[
                    new_key
                ] = value


        if not cleaned_state_dict:

            raise RuntimeError(

                "No DenseNet feature weights "
                "were found in the checkpoint."
            )


        # ----------------------------------------------------
        # STRICT LOAD
        #
        # This is deliberate.
        #
        # We do NOT want strict=False hiding
        # incorrect checkpoint mapping.
        # ----------------------------------------------------

        result = self.features.load_state_dict(

            cleaned_state_dict,

            strict=True
        )


        print(
            "DenseNet feature checkpoint "
            "loaded successfully."
        )

        print(
            result
        )

        print(
            "Loaded feature keys:",
            len(cleaned_state_dict)
        )


    def forward(self, x):

        x = self.features(x)

        x = F.relu(
            x,
            inplace=False
        )


        # ----------------------------------------------------
        # KEEP SPATIAL INFORMATION
        #
        # [B, 1024, H, W]
        #
        # becomes
        #
        # [B, H*W, 1024]
        # ----------------------------------------------------

        x = x.flatten(
            2
        )

        x = x.transpose(
            1,
            2
        )


        return x


# ============================================================
# CLINICAL TEXT ENCODER
# ============================================================

class ClinicalTextEncoder(nn.Module):

    def __init__(
        self,
        vocab_size,
        pad_idx
    ):

        super().__init__()


        self.embedding = nn.Embedding(

            num_embeddings=vocab_size,

            embedding_dim=EMBEDDING_DIM,

            padding_idx=pad_idx
        )


        self.lstm = nn.LSTM(

            input_size=EMBEDDING_DIM,

            hidden_size=LSTM_HIDDEN_DIM,

            num_layers=2,

            batch_first=True,

            bidirectional=True,

            dropout=0.15
        )


        self.projection = nn.Sequential(

            nn.Linear(

                LSTM_HIDDEN_DIM * 2,

                TEXT_FEATURE_DIM
            ),

            nn.LayerNorm(
                TEXT_FEATURE_DIM
            ),

            nn.GELU(),

            nn.Dropout(0.15)
        )


    def forward(
        self,
        input_ids,
        attention_mask
    ):

        embeddings = self.embedding(
            input_ids
        )


        lstm_output, _ = self.lstm(
            embeddings
        )


        token_features = self.projection(
            lstm_output
        )


        # [B, sequence_length, 256]

        return token_features


# ============================================================
# JOINT FUSION MODEL
# ============================================================

class JointFusionModel(nn.Module):

    """
    TRUE JOINT MULTIMODAL FUSION

    Image:
        DenseNet121
        ↓
        spatial image tokens

    Text:
        Embedding
        ↓
        BiLSTM
        ↓
        text tokens

    Fusion:
        image queries text
        text queries image

    Then:
        residual connections
        LayerNorm
        FFN

    Finally:
        pooled image
        pooled text
        image * text
        |image - text|

        ↓

        classifier
    """

    def __init__(
        self,
        vocab_size,
        pad_idx,
        checkpoint_path
    ):

        super().__init__()


        # ====================================================
        # IMAGE ENCODER
        # ====================================================

        self.image_encoder = (
            DenseNetImageEncoder(
                checkpoint_path
            )
        )


        self.image_projection = nn.Sequential(

            nn.Linear(

                IMAGE_FEATURE_DIM,

                IMAGE_TOKEN_DIM
            ),

            nn.LayerNorm(
                IMAGE_TOKEN_DIM
            ),

            nn.GELU()
        )


        # ====================================================
        # TEXT ENCODER
        # ====================================================

        self.text_encoder = (
            ClinicalTextEncoder(

                vocab_size,

                pad_idx
            )
        )


        # ====================================================
        # IMAGE → TEXT ATTENTION
        # ====================================================

        self.image_to_text_attention = (
            nn.MultiheadAttention(

                embed_dim=IMAGE_TOKEN_DIM,

                num_heads=NUM_HEADS,

                dropout=TRANSFORMER_DROPOUT,

                batch_first=True
            )
        )


        # ====================================================
        # TEXT → IMAGE ATTENTION
        # ====================================================

        self.text_to_image_attention = (
            nn.MultiheadAttention(

                embed_dim=IMAGE_TOKEN_DIM,

                num_heads=NUM_HEADS,

                dropout=TRANSFORMER_DROPOUT,

                batch_first=True
            )
        )


        # ====================================================
        # SEPARATE RESIDUAL NORMALIZATION
        # ====================================================

        self.image_attn_norm = nn.LayerNorm(
            IMAGE_TOKEN_DIM
        )

        self.image_ffn_norm = nn.LayerNorm(
            IMAGE_TOKEN_DIM
        )

        self.text_attn_norm = nn.LayerNorm(
            IMAGE_TOKEN_DIM
        )

        self.text_ffn_norm = nn.LayerNorm(
            IMAGE_TOKEN_DIM
        )


        # ====================================================
        # IMAGE FFN
        # ====================================================

        self.image_ffn = nn.Sequential(

            nn.Linear(

                IMAGE_TOKEN_DIM,

                IMAGE_TOKEN_DIM * 2
            ),

            nn.GELU(),

            nn.Dropout(
                TRANSFORMER_DROPOUT
            ),

            nn.Linear(

                IMAGE_TOKEN_DIM * 2,

                IMAGE_TOKEN_DIM
            )
        )


        # ====================================================
        # TEXT FFN
        # ====================================================

        self.text_ffn = nn.Sequential(

            nn.Linear(

                IMAGE_TOKEN_DIM,

                IMAGE_TOKEN_DIM * 2
            ),

            nn.GELU(),

            nn.Dropout(
                TRANSFORMER_DROPOUT
            ),

            nn.Linear(

                IMAGE_TOKEN_DIM * 2,

                IMAGE_TOKEN_DIM
            )
        )


        # ====================================================
        # JOINT CLASSIFIER
        # ====================================================

        # 256 image
        # + 256 text
        # + 256 product
        # + 256 difference
        #
        # = 1024
        #

        joint_dimension = (
            IMAGE_TOKEN_DIM * 4
        )


        self.classifier = nn.Sequential(

            nn.Linear(
                joint_dimension,
                512
            ),

            nn.LayerNorm(
                512
            ),

            nn.GELU(),

            nn.Dropout(
                0.30
            ),

            nn.Linear(
                512,
                128
            ),

            nn.GELU(),

            nn.Dropout(
                0.20
            ),

            nn.Linear(
                128,
                1
            )
        )


    # ========================================================
    # MASKED MEAN
    # ========================================================

    @staticmethod
    def masked_mean(
        x,
        mask
    ):

        # x:
        # [B, sequence_length, D]

        # mask:
        # [B, sequence_length]

        mask_float = (
            mask
            .unsqueeze(-1)
            .float()
        )


        x = x * mask_float


        denominator = (
            mask_float
            .sum(dim=1)
            .clamp(min=1.0)
        )


        return (
            x.sum(dim=1)
            /
            denominator
        )


    # ========================================================
    # FORWARD
    # ========================================================

    def forward(
        self,
        image,
        input_ids,
        attention_mask
    ):


        # ====================================================
        # IMAGE TOKENS
        # ====================================================

        image_tokens = (
            self.image_encoder(
                image
            )
        )


        image_tokens = (
            self.image_projection(
                image_tokens
            )
        )


        # ====================================================
        # TEXT TOKENS
        # ====================================================

        text_tokens = (
            self.text_encoder(

                input_ids,

                attention_mask
            )
        )


        # ====================================================
        # TEXT PADDING MASK
        #
        # MultiheadAttention wants:
        #
        # True  = ignore
        # False = valid
        #
        # Dataset attention mask uses:
        #
        # True  = valid
        # False = padding
        #
        # Therefore invert it.
        # ====================================================

        text_key_padding_mask = (
            ~attention_mask
        )


        # ====================================================
        # IMAGE → TEXT
        # ====================================================

        image_attended, _ = (
            self.image_to_text_attention(

                query=image_tokens,

                key=text_tokens,

                value=text_tokens,

                key_padding_mask=
                    text_key_padding_mask,

                need_weights=False
            )
        )


        image_tokens = (
            self.image_attn_norm(

                image_tokens
                +
                image_attended
            )
        )


        image_tokens = (
            self.image_ffn_norm(

                image_tokens
                +
                self.image_ffn(
                    image_tokens
                )
            )
        )


        # ====================================================
        # TEXT → IMAGE
        # ====================================================

        text_attended, _ = (
            self.text_to_image_attention(

                query=text_tokens,

                key=image_tokens,

                value=image_tokens,

                need_weights=False
            )
        )


        text_tokens = (
            self.text_attn_norm(

                text_tokens
                +
                text_attended
            )
        )


        text_tokens = (
            self.text_ffn_norm(

                text_tokens
                +
                self.text_ffn(
                    text_tokens
                )
            )
        )


        # ====================================================
        # POOLING
        # ====================================================

        # Image has no padding.

        pooled_image = (
            image_tokens.mean(
                dim=1
            )
        )


        # Text DOES have padding,
        # so use masked mean.

        pooled_text = (
            self.masked_mean(

                text_tokens,

                attention_mask
            )
        )


        # ====================================================
        # MULTIMODAL INTERACTION
        # ====================================================

        product = (
            pooled_image
            *
            pooled_text
        )


        absolute_difference = torch.abs(

            pooled_image
            -
            pooled_text
        )


        joint_representation = torch.cat(

            [

                pooled_image,

                pooled_text,

                product,

                absolute_difference

            ],

            dim=1
        )


        # ====================================================
        # CLASSIFIER
        # ====================================================

        logits = self.classifier(
            joint_representation
        )


        return logits.squeeze(1)


# ============================================================
# METRICS
# ============================================================

def calculate_metrics(
    labels,
    probabilities
):

    predictions = (
        probabilities >= 0.5
    ).astype(int)


    return {

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


# ============================================================
# ARCHITECTURE SANITY CHECK
# ============================================================

@torch.no_grad()
def architecture_sanity_check(
    model,
    loader
):

    print()

    print("=" * 70)

    print(
        "RUNNING JOINT FUSION ARCHITECTURE SANITY CHECK"
    )

    print("=" * 70)


    batch = next(
        iter(loader)
    )


    images = batch[
        "image"
    ].to(DEVICE)


    input_ids = batch[
        "input_ids"
    ].to(DEVICE)


    attention_mask = batch[
        "attention_mask"
    ].to(DEVICE)


    labels = batch[
        "label"
    ].to(DEVICE)


    model.eval()


    # --------------------------------------------------------
    # Check image tokens
    # --------------------------------------------------------

    image_tokens = (
        model.image_encoder(
            images
        )
    )


    projected_image_tokens = (
        model.image_projection(
            image_tokens
        )
    )


    # --------------------------------------------------------
    # Check text tokens
    # --------------------------------------------------------

    text_tokens = (
        model.text_encoder(

            input_ids,

            attention_mask
        )
    )


    print(
        "Input image shape:",
        tuple(images.shape)
    )


    print(
        "DenseNet spatial tokens:",
        tuple(image_tokens.shape)
    )


    print(
        "Projected image tokens:",
        tuple(
            projected_image_tokens.shape
        )
    )


    print(
        "Text tokens:",
        tuple(
            text_tokens.shape
        )
    )


    # --------------------------------------------------------
    # Critical architecture assertions
    # --------------------------------------------------------

    if image_tokens.shape[1] <= 1:

        raise RuntimeError(

            "SANITY CHECK FAILED: "
            "DenseNet output is not spatially tokenized."
        )


    if (
        projected_image_tokens.shape[-1]
        != IMAGE_TOKEN_DIM
    ):

        raise RuntimeError(

            "SANITY CHECK FAILED: "
            "incorrect image token dimension."
        )


    if (
        text_tokens.shape[-1]
        != TEXT_FEATURE_DIM
    ):

        raise RuntimeError(

            "SANITY CHECK FAILED: "
            "incorrect text token dimension."
        )


    # --------------------------------------------------------
    # Full forward pass
    # --------------------------------------------------------

    logits = model(

        images,

        input_ids,

        attention_mask
    )


    print(
        "Logits shape:",
        tuple(logits.shape)
    )


    expected_shape = (
        images.shape[0],
    )


    if tuple(logits.shape) != expected_shape:

        raise RuntimeError(

            "SANITY CHECK FAILED: "
            f"expected logits shape "
            f"{expected_shape}, "
            f"got {tuple(logits.shape)}"
        )


    if not torch.isfinite(
        logits
    ).all():

        raise RuntimeError(

            "SANITY CHECK FAILED: "
            "NaN/Inf detected in logits."
        )


    if not torch.isfinite(
        labels
    ).all():

        raise RuntimeError(

            "SANITY CHECK FAILED: "
            "NaN/Inf detected in labels."
        )


    print()
    print(
        "✓ DenseNet checkpoint loading: VALID"
    )

    print(
        "✓ Spatial image tokens: VALID"
    )

    print(
        "✓ Text token sequence: VALID"
    )

    print(
        "✓ Cross-modal attention: VALID"
    )

    print(
        "✓ Forward pass: VALID"
    )

    print(
        "✓ No NaN/Inf detected"
    )

    print()

    print(
        "ARCHITECTURE SANITY CHECK PASSED"
    )

    print("=" * 70)


# ============================================================
# TRAIN / VALIDATION EPOCH
# ============================================================

def run_epoch(
    model,
    loader,
    criterion,
    optimizer=None
):

    training = (
        optimizer is not None
    )


    if training:

        model.train()

    else:

        model.eval()


    total_loss = 0.0

    all_labels = []

    all_probabilities = []


    for batch in loader:


        images = batch[
            "image"
        ].to(

            DEVICE,

            non_blocking=True
        )


        input_ids = batch[
            "input_ids"
        ].to(

            DEVICE,

            non_blocking=True
        )


        attention_mask = batch[
            "attention_mask"
        ].to(

            DEVICE,

            non_blocking=True
        )


        labels = batch[
            "label"
        ].to(

            DEVICE,

            non_blocking=True
        )


        if training:

            optimizer.zero_grad(
                set_to_none=True
            )


        with torch.set_grad_enabled(
            training
        ):


            logits = model(

                images,

                input_ids,

                attention_mask
            )


            loss = criterion(

                logits,

                labels
            )


            if training:

                loss.backward()


                torch.nn.utils.clip_grad_norm_(

                    model.parameters(),

                    max_norm=1.0
                )


                optimizer.step()


        total_loss += (

            loss.item()
            *
            labels.size(0)
        )


        probabilities = (
            torch.sigmoid(
                logits
            )
        )


        all_labels.extend(

            labels.detach()
            .cpu()
            .numpy()
        )


        all_probabilities.extend(

            probabilities.detach()
            .cpu()
            .numpy()
        )


    all_labels = np.asarray(
        all_labels
    )


    all_probabilities = np.asarray(
        all_probabilities
    )


    average_loss = (

        total_loss
        /
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

    print(
        "JOINT FUSION — TRUE MULTIMODAL CROSS-ATTENTION TRAINING"
    )

    print("=" * 70)


    print()

    print(
        "Device:",
        DEVICE
    )


    # ========================================================
    # LOAD MANIFEST
    # ========================================================

    print()

    print(
        "Loading fusion manifest..."
    )


    df = pd.read_csv(
        MANIFEST_PATH
    )


    required_columns = {

        "patient_id",

        "image_path",

        "clinical_text",

        "label",

        "final_split"
    }


    missing_columns = (
        required_columns
        -
        set(df.columns)
    )


    if missing_columns:

        raise ValueError(

            "Manifest is missing required "
            f"columns: {sorted(missing_columns)}"
        )


    train_df = df[
        df["final_split"] == "train"
    ].copy()


    val_df = df[
        df["final_split"] == "val"
    ].copy()


    if (
        len(train_df) == 0
        or
        len(val_df) == 0
    ):

        raise ValueError(
            "Train/validation split is empty."
        )


    print(
        "Total records:",
        len(df)
    )


    print(
        "Training records:",
        len(train_df)
    )


    print(
        "Validation records:",
        len(val_df)
    )


    # ========================================================
    # LABEL CHECK
    # ========================================================

    train_labels = (
        train_df["label"]
        .astype(int)
        .values
    )


    val_labels = (
        val_df["label"]
        .astype(int)
        .values
    )


    print()

    print(
        "Training labels:",
        np.bincount(
            train_labels
        )
    )


    print(
        "Validation labels:",
        np.bincount(
            val_labels
        )
    )


    # ========================================================
    # LOAD TRAINING VOCABULARY
    # ========================================================

    print()

    print(
        "Loading training vocabulary..."
    )


    with open(

        VOCAB_PATH,

        "r",

        encoding="utf-8"

    ) as f:

        vocab_data = json.load(f)


    vocab = vocab_data[
        "vocabulary"
    ]


    vocab_size = len(vocab)


    pad_idx = vocab[
        "<PAD>"
    ]


    if "<UNK>" not in vocab:

        raise ValueError(
            "Vocabulary does not contain <UNK>."
        )


    print(
        "Vocabulary size:",
        vocab_size
    )


    print(
        "PAD index:",
        pad_idx
    )


    # ========================================================
    # DATASETS
    # ========================================================

    print()

    print(
        "Creating datasets..."
    )


    train_dataset = (
        JointFusionDataset(

            train_df,

            vocab
        )
    )


    val_dataset = (
        JointFusionDataset(

            val_df,

            vocab
        )
    )


    train_loader = DataLoader(

        train_dataset,

        batch_size=BATCH_SIZE,

        shuffle=True,

        num_workers=0,

        pin_memory=torch.cuda.is_available()
    )


    val_loader = DataLoader(

        val_dataset,

        batch_size=BATCH_SIZE,

        shuffle=False,

        num_workers=0,

        pin_memory=torch.cuda.is_available()
    )


    # ========================================================
    # BUILD MODEL
    # ========================================================

    print()

    print(
        "Building Joint Fusion model..."
    )


    model = JointFusionModel(

        vocab_size=vocab_size,

        pad_idx=pad_idx,

        checkpoint_path=CHECKPOINT_PATH

    ).to(DEVICE)


    # ========================================================
    # PARAMETER SUMMARY
    # ========================================================

    total_parameters = sum(

        p.numel()

        for p in model.parameters()
    )


    trainable_parameters = sum(

        p.numel()

        for p in model.parameters()

        if p.requires_grad
    )


    print()

    print(
        "Total parameters:",
        f"{total_parameters:,}"
    )


    print(
        "Trainable parameters:",
        f"{trainable_parameters:,}"
    )


    # ========================================================
    # CRITICAL:
    # RUN SANITY CHECK BEFORE TRAINING
    # ========================================================

    architecture_sanity_check(

        model,

        val_loader
    )


    # ========================================================
    # CLASS WEIGHT
    # ========================================================

    positive_count = int(

        (
            train_df["label"]
            ==
            1
        ).sum()
    )


    negative_count = int(

        (
            train_df["label"]
            ==
            0
        ).sum()
    )


    if positive_count == 0:

        raise ValueError(
            "Training set contains no positive samples."
        )


    pos_weight_value = (

        negative_count
        /
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
        positive_count
    )


    print(
        "Negative samples:",
        negative_count
    )


    print(
        "Positive class weight:",
        float(pos_weight_value)
    )


    criterion = nn.BCEWithLogitsLoss(

        pos_weight=pos_weight
    )


    # ========================================================
    # OPTIMIZER
    # ========================================================

    image_parameters = list(

        model.image_encoder.parameters()
    )


    other_parameters = [

        parameter

        for name, parameter
        in model.named_parameters()

        if not name.startswith(
            "image_encoder."
        )
    ]


    optimizer = torch.optim.AdamW(

        [

            {
                "params":
                    image_parameters,

                "lr":
                    DENSENET_LEARNING_RATE
            },

            {
                "params":
                    other_parameters,

                "lr":
                    LEARNING_RATE
            }

        ],

        weight_decay=WEIGHT_DECAY
    )


    # ========================================================
    # SCHEDULER
    # ========================================================

    scheduler = (
        torch.optim.lr_scheduler
        .ReduceLROnPlateau(

            optimizer,

            mode="max",

            factor=0.5,

            patience=2
        )
    )


    # ========================================================
    # TRAINING
    # ========================================================

    print()

    print("=" * 70)

    print(
        "STARTING TRUE JOINT MULTIMODAL TRAINING"
    )

    print("=" * 70)


    print()

    print(
        "DenseNet learning rate:",
        DENSENET_LEARNING_RATE
    )


    print(
        "Fusion/text learning rate:",
        LEARNING_RATE
    )


    print(
        "Batch size:",
        BATCH_SIZE
    )


    print(
        "Maximum epochs:",
        EPOCHS
    )


    best_auc = -1.0

    best_epoch = 0

    patience_counter = 0

    history = []


    for epoch in range(

        1,

        EPOCHS + 1

    ):


        train_metrics = run_epoch(

            model,

            train_loader,

            criterion,

            optimizer
        )


        val_metrics = run_epoch(

            model,

            val_loader,

            criterion
        )


        scheduler.step(

            val_metrics[
                "roc_auc"
            ]
        )


        print()

        print(
            f"Epoch {epoch:02d}/{EPOCHS}"
        )


        print(
            f"  Train Loss   : "
            f"{train_metrics['loss']:.4f}"
        )


        print(
            f"  Val Loss     : "
            f"{val_metrics['loss']:.4f}"
        )


        print(
            f"  Val Accuracy : "
            f"{val_metrics['accuracy']:.4f}"
        )


        print(
            f"  Val Precision: "
            f"{val_metrics['precision']:.4f}"
        )


        print(
            f"  Val Recall   : "
            f"{val_metrics['recall']:.4f}"
        )


        print(
            f"  Val F1       : "
            f"{val_metrics['f1_score']:.4f}"
        )


        print(
            f"  Val ROC-AUC  : "
            f"{val_metrics['roc_auc']:.4f}"
        )


        history.append({

            "epoch": epoch,

            "train":
                train_metrics,

            "validation":
                val_metrics
        })


        current_auc = (
            val_metrics[
                "roc_auc"
            ]
        )


        # ====================================================
        # SAVE BEST MODEL
        # ====================================================

        if current_auc > best_auc:

            best_auc = current_auc

            best_epoch = epoch

            patience_counter = 0


            torch.save(

                {

                    "model_state_dict":
                        model.state_dict(),

                    "vocab_size":
                        vocab_size,

                    "pad_idx":
                        pad_idx,

                    "best_val_roc_auc":
                        best_auc,

                    "best_epoch":
                        best_epoch,

                    "image_size":
                        IMAGE_SIZE,

                    "max_sequence_length":
                        MAX_SEQUENCE_LENGTH,

                    "embedding_dim":
                        EMBEDDING_DIM,

                    "lstm_hidden_dim":
                        LSTM_HIDDEN_DIM,

                    "image_token_dim":
                        IMAGE_TOKEN_DIM,

                    "num_heads":
                        NUM_HEADS,

                    "fusion_type":
                        "joint_multimodal_cross_attention"

                },

                MODEL_PATH
            )


            print(
                "  ✓ New best Joint Fusion "
                "model saved."
            )


        else:

            patience_counter += 1


        # ====================================================
        # EARLY STOPPING
        # ====================================================

        if (
            patience_counter
            >=
            PATIENCE
        ):

            print()

            print(
                "Early stopping triggered."
            )

            break


    # ========================================================
    # LOAD BEST MODEL
    # ========================================================

    print()

    print("=" * 70)

    print(
        "LOADING BEST JOINT FUSION MODEL"
    )

    print("=" * 70)


    best_checkpoint = torch.load(

        MODEL_PATH,

        map_location=DEVICE
    )


    load_result = model.load_state_dict(

        best_checkpoint[
            "model_state_dict"
        ],

        strict=True
    )


    print(
        "Best checkpoint loaded:"
    )

    print(
        load_result
    )


    # ========================================================
    # FINAL VALIDATION
    # ========================================================

    final_metrics = run_epoch(

        model,

        val_loader,

        criterion
    )


    # ========================================================
    # FINAL RESULT
    # ========================================================

    print()

    print("=" * 70)

    print(
        "JOINT FUSION — FINAL VALIDATION RESULTS"
    )

    print("=" * 70)


    print()

    print(
        f"Accuracy : "
        f"{final_metrics['accuracy']:.4f}"
    )


    print(
        f"Precision: "
        f"{final_metrics['precision']:.4f}"
    )


    print(
        f"Recall   : "
        f"{final_metrics['recall']:.4f}"
    )


    print(
        f"F1 Score : "
        f"{final_metrics['f1_score']:.4f}"
    )


    print(
        f"ROC-AUC  : "
        f"{final_metrics['roc_auc']:.4f}"
    )


    print()

    print(
        "Confusion Matrix:"
    )


    print(
        np.array(
            final_metrics[
                "confusion_matrix"
            ]
        )
    )


    # ========================================================
    # SAVE RESULTS
    # ========================================================

    results = {

        "model":
            "Joint Fusion",

        "fusion_type":
            "Joint multimodal cross-attention fusion",

        "architecture": {

            "image_encoder":
                "DenseNet121 spatial feature encoder",

            "text_encoder":
                "2-layer bidirectional LSTM",

            "cross_modal_fusion":
                "bidirectional image-text cross-attention",

            "image_tokens":
                "DenseNet spatial regions; "
                "no global pooling before attention",

            "text_padding_mask":
                True,

            "interaction_features": [

                "pooled image representation",

                "masked pooled text representation",

                "element-wise product",

                "absolute difference"
            ]
        },

        "device":
            str(DEVICE),

        "seed":
            SEED,

        "total_records":
            len(df),

        "train_records":
            len(train_df),

        "validation_records":
            len(val_df),

        "vocabulary_size":
            vocab_size,

        "max_sequence_length":
            MAX_SEQUENCE_LENGTH,

        "image_size":
            IMAGE_SIZE,

        "image_token_dim":
            IMAGE_TOKEN_DIM,

        "attention_heads":
            NUM_HEADS,

        "total_parameters":
            total_parameters,

        "trainable_parameters":
            trainable_parameters,

        "best_epoch":
            best_epoch,

        "best_val_roc_auc":
            float(best_auc),

        "final_metrics":
            final_metrics,

        "training_history":
            history
    }


    with open(

        RESULTS_PATH,

        "w",

        encoding="utf-8"

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

    print(
        "JOINT FUSION TRAINING COMPLETE"
    )

    print("=" * 70)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":

    main()