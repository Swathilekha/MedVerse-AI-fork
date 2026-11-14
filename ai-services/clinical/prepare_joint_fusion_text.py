import os
import json
import re
from collections import Counter

import pandas as pd


# ============================================================
# CONFIGURATION
# ============================================================

MANIFEST_PATH = (
    "data/clinical_text/fusion_experiment_dataset.csv"
)

OUTPUT_DIR = "data/clinical_text"

VOCAB_PATH = os.path.join(
    OUTPUT_DIR,
    "joint_fusion_vocab.json"
)

TRAIN_TEXT_PATH = os.path.join(
    OUTPUT_DIR,
    "joint_fusion_train_text.json"
)

VAL_TEXT_PATH = os.path.join(
    OUTPUT_DIR,
    "joint_fusion_val_text.json"
)

MIN_FREQUENCY = 2

MAX_VOCAB_SIZE = 10000

MAX_SEQUENCE_LENGTH = 256


# ============================================================
# TOKENIZER
# ============================================================

def tokenize(text):

    text = str(text).lower()

    # Keep words and useful medical numbers
    tokens = re.findall(
        r"[a-z]+(?:[-'][a-z]+)*|\d+(?:\.\d+)?",
        text
    )

    return tokens


# ============================================================
# BUILD VOCABULARY
# ============================================================

def build_vocabulary(train_texts):

    counter = Counter()

    for text in train_texts:

        tokens = tokenize(text)

        counter.update(tokens)

    # Special tokens
    vocabulary = {

        "<PAD>": 0,

        "<UNK>": 1
    }

    # Most frequent training tokens
    candidates = [

        (token, count)

        for token, count
        in counter.most_common()

        if count >= MIN_FREQUENCY
    ]

    candidates = candidates[
        :MAX_VOCAB_SIZE - len(vocabulary)
    ]

    for token, count in candidates:

        if token not in vocabulary:

            vocabulary[token] = len(vocabulary)

    return vocabulary, counter


# ============================================================
# ENCODE TEXT
# ============================================================

def encode_text(
    text,
    vocabulary
):

    tokens = tokenize(text)

    token_ids = []

    for token in tokens:

        token_ids.append(
            vocabulary.get(
                token,
                vocabulary["<UNK>"]
            )
        )

    # Truncate
    token_ids = token_ids[
        :MAX_SEQUENCE_LENGTH
    ]

    # Padding
    if len(token_ids) < MAX_SEQUENCE_LENGTH:

        token_ids += [

            vocabulary["<PAD>"]
        ] * (
            MAX_SEQUENCE_LENGTH -
            len(token_ids)
        )

    return token_ids


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print(
        "JOINT FUSION — TEXT PREPROCESSING"
    )
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load dataset
    # --------------------------------------------------------

    print()
    print(
        "Loading fusion manifest..."
    )

    df = pd.read_csv(
        MANIFEST_PATH
    )

    print(
        "Total records:",
        len(df)
    )

    # --------------------------------------------------------
    # Same split as Early Fusion
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
    # Extract text
    # --------------------------------------------------------

    train_texts = (
        train_df["clinical_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    val_texts = (
        val_df["clinical_text"]
        .fillna("")
        .astype(str)
        .tolist()
    )

    # --------------------------------------------------------
    # Build vocabulary ONLY from training text
    # --------------------------------------------------------

    print()
    print(
        "Building vocabulary from TRAINING text only..."
    )

    vocabulary, counter = build_vocabulary(
        train_texts
    )

    print(
        "Vocabulary size:",
        len(vocabulary)
    )

    print(
        "Minimum token frequency:",
        MIN_FREQUENCY
    )

    print(
        "Maximum vocabulary size:",
        MAX_VOCAB_SIZE
    )

    print(
        "Maximum sequence length:",
        MAX_SEQUENCE_LENGTH
    )

    # --------------------------------------------------------
    # Encode training text
    # --------------------------------------------------------

    print()
    print(
        "Encoding training text..."
    )

    train_encoded = []

    for text in train_texts:

        train_encoded.append(
            encode_text(
                text,
                vocabulary
            )
        )

    # --------------------------------------------------------
    # Encode validation text
    # --------------------------------------------------------

    print(
        "Encoding validation text..."
    )

    val_encoded = []

    for text in val_texts:

        val_encoded.append(
            encode_text(
                text,
                vocabulary
            )
        )

    # --------------------------------------------------------
    # Save vocabulary
    # --------------------------------------------------------

    vocab_data = {

        "vocabulary": vocabulary,

        "vocab_size": len(vocabulary),

        "pad_token": "<PAD>",

        "pad_index": vocabulary["<PAD>"],

        "unk_token": "<UNK>",

        "unk_index": vocabulary["<UNK>"],

        "min_frequency": MIN_FREQUENCY,

        "max_vocab_size": MAX_VOCAB_SIZE,

        "max_sequence_length":
            MAX_SEQUENCE_LENGTH
    }

    with open(
        VOCAB_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            vocab_data,
            f,
            indent=2
        )

    # --------------------------------------------------------
    # Save encoded datasets
    # --------------------------------------------------------

    train_records = []

    for i, (_, row) in enumerate(
        train_df.iterrows()
    ):

        train_records.append({

            "patient_id":
                str(row["patient_id"]),

            "label":
                int(row["label"]),

            "sequence":
                train_encoded[i]
        })

    val_records = []

    for i, (_, row) in enumerate(
        val_df.iterrows()
    ):

        val_records.append({

            "patient_id":
                str(row["patient_id"]),

            "label":
                int(row["label"]),

            "sequence":
                val_encoded[i]
        })

    with open(
        TRAIN_TEXT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            train_records,
            f
        )

    with open(
        VAL_TEXT_PATH,
        "w",
        encoding="utf-8"
    ) as f:

        json.dump(
            val_records,
            f
        )

    # --------------------------------------------------------
    # Statistics
    # --------------------------------------------------------

    sequence_lengths = []

    for text in train_texts:

        sequence_lengths.append(
            min(
                len(tokenize(text)),
                MAX_SEQUENCE_LENGTH
            )
        )

    average_length = (
        sum(sequence_lengths) /
        len(sequence_lengths)
    )

    print()
    print("=" * 70)
    print(
        "JOINT FUSION TEXT PREPROCESSING COMPLETE"
    )
    print("=" * 70)

    print()
    print(
        "Vocabulary size:",
        len(vocabulary)
    )

    print(
        "Training sequences:",
        len(train_encoded)
    )

    print(
        "Validation sequences:",
        len(val_encoded)
    )

    print(
        "Sequence length:",
        MAX_SEQUENCE_LENGTH
    )

    print(
        "Average training token length:",
        round(average_length, 2)
    )

    print()
    print(
        "Vocabulary saved to:"
    )

    print(
        VOCAB_PATH
    )

    print()
    print(
        "Training text saved to:"
    )

    print(
        TRAIN_TEXT_PATH
    )

    print()
    print(
        "Validation text saved to:"
    )

    print(
        VAL_TEXT_PATH
    )

    print()
    print("=" * 70)
    print("DONE")
    print("=" * 70)


if __name__ == "__main__":

    main()