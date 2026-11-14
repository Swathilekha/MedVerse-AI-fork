import os
import numpy as np
import pandas as pd
import json
import joblib

from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    brier_score_loss,
    confusion_matrix,
)


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

XRAY_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clinical_text",
    "xray_predictions.csv",
)

EVIDENCE_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clinical_text",
    "ai_evidence.csv",
)

DATASET_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clinical_text",
    "multimodal_dataset.csv",
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "data",
    "clinical_text",
)

FUSION_RESULTS_PATH = os.path.join(
    OUTPUT_DIR,
    "multimodal_fusion_results.csv",
)

METRICS_PATH = os.path.join(
    OUTPUT_DIR,
    "fusion_metrics.csv",
)

CONFLICT_PATH = os.path.join(
    OUTPUT_DIR,
    "fusion_conflict_analysis.csv",
)

CALIBRATION_PATH = os.path.join(
    OUTPUT_DIR,
    "fusion_calibration.csv",
)

AARF_MODEL_PATH = os.path.join(
    OUTPUT_DIR,
    "aarf_v3_model.joblib",
)

AARF_FEATURES_PATH = os.path.join(
    OUTPUT_DIR,
    "aarf_v3_features.json",
)

# ============================================================
# RESEARCH DESIGN PARAMETERS
# ============================================================

# These are fixed BEFORE validation evaluation.
# They are NOT tuned using validation performance.

VISUAL_HIGH_THRESHOLD = 0.75
VISUAL_LOW_THRESHOLD = 0.25

STRONG_CONFLICT_THRESHOLD = 0.75

# Selective prediction:
# defer when the proposed model is uncertain or evidence is conflicting.
DEFERRAL_LOW_CONFIDENCE = 0.60

# Findings explicitly treated as pneumonia-related SUPPORTING evidence.
# They are not treated as equivalent to a pneumonia diagnosis.
SUPPORTING_FINDINGS = {
    "consolidation",
    "lung_opacity",
}

# Findings retained in the evidence profile but NOT converted
# directly into pneumonia-positive evidence.
CONTEXTUAL_FINDINGS = {
    "atelectasis",
    "pulmonary_edema",
    "pleural_effusion",
    "pneumothorax",
    "cardiomegaly",
}

DIRECT_FINDING = "pneumonia"

ALL_FINDINGS = [
    DIRECT_FINDING,
    "consolidation",
    "lung_opacity",
    "atelectasis",
    "pulmonary_edema",
    "pleural_effusion",
    "pneumothorax",
    "cardiomegaly",
]


# ============================================================
# HELPERS
# ============================================================

def normalize_patient_id(value):
    if pd.isna(value):
        return ""
    return str(value).strip()


def safe_float(value, default=0.0):
    try:
        if pd.isna(value):
            return default
        return float(value)
    except Exception:
        return default


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -50, 50)))


def entropy_binary(p):
    """
    Binary entropy normalized to [0, 1].
    0 = highly confident
    1 = maximum uncertainty
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-8, 1 - 1e-8)
    h = -(p * np.log2(p) + (1 - p) * np.log2(1 - p))
    return h


# ============================================================
# TEXT EVIDENCE AGGREGATION
# ============================================================

def aggregate_text_evidence(evidence_df):
    """
    Convert sentence-level evidence into one patient/finding profile.

    Important:
    PRESENT / ABSENT / UNCERTAIN are preserved.
    We do NOT interpret NLP confidence as a calibrated disease
    probability.
    """

    e = evidence_df.copy()

    e["patient_id"] = e["patient_id"].apply(normalize_patient_id)
    e["finding"] = e["finding"].astype(str).str.lower().str.strip()
    e["status"] = e["status"].astype(str).str.lower().str.strip()

    if "model_confidence" in e.columns:
        e["model_confidence"] = pd.to_numeric(
            e["model_confidence"],
            errors="coerce"
        ).fillna(0.0)
    else:
        e["model_confidence"] = 0.0

    if "extraction_confidence" in e.columns:
        e["extraction_confidence"] = pd.to_numeric(
            e["extraction_confidence"],
            errors="coerce"
        ).fillna(0.0)
    else:
        e["extraction_confidence"] = 0.0

    e["combined_confidence"] = (
        e["model_confidence"] *
        e["extraction_confidence"]
    )

    rows = []

    grouped = e.groupby(
        ["patient_id", "finding"],
        dropna=False
    )

    for (patient_id, finding), group in grouped:

        present = group.loc[
            group["status"] == "present",
            "combined_confidence"
        ]

        absent = group.loc[
            group["status"] == "absent",
            "combined_confidence"
        ]

        uncertain = group.loc[
            group["status"] == "uncertain",
            "combined_confidence"
        ]

        max_present = present.max() if len(present) else 0.0
        max_absent = absent.max() if len(absent) else 0.0
        max_uncertain = uncertain.max() if len(uncertain) else 0.0

        has_present = len(present) > 0
        has_absent = len(absent) > 0
        has_uncertain = len(uncertain) > 0

        if has_present and has_absent:
            state = "MIXED"
        elif has_present:
            state = "PRESENT"
        elif has_absent:
            state = "ABSENT"
        elif has_uncertain:
            state = "UNCERTAIN"
        else:
            state = "NO_EVIDENCE"

        if state == "PRESENT":
            support = max_present
        elif state == "ABSENT":
            support = 1.0 - max_absent
        elif state == "UNCERTAIN":
            support = 0.5
        elif state == "MIXED":
            support = (
                max_present +
                (1.0 - max_absent)
            ) / 2.0
        else:
            support = 0.5

        rows.append({
            "patient_id": patient_id,
            "finding": finding,
            "text_state": state,
            "text_support": float(support),
            "text_confidence": float(
                max(
                    max_present,
                    max_absent,
                    max_uncertain
                )
            ),
            "max_present": float(max_present),
            "max_absent": float(max_absent),
            "max_uncertain": float(max_uncertain),
            "evidence_count": len(group),
        })

    return pd.DataFrame(rows)


# ============================================================
# BUILD COMPLETE CLINICAL PROFILE
# ============================================================

def build_evidence_profile(xray_df, evidence_df):
    """
    Creates one row per X-ray record with a complete clinical
    evidence profile.

    Direct, supporting and contextual findings remain separate.
    """

    profile = xray_df.copy()

    profile["patient_id"] = profile["patient_id"].apply(
        normalize_patient_id
    )

    evidence_agg = aggregate_text_evidence(evidence_df)

    # --------------------------------------------------------
    # Pivot each finding into explicit columns
    # --------------------------------------------------------

    for finding in ALL_FINDINGS:

        sub = evidence_agg[
            evidence_agg["finding"] == finding
        ].copy()

        if len(sub) == 0:
            profile[f"{finding}_state"] = "NO_EVIDENCE"
            profile[f"{finding}_support"] = 0.5
            profile[f"{finding}_confidence"] = 0.0
            profile[f"{finding}_count"] = 0
            continue

        sub = sub.set_index("patient_id")

        profile[f"{finding}_state"] = (
            profile["patient_id"]
            .map(sub["text_state"])
            .fillna("NO_EVIDENCE")
        )

        profile[f"{finding}_support"] = (
            profile["patient_id"]
            .map(sub["text_support"])
            .fillna(0.5)
            .astype(float)
        )

        profile[f"{finding}_confidence"] = (
            profile["patient_id"]
            .map(sub["text_confidence"])
            .fillna(0.0)
            .astype(float)
        )

        profile[f"{finding}_count"] = (
            profile["patient_id"]
            .map(sub["evidence_count"])
            .fillna(0)
            .astype(int)
        )

    # --------------------------------------------------------
    # Evidence coverage
    # --------------------------------------------------------

    state_cols = [
        f"{finding}_state"
        for finding in ALL_FINDINGS
    ]

    profile["evidence_findings_present"] = (
        profile[state_cols]
        .apply(
            lambda row: sum(
                state != "NO_EVIDENCE"
                for state in row
            ),
            axis=1
        )
    )

    profile["evidence_coverage"] = (
        profile["evidence_findings_present"] /
        float(len(ALL_FINDINGS))
    )

    # --------------------------------------------------------
    # Direct pneumonia evidence
    # --------------------------------------------------------

    profile["direct_pneumonia_support"] = (
        profile["pneumonia_support"]
    )

    profile["direct_pneumonia_confidence"] = (
        profile["pneumonia_confidence"]
    )

    # --------------------------------------------------------
    # Supporting evidence profile
    # --------------------------------------------------------

    profile["supporting_present_count"] = (
        (profile["consolidation_state"] == "PRESENT").astype(int)
        +
        (profile["lung_opacity_state"] == "PRESENT").astype(int)
    )

    profile["supporting_uncertain_count"] = (
        (profile["consolidation_state"] == "UNCERTAIN").astype(int)
        +
        (profile["lung_opacity_state"] == "UNCERTAIN").astype(int)
    )

    profile["supporting_absent_count"] = (
        (profile["consolidation_state"] == "ABSENT").astype(int)
        +
        (profile["lung_opacity_state"] == "ABSENT").astype(int)
    )

    profile["supporting_confidence"] = (
        profile[
            [
                "consolidation_confidence",
                "lung_opacity_confidence",
            ]
        ].max(axis=1)
    )

    # --------------------------------------------------------
    # Textual uncertainty
    # --------------------------------------------------------

    profile["uncertainty_count"] = sum(
        (
            profile[f"{finding}_state"] == "UNCERTAIN"
        ).astype(int)
        for finding in ALL_FINDINGS
    )

    profile["mixed_count"] = sum(
        (
            profile[f"{finding}_state"] == "MIXED"
        ).astype(int)
        for finding in ALL_FINDINGS
    )

    profile["text_uncertainty"] = (
        profile["uncertainty_count"] /
        float(len(ALL_FINDINGS))
    )

    return profile


# ============================================================
# SEMANTIC CONFLICT DETECTION
# ============================================================

def classify_visual_state(probability):
    if probability >= VISUAL_HIGH_THRESHOLD:
        return "STRONG_POSITIVE"
    elif probability <= VISUAL_LOW_THRESHOLD:
        return "STRONG_NEGATIVE"
    else:
        return "UNCERTAIN"


def detect_semantic_conflict(row):
    """
    Clinical-semantic conflict is different from numerical
    probability disagreement.

    Strong visual positive + explicit text absence
    OR
    Strong visual negative + explicit text presence.
    """

    visual_state = classify_visual_state(
        row["xray_probability"]
    )

    text_state = row["pneumonia_state"]

    if (
        visual_state == "STRONG_POSITIVE"
        and text_state == "ABSENT"
    ):
        return "VISUAL_POSITIVE_TEXT_ABSENT"

    if (
        visual_state == "STRONG_NEGATIVE"
        and text_state == "PRESENT"
    ):
        return "VISUAL_NEGATIVE_TEXT_PRESENT"

    return "NONE"


def calculate_discordance(row):
    """
    Measures the gap between visual probability and the
    NLP-derived evidence-support score.

    IMPORTANT:
    These are different evidence representations, so this
    is reported as a support-score gap rather than treating
    it as calibrated probability disagreement.
    """

    xray_probability = row["xray_probability"]
    text_support = row["pneumonia_support"]

    return abs(
        xray_probability -
        text_support
    )


# ============================================================
# NAIVE FUSION
# ============================================================

def naive_fusion_probability(row):
    """
    Baseline:
        average(X-ray probability, text evidence support)

    text_support is an evidence-support score, not a calibrated
    pneumonia probability.
    """

    return (
        row["xray_probability"] +
        row["pneumonia_support"]
    ) / 2.0


# ============================================================
# META-LEVEL FEATURE CONSTRUCTION
# ============================================================

def create_meta_features(df, mode="full"):
    """
    Build fusion features.

    mode:
        full       = X-ray + direct + supporting + uncertainty
        direct     = X-ray + direct pneumonia evidence
        xray       = X-ray-only feature representation
    """

    features = pd.DataFrame(index=df.index)

    # --------------------------------------------------------
    # X-ray evidence
    # --------------------------------------------------------

    xray_prob = df["xray_probability"].astype(float)

    features["xray_probability"] = xray_prob

    features["xray_uncertainty"] = entropy_binary(
        xray_prob
    )

    # --------------------------------------------------------
    # Direct pneumonia evidence
    # --------------------------------------------------------

    if mode in ["full", "direct"]:

        features["pneumonia_present"] = (
            df["pneumonia_state"] == "PRESENT"
        ).astype(float)

        features["pneumonia_absent"] = (
            df["pneumonia_state"] == "ABSENT"
        ).astype(float)

        features["pneumonia_uncertain"] = (
            df["pneumonia_state"] == "UNCERTAIN"
        ).astype(float)

        features["pneumonia_mixed"] = (
            df["pneumonia_state"] == "MIXED"
        ).astype(float)

        features["pneumonia_support"] = (
            df["pneumonia_support"]
        )

        features["pneumonia_confidence"] = (
            df["pneumonia_confidence"]
        )

    # --------------------------------------------------------
    # Supporting pulmonary findings
    # --------------------------------------------------------

    if mode == "full":

        for finding in SUPPORTING_FINDINGS:

            features[f"{finding}_present"] = (
                df[f"{finding}_state"] == "PRESENT"
            ).astype(float)

            features[f"{finding}_absent"] = (
                df[f"{finding}_state"] == "ABSENT"
            ).astype(float)

            features[f"{finding}_uncertain"] = (
                df[f"{finding}_state"] == "UNCERTAIN"
            ).astype(float)

            features[f"{finding}_confidence"] = (
                df[f"{finding}_confidence"]
            )

            features[f"{finding}_support"] = (
                df[f"{finding}_support"]
            )

        features["evidence_coverage"] = (
            df["evidence_coverage"]
        )

        features["uncertainty_ratio"] = (
            df["text_uncertainty"]
        )

        features["mixed_ratio"] = (
            df["mixed_count"] /
            float(len(ALL_FINDINGS))
        )

        features["supporting_present_count"] = (
            df["supporting_present_count"]
        )

        features["supporting_uncertain_count"] = (
            df["supporting_uncertain_count"]
        )

        features["semantic_conflict"] = (
            df["semantic_conflict"] != "NONE"
        ).astype(float)

    return features.replace(
        [np.inf, -np.inf],
        0
    ).fillna(0)


# ============================================================
# METRICS
# ============================================================

def expected_calibration_error(y_true, probabilities, bins=10):

    y_true = np.asarray(y_true)
    probabilities = np.asarray(probabilities)

    bin_edges = np.linspace(0, 1, bins + 1)

    ece = 0.0

    for i in range(bins):

        if i == bins - 1:
            mask = (
                (probabilities >= bin_edges[i]) &
                (probabilities <= bin_edges[i + 1])
            )
        else:
            mask = (
                (probabilities >= bin_edges[i]) &
                (probabilities < bin_edges[i + 1])
            )

        if not np.any(mask):
            continue

        confidence = probabilities[mask].mean()
        accuracy = y_true[mask].mean()

        ece += (
            np.sum(mask) /
            len(y_true)
        ) * abs(
            accuracy -
            confidence
        )

    return float(ece)


def calculate_metrics(y_true, probabilities):

    predictions = (
        probabilities >= 0.5
    ).astype(int)

    result = {
        "accuracy": accuracy_score(
            y_true,
            predictions
        ),
        "precision": precision_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "recall": recall_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "f1": f1_score(
            y_true,
            predictions,
            zero_division=0
        ),
        "brier_score": brier_score_loss(
            y_true,
            probabilities
        ),
        "ece": expected_calibration_error(
            y_true,
            probabilities
        ),
    }

    try:
        result["roc_auc"] = roc_auc_score(
            y_true,
            probabilities
        )
    except ValueError:
        result["roc_auc"] = np.nan

    tn, fp, fn, tp = confusion_matrix(
        y_true,
        predictions,
        labels=[0, 1]
    ).ravel()

    result["tn"] = tn
    result["fp"] = fp
    result["fn"] = fn
    result["tp"] = tp

    return result


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AARF v3 - CONFIDENCE-AWARE MULTIMODAL EVIDENCE FUSION")
    print("=" * 70)

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Load data
    # --------------------------------------------------------

    print("\nLoading datasets...")

    xray = pd.read_csv(
        XRAY_PATH
    )

    evidence = pd.read_csv(
        EVIDENCE_PATH
    )

    dataset = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"X-ray predictions: {len(xray)}"
    )

    print(
        f"Clinical evidence rows: {len(evidence)}"
    )

    print(
        f"Multimodal dataset rows: {len(dataset)}"
    )

    # --------------------------------------------------------
    # Normalize identifiers
    # --------------------------------------------------------

    xray["patient_id"] = (
        xray["patient_id"]
        .apply(normalize_patient_id)
    )

    evidence["patient_id"] = (
        evidence["patient_id"]
        .apply(normalize_patient_id)
    )

    dataset["patient_id"] = (
        dataset["patient_id"]
        .apply(normalize_patient_id)
    )

    # --------------------------------------------------------
    # Inspect columns
    # --------------------------------------------------------

    probability_column = None

    for candidate in [
        "pneumonia_probability",
        "probability",
        "xray_probability",
        "PNEUMONIA_probability",
        "PNEUMONIA_prob",
    ]:
        if candidate in xray.columns:
            probability_column = candidate
            break

    if probability_column is None:

        probability_candidates = [
            c for c in xray.columns
            if "pneumonia" in c.lower()
            and (
                "prob" in c.lower()
                or "score" in c.lower()
            )
        ]

        if probability_candidates:
            probability_column = (
                probability_candidates[0]
            )

    if probability_column is None:
        raise ValueError(
            "Could not identify pneumonia probability column "
            f"in xray_predictions.csv.\nColumns: "
            f"{list(xray.columns)}"
        )

    xray["xray_probability"] = (
        pd.to_numeric(
            xray[probability_column],
            errors="coerce"
        )
        .fillna(0.5)
        .clip(0, 1)
    )

    # --------------------------------------------------------
    # Ground-truth class
    # --------------------------------------------------------

    class_column = None

    for candidate in [
        "class_name",
        "class",
        "label",
        "target",
    ]:
        if candidate in dataset.columns:
            class_column = candidate
            break

    if class_column is None:
        raise ValueError(
            "Could not find ground-truth class column."
        )

    dataset["y_true"] = (
        dataset[class_column]
        .astype(str)
        .str.upper()
        .map({
            "NORMAL": 0,
            "PNEUMONIA": 1,
        })
    )

    # --------------------------------------------------------
    # Merge X-ray predictions with dataset labels/split
    # --------------------------------------------------------

    merge_columns = [
        "patient_id",
        "y_true",
    ]

    if "final_split" in dataset.columns:
        merge_columns.append(
            "final_split"
        )

    # Use image path where possible for exact image alignment.
    if "local_path" in dataset.columns and "local_path" in xray.columns:

        merged = xray.merge(
            dataset[
                [
                    "local_path",
                    "patient_id",
                    "y_true"
                ]
                + (
                    ["final_split"]
                    if "final_split" in dataset.columns
                    else []
                )
            ],
            on="local_path",
            how="inner",
            suffixes=("", "_dataset"),
        )

    elif "png_path" in dataset.columns and "png_path" in xray.columns:

        merged = xray.merge(
            dataset[
                [
                    "png_path",
                    "patient_id",
                    "y_true"
                ]
                + (
                    ["final_split"]
                    if "final_split" in dataset.columns
                    else []
                )
            ],
            on="png_path",
            how="inner",
            suffixes=("", "_dataset"),
        )

    else:

        merged = xray.merge(
            dataset[
                merge_columns
            ].drop_duplicates(
                subset=["patient_id"]
            ),
            on="patient_id",
            how="inner",
            suffixes=("", "_dataset"),
        )

    if "patient_id_dataset" in merged.columns:
        merged["patient_id"] = (
            merged["patient_id_dataset"]
        )
        merged.drop(
            columns=["patient_id_dataset"],
            inplace=True
        )

    print(
        f"Aligned multimodal records: {len(merged)}"
    )

    if len(merged) == 0:
        raise RuntimeError(
            "No records could be aligned."
        )

    # --------------------------------------------------------
    # Build complete evidence profile
    # --------------------------------------------------------

    print("\nBuilding clinical evidence profile...")

    profile = build_evidence_profile(
        merged,
        evidence
    )

    # --------------------------------------------------------
    # Conflict analysis
    # --------------------------------------------------------

    profile["visual_state"] = (
        profile["xray_probability"]
        .apply(classify_visual_state)
    )

    profile["semantic_conflict"] = (
        profile.apply(
            detect_semantic_conflict,
            axis=1
        )
    )

    profile["numerical_discordance"] = (
        profile.apply(
            calculate_discordance,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Baseline 1: X-ray only
    # --------------------------------------------------------

    profile["xray_only_probability"] = (
        profile["xray_probability"]
    )

    # --------------------------------------------------------
    # Baseline 2: Naive fusion
    # --------------------------------------------------------

    profile["naive_fusion_probability"] = (
        profile.apply(
            naive_fusion_probability,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Proposed AARF v3
    # --------------------------------------------------------

    meta_features = create_meta_features(
        profile
    )

    # --------------------------------------------------------
    # Train/validation split
    # --------------------------------------------------------

    if "final_split" not in profile.columns:

        raise ValueError(
            "final_split is required for research-safe "
            "train/validation fusion."
        )

    profile["final_split"] = (
        profile["final_split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    train_mask = profile["final_split"].isin(
        ["train", "training"]
    )

    val_mask = profile["final_split"].isin(
        ["val", "validation"]
    )

    train = profile.loc[
        train_mask
    ].copy()

    val = profile.loc[
        val_mask
    ].copy()

    print(
        f"\nFusion training records: {len(train)}"
    )

    print(
        f"Fusion validation records: {len(val)}"
    )

    # --------------------------------------------------------
    # Patient leakage check
    # --------------------------------------------------------

    train_patients = set(
        train["patient_id"]
    )

    val_patients = set(
        val["patient_id"]
    )

    overlap = (
        train_patients &
        val_patients
    )

    print(
        f"Train/validation patient overlap: {len(overlap)}"
    )

    if len(overlap) > 0:
        raise RuntimeError(
            "Patient leakage detected between train and "
            "validation splits."
        )

    # --------------------------------------------------------
    # Train proposed fusion model
    # --------------------------------------------------------

    X_train = meta_features.loc[
        train.index
    ]

    y_train = train["y_true"].astype(int)

    X_val = meta_features.loc[
        val.index
    ]

    y_val = val["y_true"].astype(int)

    print(
        "\nTraining confidence-aware evidence fusion model..."
    )

    fusion_model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    fusion_model.fit(
        X_train,
        y_train
    )
        # --------------------------------------------------------
    # Persist trained AARF v3 model
    # --------------------------------------------------------

    joblib.dump(
        fusion_model,
        AARF_MODEL_PATH
    )

    with open(
        AARF_FEATURES_PATH,
        "w",
        encoding="utf-8"
    ) as f:
        json.dump(
            list(X_train.columns),
            f,
            indent=2
        )

    print("\nAARF V3 model saved:")
    print(AARF_MODEL_PATH)

    print("AARF V3 feature schema saved:")
    print(AARF_FEATURES_PATH)
    
    print("\nAARF V3 learned feature coefficients:")

    coef_df = pd.DataFrame({
        "feature": X_train.columns,
        "coefficient": fusion_model.coef_[0],
    })

    coef_df["absolute_coefficient"] = (
        coef_df["coefficient"].abs()
    )

    print(
        coef_df.sort_values(
            "absolute_coefficient",
            ascending=False
        ).to_string(index=False)
    )
    val_probability = (
        fusion_model.predict_proba(
            X_val
        )[:, 1]
    )

    # --------------------------------------------------------
    # Put proposed probability back into profile
    # --------------------------------------------------------

    profile["aarf_v3_probability"] = np.nan

    profile.loc[
        val.index,
        "aarf_v3_probability"
    ] = val_probability

    # --------------------------------------------------------
    # Deferral logic
    # --------------------------------------------------------

    profile["defer_reason"] = "NONE"

    val_xray = (
        profile.loc[
            val.index,
            "xray_probability"
        ]
    )

    val_semantic_conflict = (
        profile.loc[
            val.index,
            "semantic_conflict"
        ]
    )

    val_discordance = (
        profile.loc[
            val.index,
            "numerical_discordance"
        ]
    )

    val_aarf_probability = pd.Series(
        val_probability,
        index=val.index
    )

    fusion_confidence = np.maximum(
        val_aarf_probability,
        1.0 - val_aarf_probability
    )

    low_confidence = (
        fusion_confidence <
        DEFERRAL_LOW_CONFIDENCE
    )

    strong_semantic_conflict = (
        val_semantic_conflict != "NONE"
    )

    high_discordance = (
        val_discordance >=
        STRONG_CONFLICT_THRESHOLD
    )

    profile.loc[
        val.index[low_confidence],
        "defer_reason"
    ] = "LOW_FUSION_CONFIDENCE"

    profile.loc[
        val.index[
            strong_semantic_conflict
        ],
        "defer_reason"
    ] = "SEMANTIC_CONFLICT"

    profile.loc[
        val.index[
            high_discordance
        ],
        "defer_reason"
    ] = "HIGH_NUMERICAL_DISCORDANCE"

    # --------------------------------------------------------
    # Evidence interpretation
    # --------------------------------------------------------

    def evidence_interpretation(row):

        direct = row["pneumonia_state"]

        consolidation = row[
            "consolidation_state"
        ]

        opacity = row[
            "lung_opacity_state"
        ]

        if row["semantic_conflict"] != "NONE":
            return "CONFLICT"

        if direct == "PRESENT":
            return "DIRECT_PNEUMONIA_EVIDENCE"

        if direct == "UNCERTAIN":
            return "UNCERTAIN_PNEUMONIA_EVIDENCE"

        if (
            consolidation == "PRESENT"
            or opacity == "PRESENT"
        ):
            return "SUPPORTING_PULMONARY_EVIDENCE"

        if (
            consolidation == "UNCERTAIN"
            or opacity == "UNCERTAIN"
        ):
            return "UNCERTAIN_SUPPORTING_EVIDENCE"

        if direct == "ABSENT":
            return "EXPLICITLY_ABSENT"

        return "NO_DIRECT_PNEUMONIA_EVIDENCE"

    profile["evidence_interpretation"] = (
        profile.apply(
            evidence_interpretation,
            axis=1
        )
    )

    # --------------------------------------------------------
    # Final decision
    # --------------------------------------------------------

    profile["aarf_v3_prediction"] = np.nan

    profile.loc[
        val.index,
        "aarf_v3_prediction"
    ] = (
        val_probability >= 0.5
    ).astype(int)

    profile["decision"] = "NOT_EVALUATED"

    profile.loc[
        val.index,
        "decision"
    ] = np.where(
        profile.loc[
            val.index,
            "defer_reason"
        ] != "NONE",
        "DEFER_TO_CLINICIAN",
        np.where(
            val_probability >= 0.5,
            "PNEUMONIA_SUPPORT",
            "NO_PNEUMONIA_SUPPORT"
        )
    )

    # --------------------------------------------------------
    # Metrics - validation only
    # --------------------------------------------------------

    y_val_np = y_val.to_numpy()

    xray_val_probability = (
        profile.loc[
            val.index,
            "xray_only_probability"
        ].to_numpy()
    )

    naive_val_probability = (
        profile.loc[
            val.index,
            "naive_fusion_probability"
        ].to_numpy()
    )

    methods = {
        "X-RAY ONLY": xray_val_probability,
        "NAIVE FUSION": naive_val_probability,
        "AARF V3": val_probability,
    }

    metric_rows = []

    for method_name, probabilities in methods.items():

        metrics = calculate_metrics(
            y_val_np,
            probabilities
        )

        metrics["method"] = method_name
        metrics["split"] = "validation"

        metric_rows.append(metrics)

    metrics_df = pd.DataFrame(
        metric_rows
    )

    # --------------------------------------------------------
    # Conflict metrics
    # --------------------------------------------------------

    validation_profile = profile.loc[
        val.index
    ].copy()

    semantic_conflict_rate = (
        (
            validation_profile[
                "semantic_conflict"
            ] != "NONE"
        ).mean()
    )

    moderate_discordance_rate = (
        (
            validation_profile[
                "numerical_discordance"
            ] >= 0.30
        ).mean()
    )

    deferral_rate = (
        (
            validation_profile[
                "defer_reason"
            ] != "NONE"
        ).mean()
    )

    conflict_summary = pd.DataFrame({
        "metric": [
            "semantic_conflict_rate",
            "moderate_discordance_rate",
            "deferral_rate",
            "validation_records",
            "validation_patients",
        ],
        "value": [
            semantic_conflict_rate,
            moderate_discordance_rate,
            deferral_rate,
            len(validation_profile),
            validation_profile[
                "patient_id"
            ].nunique(),
        ]
    })

    # --------------------------------------------------------
    # Deferred-case selective risk
    # --------------------------------------------------------

    deferred = validation_profile[
        validation_profile["defer_reason"] != "NONE"
    ].copy()

    non_deferred = validation_profile[
        validation_profile["defer_reason"] == "NONE"
    ].copy()

    if len(deferred) > 0:

        deferred_predictions = (
            deferred["aarf_v3_probability"]
            >= 0.5
        ).astype(int)

        deferred_error = (
            deferred_predictions !=
            deferred["y_true"].astype(int)
        ).mean()

    else:

        deferred_error = np.nan

    if len(non_deferred) > 0:

        non_deferred_predictions = (
            non_deferred["aarf_v3_probability"]
            >= 0.5
        ).astype(int)

        non_deferred_error = (
            non_deferred_predictions !=
            non_deferred["y_true"].astype(int)
        ).mean()

    else:

        non_deferred_error = np.nan

    selective_rows = pd.DataFrame({
        "metric": [
            "deferred_cases",
            "deferred_error_rate",
            "non_deferred_cases",
            "non_deferred_error_rate",
            "coverage",
        ],
        "value": [
            len(deferred),
            deferred_error,
            len(non_deferred),
            non_deferred_error,
            len(non_deferred) /
            len(validation_profile)
            if len(validation_profile)
            else np.nan,
        ]
    })

    conflict_summary = pd.concat(
        [
            conflict_summary,
            selective_rows,
        ],
        ignore_index=True
    )

    # --------------------------------------------------------
    # Calibration bins
    # --------------------------------------------------------

    calibration_rows = []

    for method_name, probabilities in methods.items():

        y = y_val_np

        bins = np.linspace(
            0,
            1,
            11
        )

        for i in range(10):

            if i == 9:
                mask = (
                    (probabilities >= bins[i]) &
                    (probabilities <= bins[i + 1])
                )
            else:
                mask = (
                    (probabilities >= bins[i]) &
                    (probabilities < bins[i + 1])
                )

            if not np.any(mask):
                continue

            calibration_rows.append({
                "method": method_name,
                "bin_lower": bins[i],
                "bin_upper": bins[i + 1],
                "count": int(mask.sum()),
                "mean_confidence": float(
                    probabilities[mask].mean()
                ),
                "empirical_accuracy": float(
                    y[mask].mean()
                ),
            })

    calibration_df = pd.DataFrame(
        calibration_rows
    )

    # --------------------------------------------------------
    # Save complete research trace
    # --------------------------------------------------------

    # Add readable class.
    profile["true_class"] = (
        profile["y_true"]
        .map({
            0: "NORMAL",
            1: "PNEUMONIA",
        })
    )

    # Keep all important evidence/provenance columns.
    profile.to_csv(
        FUSION_RESULTS_PATH,
        index=False
    )

    metrics_df.to_csv(
        METRICS_PATH,
        index=False
    )

    validation_profile[
        [
            "patient_id",
            "true_class",
            "xray_probability",
            "visual_state",
            "pneumonia_state",
            "pneumonia_confidence",
            "consolidation_state",
            "consolidation_confidence",
            "lung_opacity_state",
            "lung_opacity_confidence",
            "evidence_coverage",
            "text_uncertainty",
            "semantic_conflict",
            "numerical_discordance",
            "evidence_interpretation",
            "aarf_v3_probability",
            "defer_reason",
            "decision",
        ]
    ].to_csv(
        CONFLICT_PATH,
        index=False
    )

    calibration_df.to_csv(
        CALIBRATION_PATH,
        index=False
    )

    # --------------------------------------------------------
    # Console report
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("VALIDATION RESULTS")
    print("=" * 70)

    print(
        metrics_df[
            [
                "method",
                "accuracy",
                "precision",
                "recall",
                "f1",
                "roc_auc",
                "brier_score",
                "ece",
            ]
        ].to_string(
            index=False
        )
    )

    print("\n" + "=" * 70)
    print("EVIDENCE PROFILE")
    print("=" * 70)

    print(
        "\nEvidence interpretation:"
    )

    print(
        validation_profile[
            "evidence_interpretation"
        ]
        .value_counts()
        .to_string()
    )

    print(
        "\nSemantic conflicts:",
        int(
            (
                validation_profile[
                    "semantic_conflict"
                ] != "NONE"
            ).sum()
        )
    )

    print(
        "Moderate numerical discordance:",
        int(
            (
                validation_profile[
                    "numerical_discordance"
                ] >= 0.30
            ).sum()
        )
    )

    print(
        "Deferred cases:",
        len(deferred)
    )

    print(
        "Coverage:",
        round(
            len(non_deferred) /
            len(validation_profile),
            4
        )
        if len(validation_profile)
        else 0
    )

    print("\n" + "=" * 70)
    print("OUTPUT FILES")
    print("=" * 70)

    print(
        FUSION_RESULTS_PATH
    )

    print(
        METRICS_PATH
    )

    print(
        CONFLICT_PATH
    )

    print(
        CALIBRATION_PATH
    )

    print("\nDone.")


if __name__ == "__main__":
    main()