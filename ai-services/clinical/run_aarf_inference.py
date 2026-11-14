import os
import json
import joblib
import numpy as np
import pandas as pd


# ============================================================
# PATHS
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

MODEL_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clinical_text",
    "aarf_v3_model.joblib",
)

FEATURES_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clinical_text",
    "aarf_v3_features.json",
)

FUSION_RESULTS_PATH = os.path.join(
    BASE_DIR,
    "data",
    "clinical_text",
    "multimodal_fusion_results.csv",
)


# ============================================================
# LOAD SAVED MODEL
# ============================================================

def load_model():

    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"AARF model not found:\n{MODEL_PATH}"
        )

    if not os.path.exists(FEATURES_PATH):
        raise FileNotFoundError(
            f"AARF feature schema not found:\n{FEATURES_PATH}"
        )

    model = joblib.load(MODEL_PATH)

    with open(
        FEATURES_PATH,
        "r",
        encoding="utf-8"
    ) as f:
        feature_names = json.load(f)

    return model, feature_names


# ============================================================
# INFERENCE
# ============================================================

def run_inference(row, model, feature_names):
    """
    Reconstruct the EXACT AARF V3 meta-features used during training.
    """

    def get_float(name, default=0.0):
        if name not in row.index:
            return default

        try:
            value = float(row[name])
        except (TypeError, ValueError):
            return default

        if not np.isfinite(value):
            return default

        return value

    # =========================================================
    # 1. X-RAY FEATURES
    # EXACTLY matches entropy_binary() in fusion_engine.py
    # =========================================================

    xray_probability = get_float("xray_probability")

    p = np.clip(
        xray_probability,
        1e-8,
        1 - 1e-8
    )

    xray_uncertainty = -(
        p * np.log2(p)
        +
        (1 - p) * np.log2(1 - p)
    )

    # =========================================================
    # 2. DIRECT PNEUMONIA FEATURES
    # =========================================================

    pneumonia_state = str(
        row.get("pneumonia_state", "NO_EVIDENCE")
    ).strip().upper()

    pneumonia_present = float(
        pneumonia_state == "PRESENT"
    )

    pneumonia_absent = float(
        pneumonia_state == "ABSENT"
    )

    pneumonia_uncertain = float(
        pneumonia_state == "UNCERTAIN"
    )

    pneumonia_mixed = float(
        pneumonia_state == "MIXED"
    )

    pneumonia_support = get_float(
        "pneumonia_support"
    )

    pneumonia_confidence = get_float(
        "pneumonia_confidence"
    )

    # =========================================================
    # 3. CONSOLIDATION FEATURES
    # =========================================================

    consolidation_state = str(
        row.get("consolidation_state", "NO_EVIDENCE")
    ).strip().upper()

    consolidation_present = float(
        consolidation_state == "PRESENT"
    )

    consolidation_absent = float(
        consolidation_state == "ABSENT"
    )

    consolidation_uncertain = float(
        consolidation_state == "UNCERTAIN"
    )

    consolidation_confidence = get_float(
        "consolidation_confidence"
    )

    consolidation_support = get_float(
        "consolidation_support"
    )

    # =========================================================
    # 4. LUNG OPACITY FEATURES
    # =========================================================

    lung_opacity_state = str(
        row.get("lung_opacity_state", "NO_EVIDENCE")
    ).strip().upper()

    lung_opacity_present = float(
        lung_opacity_state == "PRESENT"
    )

    lung_opacity_absent = float(
        lung_opacity_state == "ABSENT"
    )

    lung_opacity_uncertain = float(
        lung_opacity_state == "UNCERTAIN"
    )

    lung_opacity_confidence = get_float(
        "lung_opacity_confidence"
    )

    lung_opacity_support = get_float(
        "lung_opacity_support"
    )

    # =========================================================
    # 5. EVIDENCE FEATURES
    # EXACTLY matches create_meta_features()
    # =========================================================

    evidence_coverage = get_float(
        "evidence_coverage"
    )

    # fusion_engine.py uses:
    # features["uncertainty_ratio"] = df["text_uncertainty"]

    uncertainty_ratio = get_float(
        "text_uncertainty"
    )

    # fusion_engine.py uses:
    # df["mixed_count"] / float(len(ALL_FINDINGS))
    #
    # ALL_FINDINGS contains 8 findings.

    mixed_count = get_float(
        "mixed_count"
    )

    mixed_ratio = (
        mixed_count / 8.0
    )

    supporting_present_count = get_float(
        "supporting_present_count"
    )

    supporting_uncertain_count = get_float(
        "supporting_uncertain_count"
    )

    # =========================================================
    # 6. SEMANTIC CONFLICT
    # EXACTLY matches fusion_engine.py
    # =========================================================

    semantic_conflict_value = str(
        row.get("semantic_conflict", "NONE")
    ).strip().upper()

    semantic_conflict = float(
        semantic_conflict_value != "NONE"
    )

    # =========================================================
    # 7. EXACT 24-FEATURE DICTIONARY
    # =========================================================

    feature_values = {
        "xray_probability": xray_probability,
        "xray_uncertainty": xray_uncertainty,

        "pneumonia_present": pneumonia_present,
        "pneumonia_absent": pneumonia_absent,
        "pneumonia_uncertain": pneumonia_uncertain,
        "pneumonia_mixed": pneumonia_mixed,
        "pneumonia_support": pneumonia_support,
        "pneumonia_confidence": pneumonia_confidence,

        "consolidation_present": consolidation_present,
        "consolidation_absent": consolidation_absent,
        "consolidation_uncertain": consolidation_uncertain,
        "consolidation_confidence": consolidation_confidence,
        "consolidation_support": consolidation_support,

        "lung_opacity_present": lung_opacity_present,
        "lung_opacity_absent": lung_opacity_absent,
        "lung_opacity_uncertain": lung_opacity_uncertain,
        "lung_opacity_confidence": lung_opacity_confidence,
        "lung_opacity_support": lung_opacity_support,

        "evidence_coverage": evidence_coverage,
        "uncertainty_ratio": uncertainty_ratio,
        "mixed_ratio": mixed_ratio,
        "supporting_present_count": supporting_present_count,
        "supporting_uncertain_count": supporting_uncertain_count,
        "semantic_conflict": semantic_conflict,
    }

    # =========================================================
    # 8. VERIFY SCHEMA
    # =========================================================

    missing = [
        feature
        for feature in feature_names
        if feature not in feature_values
    ]

    if missing:
        raise ValueError(
            f"Could not reconstruct AARF features: {missing}"
        )

    # =========================================================
    # 9. EXACT FEATURE ORDER FROM SAVED JSON
    # =========================================================

    X = pd.DataFrame(
        [[feature_values[feature] for feature in feature_names]],
        columns=feature_names
    )

    # =========================================================
    # 10. SAVED MODEL INFERENCE
    # =========================================================

    probability = float(
        model.predict_proba(X)[0, 1]
    )

    prediction = int(
        probability >= 0.5
    )

    confidence = max(
        probability,
        1.0 - probability
    )

    decision = (
        "PNEUMONIA_SUPPORT"
        if probability >= 0.5
        else "NO_PNEUMONIA_SUPPORT"
    )

    return {
        "aarf_v3_probability": probability,
        "aarf_v3_prediction": prediction,
        "fusion_confidence": confidence,
        "decision": decision,
    }


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 70)
    print("AARF V3 - SAVED MODEL INFERENCE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load model + schema
    # --------------------------------------------------------

    model, feature_names = load_model()

    print("\nSaved model loaded successfully.")
    print(f"Model: {MODEL_PATH}")

    print("\nFeature schema loaded successfully.")
    print(f"Number of features: {len(feature_names)}")

    print("\nFeatures:")
    for i, feature in enumerate(feature_names, start=1):
        print(f"{i:2d}. {feature}")

    # --------------------------------------------------------
    # Load previously generated fusion results
    # --------------------------------------------------------

    if not os.path.exists(FUSION_RESULTS_PATH):
        raise FileNotFoundError(
            f"Fusion results not found:\n{FUSION_RESULTS_PATH}"
        )

    results = pd.read_csv(
        FUSION_RESULTS_PATH
    )

    print(
        f"\nLoaded fusion result records: {len(results)}"
    )

    # --------------------------------------------------------
    # Select a known validation case
    #
    # Prefer patient11586 if available.
    # --------------------------------------------------------

    row = None

    if "patient_id" in results.columns:

        matches = results[
            results["patient_id"].astype(str)
            == "patient11586"
        ]

        if len(matches) > 0:
            row = matches.iloc[0]

    # Fallback: first evaluated row with AARF probability
    if row is None:

        valid = results[
            results["aarf_v3_probability"].notna()
        ]

        if len(valid) == 0:
            raise RuntimeError(
                "No evaluated AARF records found."
            )

        row = valid.iloc[0]

    # --------------------------------------------------------
    # Run saved-model inference
    # --------------------------------------------------------

    output = run_inference(
        row,
        model,
        feature_names
    )

    # --------------------------------------------------------
    # Display comparison
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("SAVED MODEL INFERENCE")
    print("=" * 70)

    print(
        f"\nPatient ID          : {row.get('patient_id', 'N/A')}"
    )

    print(
        f"X-ray probability   : "
        f"{float(row['xray_probability']):.6f}"
    )

    print(
        f"Pneumonia support   : "
        f"{float(row['pneumonia_support']):.6f}"
    )

    print(
        f"Stored AARF result  : "
        f"{float(row['aarf_v3_probability']):.6f}"
    )

    print(
        f"Saved-model result  : "
        f"{output['aarf_v3_probability']:.6f}"
    )

    print(
        f"Difference          : "
        f"{abs(float(row['aarf_v3_probability']) - output['aarf_v3_probability']):.12f}"
    )

    print(
        f"Prediction          : "
        f"{output['aarf_v3_prediction']}"
    )

    print(
        f"Confidence          : "
        f"{output['fusion_confidence']:.6f}"
    )

    print(
        f"Decision            : "
        f"{output['decision']}"
    )

    # --------------------------------------------------------
    # Verification
    # --------------------------------------------------------

    difference = abs(
        float(row["aarf_v3_probability"])
        - output["aarf_v3_probability"]
    )

    print("\n" + "=" * 70)

    if difference < 1e-10:

        print(
            "VERIFICATION PASSED"
        )

        print(
            "Saved AARF model reproduces the stored "
            "AARF probability."
        )

    elif difference < 1e-6:

        print(
            "VERIFICATION PASSED "
            "(floating-point tolerance)"
        )

    else:

        print(
            "VERIFICATION WARNING"
        )

        print(
            "Saved-model output differs from stored result."
        )

    print("=" * 70)


if __name__ == "__main__":
    main()