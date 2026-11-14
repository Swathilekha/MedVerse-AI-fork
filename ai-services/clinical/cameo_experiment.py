import os
import sys
import numpy as np
import pandas as pd

from sklearn.linear_model import LogisticRegression


# ============================================================
# PROJECT PATH
# ============================================================

BASE_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

sys.path.insert(0, os.path.dirname(__file__))

import fusion_engine as fe


# ============================================================
# PATHS
# ============================================================

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

OUTPUT_PATH = os.path.join(
    OUTPUT_DIR,
    "cameo_results.csv",
)

SUMMARY_PATH = os.path.join(
    OUTPUT_DIR,
    "cameo_summary.csv",
)


# ============================================================
# CAMEO PARAMETERS
# ============================================================

ASSERTION_STATES = [
    "PRESENT",
    "UNCERTAIN",
    "ABSENT",
]

TARGET_FINDING = "pneumonia"


# ============================================================
# LOAD DATA
# ============================================================

def load_data():

    print("=" * 70)
    print(
        "CAMEO - CLINICAL ASSERTION MODULATION "
        "AND EVIDENCE OBSERVATION"
    )
    print("=" * 70)

    print("\nLoading datasets...")

    xray = pd.read_csv(XRAY_PATH)
    evidence = pd.read_csv(EVIDENCE_PATH)
    dataset = pd.read_csv(DATASET_PATH)

    print(f"X-ray records     : {len(xray)}")
    print(f"Evidence records  : {len(evidence)}")
    print(f"Dataset records   : {len(dataset)}")

    # --------------------------------------------------------
    # Normalize patient IDs
    # --------------------------------------------------------

    xray["patient_id"] = (
        xray["patient_id"]
        .apply(fe.normalize_patient_id)
    )

    evidence["patient_id"] = (
        evidence["patient_id"]
        .apply(fe.normalize_patient_id)
    )

    dataset["patient_id"] = (
        dataset["patient_id"]
        .apply(fe.normalize_patient_id)
    )

    # --------------------------------------------------------
    # X-ray probability
    # --------------------------------------------------------

    probability_column = None

    for candidate in [
        "pneumonia_probability",
        "probability",
        "xray_probability",
    ]:
        if candidate in xray.columns:
            probability_column = candidate
            break

    if probability_column is None:
        raise ValueError(
            "Could not identify pneumonia probability column."
        )

    xray["xray_probability"] = (
        pd.to_numeric(
            xray[probability_column],
            errors="coerce",
        )
        .fillna(0.5)
        .clip(0, 1)
    )

    # --------------------------------------------------------
    # Ground truth
    # --------------------------------------------------------

    dataset["y_true"] = (
        dataset["class_name"]
        .astype(str)
        .str.upper()
        .map({
            "NORMAL": 0,
            "PNEUMONIA": 1,
        })
    )

    if dataset["y_true"].isna().any():
        raise ValueError(
            "Unknown class_name found in dataset."
        )

    # --------------------------------------------------------
    # Align X-ray and dataset
    # --------------------------------------------------------

    if "local_path" in xray.columns:

        merged = xray.merge(
            dataset[
                [
                    "local_path",
                    "patient_id",
                    "y_true",
                    "final_split",
                ]
            ],
            on="local_path",
            how="inner",
            suffixes=("", "_dataset"),
        )

    else:

        merged = xray.merge(
            dataset[
                [
                    "patient_id",
                    "y_true",
                    "final_split",
                ]
            ].drop_duplicates(
                subset=["patient_id"]
            ),
            on="patient_id",
            how="inner",
        )

    # Prefer dataset patient ID if merge produced both
    if "patient_id_dataset" in merged.columns:

        merged["patient_id"] = (
            merged["patient_id_dataset"]
        )

        merged.drop(
            columns=["patient_id_dataset"],
            inplace=True,
        )

    # --------------------------------------------------------
    # Create unique case identifier
    #
    # CAMEO is evaluated per validation IMAGE/CASE,
    # not per unique patient.
    # --------------------------------------------------------

    if "local_path" in merged.columns:

        merged["case_id"] = (
            merged["local_path"]
            .astype(str)
        )

    else:

        merged["case_id"] = (
            merged["patient_id"].astype(str)
            + "_"
            + merged.index.astype(str)
        )

    print(
        f"Aligned records   : {len(merged)}"
    )

    return merged, evidence


# ============================================================
# TRAIN ORIGINAL AARF
# ============================================================

def train_original_aarf(
    merged,
    evidence,
):

    print(
        "\nBuilding original evidence profile..."
    )

    # Exact existing AARF evidence construction
    profile = fe.build_evidence_profile(
        merged,
        evidence,
    )

    # Visual state
    profile["visual_state"] = (
        profile["xray_probability"]
        .apply(fe.classify_visual_state)
    )

    # Semantic conflict
    profile["semantic_conflict"] = (
        profile.apply(
            fe.detect_semantic_conflict,
            axis=1,
        )
    )

    # Numerical discordance
    profile["numerical_discordance"] = (
        profile.apply(
            fe.calculate_discordance,
            axis=1,
        )
    )

    # Exact AARF v3 feature construction
    meta_features = fe.create_meta_features(
        profile,
        mode="full",
    )

    # Normalize split
    profile["final_split"] = (
        profile["final_split"]
        .astype(str)
        .str.lower()
        .str.strip()
    )

    train_mask = profile[
        "final_split"
    ].isin(
        ["train", "training"]
    )

    val_mask = profile[
        "final_split"
    ].isin(
        ["val", "validation"]
    )

    train = profile.loc[
        train_mask
    ].copy()

    validation = profile.loc[
        val_mask
    ].copy()

    print(
        f"Training records  : {len(train)}"
    )

    print(
        f"Validation records: {len(validation)}"
    )

    # --------------------------------------------------------
    # Patient leakage check
    # --------------------------------------------------------

    overlap = (
        set(train["patient_id"])
        &
        set(validation["patient_id"])
    )

    print(
        f"Patient overlap   : {len(overlap)}"
    )

    if overlap:
        raise RuntimeError(
            "Patient leakage detected."
        )

    # --------------------------------------------------------
    # Train AARF
    # --------------------------------------------------------

    X_train = meta_features.loc[
        train.index
    ]

    y_train = (
        train["y_true"]
        .astype(int)
    )

    model = LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=42,
    )

    model.fit(
        X_train,
        y_train,
    )

    print(
        "\nOriginal AARF model trained."
    )

    return (
        model,
        validation,
    )


# ============================================================
# CREATE CAMEO INTERVENTION
# ============================================================

def create_intervention_evidence(
    evidence,
    patient_id,
    assertion_state,
):

    # Only this patient's evidence
    patient_evidence = evidence.loc[
        evidence["patient_id"] == patient_id
    ].copy()

    # Remove existing pneumonia assertions
    pneumonia_mask = (
        patient_evidence["finding"]
        .astype(str)
        .str.lower()
        .str.strip()
        == TARGET_FINDING
    )

    patient_evidence = patient_evidence.loc[
        ~pneumonia_mask
    ].copy()

    # Add controlled assertion
    controlled_row = {
        "patient_id": patient_id,
        "finding": TARGET_FINDING,
        "status": assertion_state.lower(),

        "evidence_text": (
            "CAMEO controlled assertion: "
            f"pneumonia {assertion_state.lower()}"
        ),

        "matched_text": TARGET_FINDING,
        "section": "CAMEO",
        "anatomical_location": "",
        "alternative_findings": "",

        "model_predicted_status":
            assertion_state.lower(),

        "model_confidence": 1.0,
        "extraction_confidence": 1.0,

        "assertion_source":
            "cameo_intervention",

        "context": (
            f"Controlled {assertion_state.lower()} "
            "assertion intervention"
        ),
    }

    patient_evidence = pd.concat(
        [
            patient_evidence,
            pd.DataFrame([controlled_row]),
        ],
        ignore_index=True,
    )

    return patient_evidence


# ============================================================
# RUN ONE CAMEO INTERVENTION
# ============================================================

def run_intervention(
    validation_row,
    intervention_evidence,
    model,
):

    # One X-ray/case
    xray_row = pd.DataFrame(
        [validation_row]
    )

    # Same AARF evidence profile
    profile = fe.build_evidence_profile(
        xray_row,
        intervention_evidence,
    )

    # Same visual state
    profile["visual_state"] = (
        profile["xray_probability"]
        .apply(fe.classify_visual_state)
    )

    # Same semantic conflict
    profile["semantic_conflict"] = (
        profile.apply(
            fe.detect_semantic_conflict,
            axis=1,
        )
    )

    # Same discordance
    profile["numerical_discordance"] = (
        profile.apply(
            fe.calculate_discordance,
            axis=1,
        )
    )

    # Same AARF features
    features = fe.create_meta_features(
        profile,
        mode="full",
    )

    # Same trained AARF model
    probability = float(
        model.predict_proba(
            features
        )[:, 1][0]
    )

    return (
        profile.iloc[0],
        probability,
    )


# ============================================================
# MAIN CAMEO EXPERIMENT
# ============================================================

def main():

    os.makedirs(
        OUTPUT_DIR,
        exist_ok=True,
    )

    # --------------------------------------------------------
    # Load
    # --------------------------------------------------------

    merged, evidence = load_data()

    # --------------------------------------------------------
    # Train AARF
    # --------------------------------------------------------

    model, validation = (
        train_original_aarf(
            merged,
            evidence,
        )
    )

    # --------------------------------------------------------
    # Find patients having pneumonia evidence
    # --------------------------------------------------------

    pneumonia_patients = set(
        evidence.loc[
            evidence["finding"]
            .astype(str)
            .str.lower()
            .str.strip()
            == TARGET_FINDING,
            "patient_id",
        ]
    )

    validation_cases = validation[
        validation["patient_id"].isin(
            pneumonia_patients
        )
    ].copy()

    print(
        "\nValidation cases with pneumonia evidence: "
        f"{len(validation_cases)}"
    )

    if len(validation_cases) == 0:
        raise RuntimeError(
            "No validation cases with pneumonia evidence."
        )

    # --------------------------------------------------------
    # Ensure unique case IDs
    # --------------------------------------------------------

    if "case_id" not in validation_cases.columns:

        if "local_path" in validation_cases.columns:

            validation_cases["case_id"] = (
                validation_cases["local_path"]
                .astype(str)
            )

        else:

            validation_cases["case_id"] = (
                validation_cases["patient_id"]
                .astype(str)
                + "_"
                + validation_cases.index.astype(str)
            )

    duplicate_cases = (
        validation_cases["case_id"]
        .duplicated()
        .sum()
    )

    if duplicate_cases > 0:

        raise RuntimeError(
            f"Duplicate validation case IDs detected: "
            f"{duplicate_cases}"
        )

    # --------------------------------------------------------
    # CAMEO results
    # --------------------------------------------------------

    results = []

    total_cases = len(
        validation_cases
    )

    # --------------------------------------------------------
    # Each case gets EXACTLY:
    #
    # PRESENT
    # UNCERTAIN
    # ABSENT
    #
    # 170 × 3 = 510
    # --------------------------------------------------------

    for count, (_, row) in enumerate(
        validation_cases.iterrows(),
        start=1,
    ):

        case_id = row["case_id"]
        patient_id = row["patient_id"]

        if (
            count == 1
            or count % 10 == 0
            or count == total_cases
        ):

            print(
                f"Processed "
                f"{count}/{total_cases}"
            )

        for assertion_state in ASSERTION_STATES:

            # Controlled evidence
            intervention_evidence = (
                create_intervention_evidence(
                    evidence,
                    patient_id,
                    assertion_state,
                )
            )

            # Run AARF
            profile_row, probability = (
                run_intervention(
                    row,
                    intervention_evidence,
                    model,
                )
            )

            # Save result
            results.append({

                "case_id": case_id,

                "patient_id": patient_id,

                "local_path": (
                    row["local_path"]
                    if "local_path" in row.index
                    else ""
                ),

                "true_class": (
                    "PNEUMONIA"
                    if int(row["y_true"]) == 1
                    else "NORMAL"
                ),

                "xray_probability": float(
                    row["xray_probability"]
                ),

                "assertion_state":
                    assertion_state,

                "fusion_probability":
                    probability,

                "fusion_prediction": (
                    "PNEUMONIA"
                    if probability >= 0.5
                    else "NO_PNEUMONIA"
                ),

                "pneumonia_support": float(
                    profile_row[
                        "pneumonia_support"
                    ]
                ),

                "pneumonia_confidence": float(
                    profile_row[
                        "pneumonia_confidence"
                    ]
                ),

                "consolidation_state":
                    profile_row[
                        "consolidation_state"
                    ],

                "lung_opacity_state":
                    profile_row[
                        "lung_opacity_state"
                    ],

                "evidence_coverage": float(
                    profile_row[
                        "evidence_coverage"
                    ]
                ),
            })

    # --------------------------------------------------------
    # Results dataframe
    # --------------------------------------------------------

    results_df = pd.DataFrame(
        results
    )

    # --------------------------------------------------------
    # Sanity checks
    # --------------------------------------------------------

    print(
        "\nCAMEO assertion counts:"
    )

    print(
        results_df[
            "assertion_state"
        ]
        .value_counts()
        .sort_index()
        .to_string()
    )

    expected_rows = (
        total_cases
        *
        len(ASSERTION_STATES)
    )

    print(
        f"\nExpected intervention rows: "
        f"{expected_rows}"
    )

    print(
        f"Actual intervention rows  : "
        f"{len(results_df)}"
    )

    if len(results_df) != expected_rows:

        raise RuntimeError(
            "CAMEO intervention count mismatch."
        )

    # Each case must have exactly one of each state
    state_counts = (
        results_df
        .groupby(
            [
                "case_id",
                "assertion_state",
            ]
        )
        .size()
    )

    if (
        state_counts
        .max()
        != 1
    ):

        raise RuntimeError(
            "Duplicate case/assertion combinations detected."
        )

    # ========================================================
    # CAMEO METRICS
    # ========================================================

    # IMPORTANT:
    # Pivot by CASE, not patient.
    #
    # A patient can have multiple X-ray records.
    # ========================================================

    pivot = (
        results_df
        .pivot(
            index="case_id",
            columns="assertion_state",
            values="fusion_probability",
        )
        .reset_index()
    )

    pivot.columns.name = None

    # --------------------------------------------------------
    # Required columns
    # --------------------------------------------------------

    required_states = [
        "PRESENT",
        "UNCERTAIN",
        "ABSENT",
    ]

    missing = [
        state
        for state in required_states
        if state not in pivot.columns
    ]

    if missing:

        print(
            "\nERROR: Missing intervention states:"
        )

        print(missing)

        print(
            "\nActual pivot columns:"
        )

        print(
            list(pivot.columns)
        )

        raise RuntimeError(
            "CAMEO pivot failed."
        )

    # --------------------------------------------------------
    # 1. Assertion ordering
    #
    # PRESENT >= UNCERTAIN >= ABSENT
    # --------------------------------------------------------

    pivot["ordering_satisfied"] = (
        (pivot["PRESENT"] >= pivot["UNCERTAIN"])
        &
        (pivot["UNCERTAIN"] >= pivot["ABSENT"])
    )

    # --------------------------------------------------------
    # 2. Directional response
    #
    # PRESENT > ABSENT
    # --------------------------------------------------------

    pivot["directional_response"] = (
        pivot["PRESENT"]
        >
        pivot["ABSENT"]
    )

    # --------------------------------------------------------
    # 3. Response magnitude
    #
    # PRESENT - ABSENT
    # --------------------------------------------------------

    pivot["response_magnitude"] = (
        pivot["PRESENT"]
        -
        pivot["ABSENT"]
    )

    # --------------------------------------------------------
    # 4. Uncertain-between
    #
    # ABSENT <= UNCERTAIN <= PRESENT
    # --------------------------------------------------------

    pivot["uncertain_between"] = (
        (pivot["UNCERTAIN"] <= pivot["PRESENT"])
        &
        (pivot["UNCERTAIN"] >= pivot["ABSENT"])
    )

    # --------------------------------------------------------
    # Summary statistics
    # --------------------------------------------------------

    ordering_rate = (
        pivot["ordering_satisfied"]
        .mean()
    )

    directional_rate = (
        pivot["directional_response"]
        .mean()
    )

    uncertain_between_rate = (
        pivot["uncertain_between"]
        .mean()
    )

    mean_response = (
        pivot["response_magnitude"]
        .mean()
    )

    median_response = (
        pivot["response_magnitude"]
        .median()
    )

    # --------------------------------------------------------
    # Add metrics back to individual results
    # --------------------------------------------------------

    results_df = results_df.merge(
        pivot[
            [
                "case_id",
                "ordering_satisfied",
                "directional_response",
                "response_magnitude",
                "uncertain_between",
            ]
        ],
        on="case_id",
        how="left",
    )

    # ========================================================
    # SUMMARY FILE
    # ========================================================

    summary = pd.DataFrame({

        "metric": [
            "validation_cases",
            "assertion_ordering_rate",
            "directional_response_rate",
            "uncertain_between_rate",
            "mean_present_absent_response",
            "median_present_absent_response",
        ],

        "value": [
            len(pivot),
            ordering_rate,
            directional_rate,
            uncertain_between_rate,
            mean_response,
            median_response,
        ],
    })

    # ========================================================
    # SAVE
    # ========================================================

    results_df.to_csv(
        OUTPUT_PATH,
        index=False,
    )

    summary.to_csv(
        SUMMARY_PATH,
        index=False,
    )

    # ========================================================
    # FINAL OUTPUT
    # ========================================================

    print(
        "\n" + "=" * 70
    )

    print(
        "CAMEO RESULTS"
    )

    print(
        "=" * 70
    )

    print(
        f"\nCases evaluated             : "
        f"{len(pivot)}"
    )

    print(
        f"Assertion ordering rate    : "
        f"{ordering_rate:.4f}"
    )

    print(
        f"Directional response rate  : "
        f"{directional_rate:.4f}"
    )

    print(
        f"Uncertain-between rate     : "
        f"{uncertain_between_rate:.4f}"
    )

    print(
        f"Mean response magnitude    : "
        f"{mean_response:.4f}"
    )

    print(
        f"Median response magnitude  : "
        f"{median_response:.4f}"
    )

    print(
        "\nSaved:"
    )

    print(
        OUTPUT_PATH
    )

    print(
        SUMMARY_PATH
    )

    # --------------------------------------------------------
    # Example
    # --------------------------------------------------------

    print(
        "\nExample:"
    )

    print(
        results_df[
            [
                "case_id",
                "patient_id",
                "assertion_state",
                "xray_probability",
                "fusion_probability",
                "pneumonia_support",
            ]
        ]
        .head(12)
        .to_string(index=False)
    )

    print(
        "\nDone."
    )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()