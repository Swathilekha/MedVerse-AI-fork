import os
import pandas as pd


# Project root:
# C:\Users\prana\Desktop\medassistant\MedVerse-AI-Starter
PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "..")
)

DATA_DIR = os.path.join(PROJECT_ROOT, "data", "clinical_text")

XRAY_PREDICTIONS_FILE = os.path.join(
    DATA_DIR, "xray_predictions.csv"
)

FUSION_RESULTS_FILE = os.path.join(
    DATA_DIR, "multimodal_fusion_results.csv"
)

AI_EVIDENCE_FILE = os.path.join(
    DATA_DIR, "ai_evidence.csv"
)


def _load_csv(path: str):
    """Load a CSV safely."""
    if not os.path.exists(path):
        return None

    try:
        return pd.read_csv(path)
    except Exception:
        return None


def get_fusion_result(patient_id: str):
    """
    Return the AARF v3 multimodal fusion result
    for a patient.
    """

    df = _load_csv(FUSION_RESULTS_FILE)

    if df is None:
        return None

    if "patient_id" not in df.columns:
        return None

    patient_id = str(patient_id)

    result = df[
        df["patient_id"].astype(str) == patient_id
    ]

    if result.empty:
        return None

    # Use the first matching record.
    row = result.iloc[0]

    return row.where(pd.notna(row), None).to_dict()


def get_xray_result(patient_id: str = None, image_name: str = None):
    """
    Return X-ray prediction information.

    Supports lookup by patient_id or image name.
    """

    df = _load_csv(XRAY_PREDICTIONS_FILE)

    if df is None:
        return None

    # Patient lookup if available.
    if patient_id and "patient_id" in df.columns:

        result = df[
            df["patient_id"].astype(str) == str(patient_id)
        ]

        if not result.empty:
            row = result.iloc[0]
            return row.where(pd.notna(row), None).to_dict()

    # Image lookup.
    if image_name:

        possible_columns = [
            "local_path",
            "png_path",
            "image_name",
            "image"
        ]

        for column in possible_columns:

            if column in df.columns:

                mask = df[column].astype(str).str.contains(
                    str(image_name),
                    case=False,
                    na=False
                )

                result = df[mask]

                if not result.empty:
                    row = result.iloc[0]
                    return row.where(pd.notna(row), None).to_dict()

    return None


def get_patient_evidence(patient_id: str):
    """
    Return clinical evidence extracted from the
    patient's report.
    """

    df = _load_csv(AI_EVIDENCE_FILE)

    if df is None:
        return []

    if "patient_id" not in df.columns:
        return []

    result = df[
        df["patient_id"].astype(str) == str(patient_id)
    ]

    if result.empty:
        return []

    result = result.copy()

    # Convert NaN values to None.
    result = result.where(pd.notna(result), None)

    return result.to_dict(orient="records")


def get_data_status():
    """
    Useful for debugging the AI data pipeline.
    """

    return {
        "project_root": PROJECT_ROOT,
        "data_dir": DATA_DIR,
        "xray_predictions_exists": os.path.exists(
            XRAY_PREDICTIONS_FILE
        ),
        "fusion_results_exists": os.path.exists(
            FUSION_RESULTS_FILE
        ),
        "ai_evidence_exists": os.path.exists(
            AI_EVIDENCE_FILE
        ),
        "fusion_results_file": FUSION_RESULTS_FILE,
    }