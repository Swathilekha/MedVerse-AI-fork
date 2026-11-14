
import os
import glob
import re
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix

# ============================================================
# PATHS
# ============================================================

ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)

COMMON_DIR = os.path.join(
    ROOT, "ai-services", "clinical", "outputs", "common_723"
)

OUTPUT_DIR = os.path.join(ROOT, "figures")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ============================================================
# MODEL FILES
# ============================================================

PREDICTION_FILES = {
    "X-ray Only": "x_ray_only_common_predictions.csv",
    "Text Only": "text_only_common_predictions.csv",
    "History Only": "history_only_common_predictions.csv",
    "Early Fusion Best": "early_fusion_best_common_predictions.csv",
    "Early Fusion Continued": "early_fusion_continued_common_predictions.csv",
    "Joint Fusion": "joint_fusion_common_predictions.csv",
}

# ============================================================
# HELPER FUNCTIONS
# ============================================================

def find_file(filename):
    matches = glob.glob(
        os.path.join(COMMON_DIR, "**", filename),
        recursive=True
    )

    if not matches:
        raise FileNotFoundError(
            f"Could not find {filename} under {COMMON_DIR}"
        )

    return matches[0]


def normalize_name(value):
    return re.sub(r"[^a-z0-9]", "", str(value).lower())


def find_column(df, possible_names):
    normalized = {
        normalize_name(col): col for col in df.columns
    }

    for name in possible_names:
        key = normalize_name(name)
        if key in normalized:
            return normalized[key]

    return None


def normalize_metric(series):
    values = pd.to_numeric(series, errors="coerce")

    # Convert percentage values to fractions.
    if values.max(skipna=True) > 1:
        values = values / 100.0

    return values


# ============================================================
# FIGURE 1: SIX MODEL COMPARISON
# Accuracy, Precision, Recall, F1, Brier, ECE
# ============================================================

print("\nGenerating six_model_comparison.png ...")

metrics_path = find_file("model_metrics_and_calibration.csv")
metrics_df = pd.read_csv(metrics_path)

model_col = find_column(
    metrics_df, ["model", "model_name", "classifier"]
)

if model_col is None:
    raise ValueError(
        "Cannot identify model column in model_metrics_and_calibration.csv. "
        f"Columns found: {list(metrics_df.columns)}"
    )

metric_aliases = {
    "Accuracy": ["accuracy", "acc"],
    "Precision": ["precision"],
    "Recall": ["recall", "sensitivity"],
    "F1-score": ["f1", "f1_score", "f1score"],
    "Brier score": ["brier", "brier_score"],
    "ECE": ["ece", "expected_calibration_error"],
}

metric_columns = {}

for display_name, aliases in metric_aliases.items():
    col = find_column(metrics_df, aliases)

    if col is not None:
        metric_columns[display_name] = col

missing = set(metric_aliases) - set(metric_columns)

if missing:
    raise ValueError(
        "Missing metrics in model_metrics_and_calibration.csv: "
        f"{missing}\nAvailable columns: {list(metrics_df.columns)}"
    )


def match_model(name):
    name = normalize_name(name)

    aliases = {
        "xrayonly": ["xrayonly", "xray", "densenet121"],
        "textonly": ["textonly", "text", "nlp"],
        "historyonly": ["historyonly", "history"],
        "earlyfusionbest": ["earlyfusionbest"],
        "earlyfusioncontinued": ["earlyfusioncontinued"],
        "jointfusion": ["jointfusion"],
    }

    for standard, variants in aliases.items():
        if name in variants or any(v in name for v in variants):
            return standard

    return name


model_order = [
    "xrayonly",
    "textonly",
    "historyonly",
    "earlyfusionbest",
    "earlyfusioncontinued",
    "jointfusion",
]

display_names = [
    "X-ray Only",
    "Text Only",
    "History Only",
    "Early Fusion Best",
    "Early Fusion Continued",
    "Joint Fusion",
]

metrics_df["_model_key"] = metrics_df[model_col].apply(match_model)

metrics_df = (
    metrics_df.drop_duplicates("_model_key")
    .set_index("_model_key")
)

missing_models = [
    m for m in model_order if m not in metrics_df.index
]

if missing_models:
    raise ValueError(
        f"Missing models in metrics CSV: {missing_models}\n"
        f"Model names found: {metrics_df[model_col].tolist()}"
    )

fig, axes = plt.subplots(1, 2, figsize=(17, 6))

classification_metrics = [
    "Accuracy", "Precision", "Recall", "F1-score"
]

calibration_metrics = [
    "Brier score", "ECE"
]

x = np.arange(len(display_names))
width = 0.18

for i, metric in enumerate(classification_metrics):
    values = normalize_metric(
        metrics_df.loc[model_order, metric_columns[metric]]
    )

    axes[0].bar(
        x + (i - 1.5) * width,
        values,
        width,
        label=metric
    )

axes[0].set_title("Classification Performance")
axes[0].set_ylabel("Score")
axes[0].set_xticks(x)
axes[0].set_xticklabels(display_names, rotation=25, ha="right")
axes[0].set_ylim(0, 1.05)
axes[0].legend()
axes[0].grid(axis="y", alpha=0.3)

width = 0.32

for i, metric in enumerate(calibration_metrics):
    values = normalize_metric(
        metrics_df.loc[model_order, metric_columns[metric]]
    )

    axes[1].bar(
        x + (i - 0.5) * width,
        values,
        width,
        label=metric
    )

axes[1].set_title("Calibration Performance")
axes[1].set_ylabel("Score (lower is better)")
axes[1].set_xticks(x)
axes[1].set_xticklabels(display_names, rotation=25, ha="right")
axes[1].legend()
axes[1].grid(axis="y", alpha=0.3)

fig.suptitle("Six-Model Performance and Calibration Comparison")
fig.tight_layout()

fig.savefig(
    os.path.join(OUTPUT_DIR, "six_model_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Saved six_model_comparison.png")


# ============================================================
# FIGURE 2: SIX CONFUSION MATRICES
# ============================================================

print("\nGenerating confusion_matrices.png ...")

fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.flatten()

for index, (model_name, filename) in enumerate(
    PREDICTION_FILES.items()
):

    path = find_file(filename)
    df = pd.read_csv(path)

    actual_col = find_column(
        df, ["actual_label", "label", "true_label", "y_true"]
    )

    predicted_col = find_column(
        df, ["predicted_label", "prediction", "pred_label", "y_pred"]
    )

    if actual_col is None or predicted_col is None:
        raise ValueError(
            f"Missing actual/predicted label columns in {filename}. "
            f"Columns found: {list(df.columns)}"
        )

    y_true = pd.to_numeric(df[actual_col], errors="coerce")
    y_pred = pd.to_numeric(df[predicted_col], errors="coerce")

    valid = y_true.notna() & y_pred.notna()
    y_true = y_true[valid].astype(int)
    y_pred = y_pred[valid].astype(int)

    cm = confusion_matrix(
        y_true, y_pred, labels=[0, 1]
    )

    ax = axes[index]

    image = ax.imshow(cm, interpolation="nearest")

    ax.set_title(model_name)
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("Actual label")

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Normal (0)", "Pneumonia (1)"])
    ax.set_yticklabels(["Normal (0)", "Pneumonia (1)"])

    for row in range(2):
        for col in range(2):
            ax.text(
                col, row,
                str(cm[row, col]),
                ha="center",
                va="center",
                fontsize=13
            )

    print(f"{model_name}:")
    print(f"TN={cm[0,0]}, FP={cm[0,1]}, "
          f"FN={cm[1,0]}, TP={cm[1,1]}")

fig.colorbar(image, ax=axes.tolist(), shrink=0.75)
fig.suptitle("Confusion Matrices Across Six Models", fontsize=16)
fig.tight_layout()

fig.savefig(
    os.path.join(OUTPUT_DIR, "confusion_matrices.png"),
    dpi=300,
    bbox_inches="tight"
)
plt.close(fig)

print("Saved confusion_matrices.png")


# ============================================================
# FIGURE 3: BOOTSTRAP CONFIDENCE INTERVALS
# F1-score and ROC-AUC
# ============================================================

print("\nGenerating bootstrap_comparison.png ...")

bootstrap_path = find_file(
    "cluster_bootstrap_confidence_intervals.csv"
)

boot_df = pd.read_csv(bootstrap_path)

print("Bootstrap CSV columns:", list(boot_df.columns))

# Use the exact column names from your CSV
boot_model_col = "Model"
boot_metric_col = "Metric"
estimate_col = "Estimate"
lower_col = "CI_95_Lower"
upper_col = "CI_95_Upper"

required_columns = [
    boot_model_col,
    boot_metric_col,
    estimate_col,
    lower_col,
    upper_col
]

missing_columns = [
    col for col in required_columns
    if col not in boot_df.columns
]

if missing_columns:
    raise ValueError(
        f"Missing bootstrap columns: {missing_columns}\n"
        f"Available columns: {list(boot_df.columns)}"
    )

# Convert numeric columns
for col in [estimate_col, lower_col, upper_col]:
    boot_df[col] = pd.to_numeric(
        boot_df[col], errors="coerce"
    )

# Create normalized metric names
boot_df["_metric_key"] = (
    boot_df[boot_metric_col]
    .astype(str)
    .apply(normalize_name)
)

# F1 and ROC-AUC aliases
metrics_to_plot = {
    "F1-score": ["f1", "f1score"],
    "ROC-AUC": ["rocauc", "roc_auc"]
}

fig, axes = plt.subplots(1, 2, figsize=(15, 7))

for ax, (display_metric, aliases) in zip(
    axes, metrics_to_plot.items()
):

    selected = boot_df[
        boot_df["_metric_key"].isin(
            [normalize_name(a) for a in aliases]
        )
    ].copy()

    if selected.empty:
        raise ValueError(
            f"No bootstrap rows found for {display_metric}.\n"
            f"Available metrics: "
            f"{boot_df[boot_metric_col].unique()}"
        )

    # Match the six model names
    selected["_model_key"] = (
        selected[boot_model_col].apply(match_model)
    )

    # Keep only the six models in the required order
    selected = selected[
        selected["_model_key"].isin(model_order)
    ]

    selected = (
        selected.drop_duplicates("_model_key")
        .set_index("_model_key")
    )

    available = [
        model for model in model_order
        if model in selected.index
    ]

    if not available:
        raise ValueError(
            f"No recognized models found for {display_metric}."
        )

    # Extract estimate and confidence interval values
    estimates = selected.loc[
        available, estimate_col
    ].to_numpy(dtype=float)

    lows = selected.loc[
        available, lower_col
    ].to_numpy(dtype=float)

    highs = selected.loc[
        available, upper_col
    ].to_numpy(dtype=float)

    # Remove invalid rows
    valid = (
        np.isfinite(estimates)
        & np.isfinite(lows)
        & np.isfinite(highs)
    )

    estimates = estimates[valid]
    lows = lows[valid]
    highs = highs[valid]

    available = [
        model for model, keep in zip(available, valid)
        if keep
    ]

    if len(available) == 0:
        raise ValueError(
            f"No valid confidence intervals for {display_metric}."
        )

    # Convert percentages to fractions if necessary
    if np.nanmax(highs) > 1:
        estimates = estimates / 100.0
        lows = lows / 100.0
        highs = highs / 100.0

    # Calculate asymmetric error bars around the estimate
    lower_errors = estimates - lows
    upper_errors = highs - estimates

    # Guard against inconsistent estimate/CI values
    if np.any(lower_errors < -1e-8) or np.any(upper_errors < -1e-8):
        raise ValueError(
            f"Some estimates fall outside their confidence intervals "
            f"for {display_metric}. Check the bootstrap CSV."
        )

    errors = np.vstack([
        np.maximum(lower_errors, 0),
        np.maximum(upper_errors, 0)
    ])

    labels = [
        display_names[model_order.index(model)]
        for model in available
    ]

    positions = np.arange(len(available))

    ax.errorbar(
        estimates,
        positions,
        xerr=errors,
        fmt="o",
        capsize=5,
        markersize=7
    )

    ax.set_yticks(positions)
    ax.set_yticklabels(labels)

    ax.set_xlabel(display_metric)
    ax.set_title(
        f"{display_metric} – 95% Bootstrap Confidence Intervals"
    )

    ax.grid(axis="x", alpha=0.3)
    ax.invert_yaxis()

fig.suptitle(
    "Bootstrap 95% Confidence Intervals Across Six Models",
    fontsize=15
)

fig.tight_layout()

fig.savefig(
    os.path.join(OUTPUT_DIR, "bootstrap_comparison.png"),
    dpi=300,
    bbox_inches="tight"
)

plt.close(fig)

print("Saved bootstrap_comparison.png")

print("\n========================================")
print("ALL THREE FIGURES GENERATED SUCCESSFULLY")
print("Output folder:", OUTPUT_DIR)
print("========================================")