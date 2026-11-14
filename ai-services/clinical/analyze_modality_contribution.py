import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, brier_score_loss,
    balanced_accuracy_score
)

FOLDER = "ai-services/clinical/outputs/common_723"
OUT = os.path.join(FOLDER, "modality_analysis")
os.makedirs(OUT, exist_ok=True)

files = {
    "X-ray Only": ("x_ray_only_common_predictions.csv", "probability_pneumonia"),
    "Text Only": ("text_only_common_predictions.csv", "predicted_probability"),
    "History Only": ("history_only_common_predictions.csv", "predicted_probability"),
    "Early Fusion Best": ("early_fusion_best_common_predictions.csv", "predicted_probability"),
    "Early Fusion Continued": ("early_fusion_continued_common_predictions.csv", "predicted_probability"),
    "Joint Fusion": ("joint_fusion_common_predictions.csv", "original_probability")
}

results = []
probabilities = {}
reference = None

for name, (filename, probability_col) in files.items():
    df = pd.read_csv(os.path.join(FOLDER, filename))
    df = df.sort_values(["patient_id", "path_key"]).reset_index(drop=True)

    if reference is None:
        reference = df[["patient_id", "path_key", "actual_label"]].copy()
    else:
        check = df[["patient_id", "path_key", "actual_label"]]
        assert reference.equals(check), f"Alignment mismatch: {name}"

    y = df["actual_label"].astype(int).to_numpy()
    p = df[probability_col].astype(float).to_numpy()
    pred = df["predicted_label"].astype(int).to_numpy()

    probabilities[name] = p

    results.append({
        "Model": name,
        "Records": len(df),
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, p),
        "PR_AUC": average_precision_score(y, p),
        "Balanced_Accuracy": balanced_accuracy_score(y, pred),
        "Brier_Score": brier_score_loss(y, p)
    })

metrics = pd.DataFrame(results)
metrics.to_csv(os.path.join(OUT, "modality_metrics.csv"), index=False)

print("\n========== MODALITY PERFORMANCE ==========")
print(metrics.round(4).to_string(index=False))

# Compare fused models against unimodal baselines
baseline_names = ["X-ray Only", "Text Only", "History Only"]
fusion_names = ["Early Fusion Best", "Early Fusion Continued", "Joint Fusion"]

deltas = []

for fusion in fusion_names:
    fusion_row = metrics[metrics["Model"] == fusion].iloc[0]

    for baseline in baseline_names:
        base_row = metrics[metrics["Model"] == baseline].iloc[0]

        deltas.append({
            "Fusion Model": fusion,
            "Compared Against": baseline,
            "Accuracy_Delta": fusion_row["Accuracy"] - base_row["Accuracy"],
            "F1_Delta": fusion_row["F1"] - base_row["F1"],
            "ROC_AUC_Delta": fusion_row["ROC_AUC"] - base_row["ROC_AUC"],
            "PR_AUC_Delta": fusion_row["PR_AUC"] - base_row["PR_AUC"],
            "Brier_Delta": fusion_row["Brier_Score"] - base_row["Brier_Score"]
        })

delta_df = pd.DataFrame(deltas)
delta_df.to_csv(os.path.join(OUT, "fusion_vs_unimodal_deltas.csv"), index=False)

print("\n========== FUSION IMPROVEMENT ==========")
print(delta_df.round(4).to_string(index=False))

# Probability correlation between modalities/models
prob_df = pd.DataFrame(probabilities)
correlations = prob_df.corr()
correlations.to_csv(os.path.join(OUT, "prediction_probability_correlations.csv"))

# Performance comparison figure
plot_metrics = metrics.set_index("Model")[["ROC_AUC", "PR_AUC", "F1"]]
ax = plot_metrics.plot(kind="bar", figsize=(12, 6))
ax.set_title("Unimodal and Multimodal Model Performance")
ax.set_ylabel("Score")
ax.set_ylim(0, 1.05)
ax.set_xticklabels(ax.get_xticklabels(), rotation=25, ha="right")
plt.tight_layout()
plt.savefig(os.path.join(OUT, "modality_performance_comparison.png"), dpi=300)
plt.close()

# Probability correlation figure
fig, ax = plt.subplots(figsize=(9, 7))
im = ax.imshow(correlations, vmin=-1, vmax=1, aspect="auto")
ax.set_xticks(range(len(correlations.columns)))
ax.set_xticklabels(correlations.columns, rotation=45, ha="right")
ax.set_yticks(range(len(correlations.index)))
ax.set_yticklabels(correlations.index)
ax.set_title("Prediction Probability Correlation")
fig.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(os.path.join(OUT, "prediction_correlation.png"), dpi=300)
plt.close()

print("\nSaved all results in:", OUT)
print("Alignment verified:", len(reference), "records")
