
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from scipy.stats import binomtest
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, average_precision_score, balanced_accuracy_score,
    brier_score_loss
)

BASE = Path(__file__).resolve().parent / "outputs" / "common_723"
OUT = BASE / "statistical_results"
OUT.mkdir(exist_ok=True)

N_BOOT = 2000
SEED = 42

FILES = {
    "X-ray only": ("x_ray_only_common_predictions.csv", "probability_pneumonia"),
    "Text only": ("text_only_common_predictions.csv", "predicted_probability"),
    "History only": ("history_only_common_predictions.csv", "predicted_probability"),
    "Early Fusion Best": ("early_fusion_best_common_predictions.csv", "predicted_probability"),
    "Early Fusion Continued": ("early_fusion_continued_common_predictions.csv", "predicted_probability"),
    "Joint Fusion": ("joint_fusion_common_predictions.csv", "original_probability"),
}

def load_predictions():
    manifest = pd.read_csv(BASE / "common_723_manifest.csv")
    manifest["path_key"] = manifest["path_key"].astype(str)

    if manifest["path_key"].duplicated().any():
        raise ValueError("Duplicate path_key in common manifest.")

    data = {}

    for name, (filename, probability_col) in FILES.items():
        df = pd.read_csv(BASE / filename)
        df["path_key"] = df["path_key"].astype(str)

        required = {"patient_id", "path_key", "label", probability_col}
        missing = required - set(df.columns)
        if missing:
            raise ValueError(f"{name}: missing columns {missing}")

        if df["path_key"].duplicated().any():
            raise ValueError(f"{name}: duplicate path_key found.")

        merged = manifest.merge(
            df[["patient_id", "path_key", "label", probability_col]],
            on="path_key",
            how="left",
            suffixes=("_manifest", "_model"),
            validate="one_to_one"
        )

        if merged[probability_col].isna().any():
            raise ValueError(f"{name}: missing predictions after matching.")

        if not (merged["patient_id_manifest"] == merged["patient_id_model"]).all():
            raise ValueError(f"{name}: patient ID mismatch.")

        if not (merged["label_manifest"] == merged["label_model"]).all():
            raise ValueError(f"{name}: label mismatch.")

        data[name] = pd.DataFrame({
            "patient_id": merged["patient_id_manifest"],
            "path_key": merged["path_key"],
            "y": merged["label_manifest"].astype(int),
            "prob": merged[probability_col].astype(float)
        })

    print("Alignment verified for all models.")
    print("Records:", len(manifest))
    print("Unique patients:", manifest["patient_id"].nunique())

    return data


def metrics(y, prob):
    pred = (prob >= 0.5).astype(int)

    return {
        "Accuracy": accuracy_score(y, pred),
        "Precision": precision_score(y, pred, zero_division=0),
        "Recall": recall_score(y, pred, zero_division=0),
        "F1": f1_score(y, pred, zero_division=0),
        "ROC_AUC": roc_auc_score(y, prob),
        "PR_AUC": average_precision_score(y, prob),
        "Balanced_Accuracy": balanced_accuracy_score(y, pred),
        "Brier_Score": brier_score_loss(y, prob)
    }


def ece_score(y, prob, bins=10):
    edges = np.linspace(0, 1, bins + 1)
    ece = 0.0

    for i in range(bins):
        if i == bins - 1:
            mask = (prob >= edges[i]) & (prob <= edges[i + 1])
        else:
            mask = (prob >= edges[i]) & (prob < edges[i + 1])

        if mask.any():
            confidence = prob[mask].mean()
            accuracy = y[mask].mean()
            ece += mask.mean() * abs(accuracy - confidence)

    return ece


def cluster_bootstrap(df, rng):
    patients = df["patient_id"].unique()
    sampled = rng.choice(patients, size=len(patients), replace=True)

    # Keep every record belonging to each sampled patient.
    parts = [df[df["patient_id"] == patient] for patient in sampled]
    return pd.concat(parts, ignore_index=True)


def run_bootstrap(data):
    rng = np.random.default_rng(SEED)
    results = []

    for name, df in data.items():
        values = []

        for _ in range(N_BOOT):
            sample = cluster_bootstrap(df, rng)
            y = sample["y"].to_numpy()
            prob = sample["prob"].to_numpy()

            if len(np.unique(y)) < 2:
                continue

            values.append(metrics(y, prob))

        scores = pd.DataFrame(values)

        for metric in scores.columns:
            results.append({
                "Model": name,
                "Metric": metric,
                "Estimate": metrics(
                    df["y"].to_numpy(), df["prob"].to_numpy()
                )[metric],
                "CI_95_Lower": scores[metric].quantile(0.025),
                "CI_95_Upper": scores[metric].quantile(0.975),
                "Valid_Bootstraps": len(scores)
            })

    result_df = pd.DataFrame(results)
    result_df.to_csv(OUT / "cluster_bootstrap_confidence_intervals.csv", index=False)
    return result_df


def paired_comparison(data, model_a, model_b, rng):
    a = data[model_a].reset_index(drop=True)
    b = data[model_b].reset_index(drop=True)

    if not (
        (a["path_key"] == b["path_key"]).all()
        and (a["y"] == b["y"]).all()
    ):
        raise ValueError("Paired comparison alignment failed.")

    y = a["y"].to_numpy()
    pa = a["prob"].to_numpy()
    pb = b["prob"].to_numpy()

    pred_a = (pa >= 0.5).astype(int)
    pred_b = (pb >= 0.5).astype(int)

    # Exact McNemar test on paired record-level errors.
    # Interpret cautiously because patients may have multiple records.
    errors_a = pred_a != y
    errors_b = pred_b != y

    b_count = int(np.sum(errors_a & ~errors_b))
    c_count = int(np.sum(~errors_a & errors_b))
    discordant = b_count + c_count

    mcnemar_p = (
        binomtest(min(b_count, c_count), discordant, 0.5).pvalue
        if discordant else 1.0
    )

    # Patient-cluster bootstrap for paired ROC-AUC difference.
    paired = pd.DataFrame({
        "patient_id": a["patient_id"],
        "y": y,
        "prob_a": pa,
        "prob_b": pb
    })

    auc_diff = roc_auc_score(y, pa) - roc_auc_score(y, pb)
    bootstrap_diffs = []

    for _ in range(N_BOOT):
        sample = cluster_bootstrap(paired, rng)

        if sample["y"].nunique() < 2:
            continue

        diff = (
            roc_auc_score(sample["y"], sample["prob_a"])
            - roc_auc_score(sample["y"], sample["prob_b"])
        )
        bootstrap_diffs.append(diff)

    lower, upper = np.quantile(bootstrap_diffs, [0.025, 0.975])

    # Two-sided bootstrap sign-based approximation.
    diffs = np.asarray(bootstrap_diffs)
    p_boot = min(
        1.0,
        2 * min(np.mean(diffs <= 0), np.mean(diffs >= 0))
    )

    return {
        "Model A": model_a,
        "Model B": model_b,
        "Errors A only": b_count,
        "Errors B only": c_count,
        "McNemar exact p (record-level)": mcnemar_p,
        "ROC-AUC difference (A-B)": auc_diff,
        "ROC-AUC difference CI lower": lower,
        "ROC-AUC difference CI upper": upper,
        "Bootstrap sign-based p (approx.)": p_boot,
        "Valid bootstrap samples": len(diffs)
    }


def main():
    data = load_predictions()

    # Main metrics and calibration.
    rows = []

    for name, df in data.items():
        y = df["y"].to_numpy()
        prob = df["prob"].to_numpy()

        row = {"Model": name, **metrics(y, prob)}
        row["ECE"] = ece_score(y, prob)
        rows.append(row)

    summary = pd.DataFrame(rows)
    summary.to_csv(OUT / "model_metrics_and_calibration.csv", index=False)

    print("\nMODEL METRICS AND CALIBRATION")
    print(summary.round(4).to_string(index=False))

    # Patient-cluster bootstrap confidence intervals.
    print("\nRunning patient-cluster bootstrap...")
    ci = run_bootstrap(data)
    print(ci.round(4).to_string(index=False))

    # Main paired comparison: Joint Fusion vs Continued Early Fusion.
    rng = np.random.default_rng(SEED)

    comparisons = [
        ("Joint Fusion", "Early Fusion Continued"),
        ("Joint Fusion", "Early Fusion Best"),
        ("Joint Fusion", "X-ray only"),
        ("Joint Fusion", "Text only"),
    ]

    paired_results = []

    for model_a, model_b in comparisons:
        paired_results.append(
            paired_comparison(data, model_a, model_b, rng)
        )

    paired_df = pd.DataFrame(paired_results)
    paired_df.to_csv(OUT / "paired_model_comparisons.csv", index=False)

    print("\nPAIRED MODEL COMPARISONS")
    print(paired_df.round(5).to_string(index=False))

    # Reliability diagrams.
    fig, axes = plt.subplots(2, 3, figsize=(15, 9))
    axes = axes.flatten()

    for ax, (name, df) in zip(axes, data.items()):
        y = df["y"].to_numpy()
        prob = df["prob"].to_numpy()

        edges = np.linspace(0, 1, 11)
        mean_conf, fraction_positive = [], []

        for i in range(10):
            mask = (
                (prob >= edges[i]) &
                (prob <= edges[i + 1] if i == 9 else prob < edges[i + 1])
            )

            if mask.any():
                mean_conf.append(prob[mask].mean())
                fraction_positive.append(y[mask].mean())

        ax.plot([0, 1], [0, 1], "--", label="Perfect calibration")
        ax.plot(mean_conf, fraction_positive, "o-", label=name)
        ax.set_title(name)
        ax.set_xlabel("Mean predicted probability")
        ax.set_ylabel("Observed positive fraction")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
        ax.legend(fontsize=8)

    plt.tight_layout()
    plt.savefig(OUT / "reliability_diagrams.png", dpi=300)
    plt.close()

    print("\nCompleted. Results saved to:", OUT)


if __name__ == "__main__":
    main()
