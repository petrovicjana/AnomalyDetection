import numpy as np
import pandas as pd
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, classification_report,
    roc_auc_score, roc_curve,
    precision_recall_curve, average_precision_score
)
import os

# ---- Config 
MODEL_DIR  = "models"
OUTPUT_DIR = "outputs"
DATA_PATH  = "data/ai4i2020.csv"
LABEL_COL  = "machine_failure"

os.makedirs(OUTPUT_DIR, exist_ok=True)
plt.style.use("seaborn-v0_8-whitegrid")
COLORS = {"normal": "steelblue", "anomaly": "tomato"}

# ---- Load model and test data 
model  = joblib.load(f"{MODEL_DIR}/isolation_forest.pkl")
X_test = np.load(f"{MODEL_DIR}/X_test.npy")
y_test = np.load(f"{MODEL_DIR}/y_test.npy")

# Isolation Forest predict outputs 1 for normal and -1 for anomaly
# decision_function returns continious score, higher = more normal
raw_preds     = model.predict(X_test)
scores        = model.decision_function(X_test)
y_pred        = (raw_preds == -1).astype(int)   # convert to 0/1
anomaly_scores = -scores   # negate so higher = more anomalous

# ---- Metrics 
print(" /// Classification Report")
print(classification_report(y_test, y_pred, target_names=["Normal", "Anomaly"]))

roc_auc  = roc_auc_score(y_test, anomaly_scores)
avg_prec = average_precision_score(y_test, anomaly_scores)
print(f"ROC-AUC:       {roc_auc:.3f}")
print(f"Avg Precision: {avg_prec:.3f}")

cm = confusion_matrix(y_test, y_pred)
tn, fp, fn, tp = cm.ravel()
print(f"\nConfusion matrix:")
print(f"  TN={tn}  FP={fp}")
print(f"  FN={fn}  TP={tp}")
print(f"\nFalse positive rate: {fp/(fp+tn)*100:.1f}%")
print(f"False negative rate: {fn/(fn+tp)*100:.1f}%  "
      f"(anomalies missed)")


# Confusion matrix 
fig, ax = plt.subplots(figsize=(5, 4))
im = ax.imshow(cm, cmap="Blues")

ax.set_xticks([0, 1])
ax.set_yticks([0, 1])
ax.set_xticklabels(["Predicted\nNormal", "Predicted\nAnomaly"], fontsize=10)
ax.set_yticklabels(["Actual\nNormal", "Actual\nAnomaly"], fontsize=10)

for i in range(2):
    for j in range(2):
        ax.text(
            j, i, cm[i, j],
            ha="center", va="center",
            fontsize=20, fontweight="bold",
            color="white" if cm[i, j] > cm.max() / 2 else "black"
        )

ax.set_title(
    f"Confusion matrix\nROC-AUC: {roc_auc:.3f}",
    fontweight="bold"
)
plt.colorbar(im, ax=ax)
plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/03_confusion_matrix.png", dpi=150)
plt.close()
print("Confusion matrix saved")


# ROC and Precision-Recall curves
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# ROC curve
fpr, tpr, _ = roc_curve(y_test, anomaly_scores)
ax1.plot(fpr, tpr, color="#534ab7", lw=2, label=f"AUC = {roc_auc:.3f}")
ax1.plot([0, 1], [0, 1], "k--", lw=1, alpha=0.4, label="Random classifier")
ax1.fill_between(fpr, tpr, alpha=0.08, color="#534ab7")
ax1.set_xlabel("False positive rate")
ax1.set_ylabel("True positive rate")
ax1.set_title("ROC curve", fontweight="bold")
ax1.legend()

# Precision-Recall curve
prec, rec, _ = precision_recall_curve(y_test, anomaly_scores)
baseline = y_test.mean()
ax2.plot(rec, prec, color=COLORS["anomaly"], lw=2,
         label=f"Avg precision = {avg_prec:.3f}")
ax2.axhline(baseline, color="black", linestyle="--", lw=1, alpha=0.5,
            label=f"Baseline = {baseline:.3f}")
ax2.fill_between(rec, prec, alpha=0.08, color=COLORS["anomaly"])
ax2.set_xlabel("Recall")
ax2.set_ylabel("Precision")
ax2.set_title("Precision-Recall curve", fontweight="bold")
ax2.legend()

plt.tight_layout()
plt.savefig(f"{OUTPUT_DIR}/04_roc_pr_curves.png", dpi=150)
plt.close()
print("ROC AUC and Precision-Recall curve saved")


# Anomaly scores broken down by failure mode 
# Re-load raw data to get the individual failure mode columns for the test rows
# This plot is unique to AI4I — it shows whether the model scores each failure TYPE differently

try:
    raw = pd.read_csv(DATA_PATH)
    raw.columns = [
        c.strip().lower()
         .replace(" ", "_")
         .replace("[", "")
         .replace("]", "")
        for c in raw.columns
    ]

    failure_mode_cols = [c for c in ["twf", "hdf", "pwf", "osf", "rnf"]
                         if c in raw.columns]

    if failure_mode_cols:
        # Align the test rows back to the original dataframe 
        from sklearn.model_selection import train_test_split

        drop_ids = ["uid", "product_id"]
        raw_clean = raw.drop(columns=[c for c in drop_ids if c in raw.columns])

        _, test_idx = train_test_split(
            raw_clean.index,
            test_size=0.2,
            random_state=42,
            stratify=raw_clean[LABEL_COL]
        )

        test_meta = raw_clean.loc[test_idx, failure_mode_cols + [LABEL_COL]].reset_index(drop=True)
        test_meta["anomaly_score"] = anomaly_scores
        test_meta["predicted_anomaly"] = y_pred

        # A row can have multiple modes — we assign the first active one for display
        def failure_label(row):
            if row[LABEL_COL] == 0:
                return "Normal"
            for mode in failure_mode_cols:
                if row[mode] == 1:
                    return mode.upper()
            return "Unknown"

        test_meta["failure_mode"] = test_meta.apply(failure_label, axis=1)

        mode_order = ["Normal"] + [m.upper() for m in failure_mode_cols] + ["Unknown"]
        mode_order = [m for m in mode_order if m in test_meta["failure_mode"].unique()]

        palette = {
            "Normal": COLORS["normal"],
            "TWF": "#e07b39", "HDF": "#c0392b",
            "PWF": "#8e44ad", "OSF": "#d35400", "RNF": "#7f8c8d",
            "Unknown": "#95a5a6"
        }

        fig, ax = plt.subplots(figsize=(10, 5))
        sns.boxplot(
            data=test_meta,
            x="failure_mode",
            y="anomaly_score",
            order=mode_order,
            palette=palette,
            ax=ax,
            linewidth=0.8,
            flierprops=dict(marker="o", markersize=3, alpha=0.4)
        )
        ax.axhline(0, color="black", linestyle="--", linewidth=1.2,
                   label="Decision threshold")
        ax.set_xlabel("Failure mode  (Normal -> no failure)")
        ax.set_ylabel("Anomaly score  (higher -> more anomalous)")
        ax.set_title(
            "Anomaly score by failure mode\n"
            "Boxes above threshold line = correctly identified as anomalous",
            fontweight="bold"
        )
        ax.legend()

        # Annotate each box with its count
        for i, mode in enumerate(mode_order):
            n = (test_meta["failure_mode"] == mode).sum()
            ax.text(i, test_meta["anomaly_score"].min() - 0.01,
                    f"n={n}", ha="center", fontsize=8, color="gray")

        plt.tight_layout()
        plt.savefig(f"{OUTPUT_DIR}/05_score_by_failure_type.png", dpi=150)
        plt.close()
        print("Saved 05_score_by_failure_type.png")

        # Print per-mode detection rate
        print("\n///Detection rate by failure mode")
        for mode in mode_order:
            subset = test_meta[test_meta["failure_mode"] == mode]
            if len(subset) == 0:
                continue
            detected = subset["predicted_anomaly"].sum()
            print(f"  {mode:<8} n={len(subset):3d}  detected={detected:3d}  "
                  f"rate={100*detected/len(subset):.1f}%")

except Exception as e:
    print(f"Skipped 05_score_by_failure_type.png — could not load failure modes: {e}")

print("\nEvaluation complete")