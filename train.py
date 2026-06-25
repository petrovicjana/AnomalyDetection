import numpy as np
import joblib
import os
from sklearn.ensemble import IsolationForest

MODEL_DIR    = "models"
N_ESTIMATORS = 200
RANDOM_SEED  = 42

X_train_normal = np.load(f"{MODEL_DIR}/X_train_normal.npy")
y_train        = np.load(f"{MODEL_DIR}/y_train.npy")

# Derive contamination from actual label distribution rather than hardcoding
CONTAMINATION = float(y_train.mean())

print(f"Training on {X_train_normal.shape[0]} normal rows, "
      f"{X_train_normal.shape[1]} features.")
print(f"Contamination: {CONTAMINATION:.4f} "
      f"({y_train.sum():.0f} anomalies in {len(y_train)} train rows)")

model = IsolationForest(
    n_estimators=N_ESTIMATORS,
    contamination=CONTAMINATION,
    random_state=RANDOM_SEED,
    n_jobs=-1
)
model.fit(X_train_normal)

flagged = (model.predict(X_train_normal) == -1).sum()
print(f"Sanity: flags {flagged}/{len(X_train_normal)} normal train rows "
      f"({100*flagged/len(X_train_normal):.1f}%) — expected ~{CONTAMINATION*100:.1f}%")

os.makedirs(MODEL_DIR, exist_ok=True)
joblib.dump(model, f"{MODEL_DIR}/isolation_forest.pkl")