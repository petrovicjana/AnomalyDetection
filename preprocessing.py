import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib
import os

# Config
DATA_PATH      = "data/ai4i2020.csv"       
OUTPUT_DIR     = "models"
LABEL_COL      = "machine_failure"
TEST_SIZE      = 0.2
RANDOM_SEED    = 42
AUGMENT_FACTOR = 1    

os.makedirs(OUTPUT_DIR, exist_ok=True)
np.random.seed(RANDOM_SEED)


# ---- Step 1
df = pd.read_csv(DATA_PATH)
df.columns = [c.strip().lower().replace(" ", "_").replace("[", "").replace("]", "") for c in df.columns]

print(" /// Step 1: Loading & cleaning")
print(f"Raw columns: {df.columns.tolist()}")
print(f"Shape: {df.shape}")
print(f"Missing values:\n{df.isnull().sum()[df.isnull().sum() > 0]}")

# Dropping identifier (no value) and sub-label columns (to not leak label into features)
drop_cols = ["uid", "product_id",
             "twf", "hdf", "pwf", "osf", "rnf"]
df = df.drop(columns=[c for c in drop_cols if c in df.columns])

type_col = "type" if "type" in df.columns else None

n_normal  = (df[LABEL_COL] == 0).sum()
n_anomaly = (df[LABEL_COL] == 1).sum()
print(f"\nAfter dropping identifiers and sub-labels: {df.shape}")
print(f"Normal: {n_normal}  |  Anomaly: {n_anomaly}  |  "
      f"Anomaly rate: {n_anomaly/len(df)*100:.2f}%")


# ---- Step 2 
# We do ordinal encoding since H adds more tool wear minutes than M which adds more than L, L=0, M=1, H=2
print(" /// Step 2:Encoding categoricals")
if type_col:
    type_map = {"L": 0, "M": 1, "H": 2}
    df["type_encoded"] = df[type_col].map(type_map)
    df = df.drop(columns=[type_col])
    print(f"\nType encoding: {type_map}")
    print(df["type_encoded"].value_counts().sort_index())


# ---- Step 3
# Rename sensor columns to clean names (handles suffixes like _k)
print(" /// Step 3: Selecting base features")
rename_map = {}  
for col in df.columns:
    if "air_temperature"     in col: rename_map[col] = "air_temp"
    if "process_temperature" in col: rename_map[col] = "process_temp"
    if "rotational_speed"    in col: rename_map[col] = "rotational_speed"
    if "torque"              in col: rename_map[col] = "torque"
    if "tool_wear"           in col: rename_map[col] = "tool_wear"
df = df.rename(columns=rename_map)

BASE_FEATURES = ["type_encoded", "air_temp", "process_temp",
                 "rotational_speed", "torque", "tool_wear"]

print(f"Using: {BASE_FEATURES}")

cols_needed = BASE_FEATURES + [LABEL_COL]
df = df[cols_needed].dropna()
print(f"Shape after dropna: {df.shape}")


# ---- Step 4
print(" /// Step 4: Train,test,split")
X = df[BASE_FEATURES].copy()
y = df[LABEL_COL].copy()

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)
X_train = X_train.reset_index(drop=True)
X_test  = X_test.reset_index(drop=True)
y_train = y_train.reset_index(drop=True)
y_test  = y_test.reset_index(drop=True)

print(f"Train → normal: {(y_train==0).sum()}  anomaly: {(y_train==1).sum()}")
print(f"Test  → normal: {(y_test==0).sum()}   anomaly: {(y_test==1).sum()}")


# ---- Step 5
print(" /// Step 5: Feature Engineering")
# This is derived from the failure mode formulas in dataset description
def engineer_features(df_feat: pd.DataFrame) -> pd.DataFrame:
    out = df_feat.copy()

    # When temperature difference is low, head dissipation failure occurs (values below 8.6K)
    # Calculated by substracting air from process temperature 
    out["temp_difference"] = out["process_temp"] - out["air_temp"]

    # power watts feature: triggers power failure when outside the range of [3500,9000 W]
    # Converting rpm to rad/s: × 2π/60
    out["power_watts"] = out["torque"] * (out["rotational_speed"] * 2 * np.pi / 60)

    # tool torque high values signal overstrain failure OSF which occurs above 11000-13000 min·Nm depending on type
    out["tool_torque"] = out["tool_wear"] * out["torque"]

    # power deviation is distance from the centre of normal power range of 6250 W, large abs deviation signals anomaly
    out["power_deviation"] = np.abs(out["power_watts"] - 6250)

    return out

X_train_eng = engineer_features(X_train)
X_test_eng  = engineer_features(X_test)
feature_names = X_train_eng.columns.tolist()

print(f"Features ({len(feature_names)}): {feature_names}")
print(f"\nEngineered feature stats on training set:")
eng_cols = ["temp_difference", "power_watts", "tool_torque", "power_deviation"]
print(X_train_eng[eng_cols].describe().round(2))

# Sanity check: do engineered features show the expected failure triggers?
print(f"\nSanity checks:")
print(f"  temp_difference < 8.6 K in train: "
      f"{(X_train_eng['temp_difference'] < 8.6).sum()} rows")
print(f"  power_watts < 3500 W in train:    "
      f"{(X_train_eng['power_watts'] < 3500).sum()} rows")
print(f"  power_watts > 9000 W in train:    "
      f"{(X_train_eng['power_watts'] > 9000).sum()} rows")
print(f"  tool_torque > 11000 in train:     "
      f"{(X_train_eng['tool_torque'] > 11000).sum()} rows")


# ---- Step 6
print(" /// Step 6: Scalling")
X_train_normal = X_train_eng[y_train == 0].copy()

scaler = StandardScaler()
scaler.fit(X_train_normal)

X_train_normal_scaled = scaler.transform(X_train_normal)
X_train_scaled        = scaler.transform(X_train_eng)
X_test_scaled         = scaler.transform(X_test_eng)

print(f"Scaler fitted on {len(X_train_normal)} normal training rows only")
print(f"Train-normal shape (for IF):  {X_train_normal_scaled.shape}")  # only normal rows that get fed to isolation forest
print(f"Full train shape  (for eval): {X_train_scaled.shape}") # all training (with anomalies as well)
print(f"Test shape:                   {X_test_scaled.shape}")


# ---- Step 7
print(" /// Step 7: Saving artifacts")
joblib.dump(scaler,        f"{OUTPUT_DIR}/scaler.pkl")
joblib.dump(feature_names, f"{OUTPUT_DIR}/feature_names.pkl")

np.save(f"{OUTPUT_DIR}/X_train_normal.npy", X_train_normal_scaled)
np.save(f"{OUTPUT_DIR}/X_train.npy",         X_train_scaled)
np.save(f"{OUTPUT_DIR}/y_train.npy",          y_train.values)
np.save(f"{OUTPUT_DIR}/X_test.npy",           X_test_scaled)
np.save(f"{OUTPUT_DIR}/y_test.npy",           y_test.values)

joblib.dump(BASE_FEATURES, f"{OUTPUT_DIR}/base_features.pkl")

print(f"Saved to {OUTPUT_DIR}/")