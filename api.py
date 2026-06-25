import numpy as np
import joblib
from flask import Flask, request, jsonify
import logging

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

app      = Flask(__name__)
MODEL_DIR = "models"

# Loading artefacts
model         = joblib.load(f"{MODEL_DIR}/isolation_forest.pkl")
scaler        = joblib.load(f"{MODEL_DIR}/scaler.pkl")
feature_names = joblib.load(f"{MODEL_DIR}/feature_names.pkl")
logger.info(f"Model loaded. Features: {feature_names}")

TYPE_MAP = {"L": 0, "M": 1, "H": 2}

# Feature engineering that matches preprocessing feature engineering
def engineer_features(type_enc, air_temp, process_temp,
                      rotational_speed, torque, tool_wear):
    import math
    temp_difference = process_temp - air_temp
    power_watts     = torque * (rotational_speed * 2 * math.pi / 60)
    tool_torque     = tool_wear * torque
    power_deviation = abs(power_watts - 6250)

    # Column order must match feature_names saved by preprocessing
    return np.array([[type_enc, air_temp, process_temp, rotational_speed,
                      torque, tool_wear,
                      temp_difference, power_watts, tool_torque, power_deviation]])


def predict_one(data: dict):
    # Validate type field
    raw_type = str(data.get("type", "")).upper()
    if raw_type not in TYPE_MAP:
        raise ValueError(f"'type' must be L, M, or H — got '{raw_type}'")
    type_enc = TYPE_MAP[raw_type]

    air_temp   = float(data["air_temp"])
    proc_temp  = float(data["process_temp"])
    rpm        = float(data["rotational_speed"])
    torque     = float(data["torque"])
    tool_wear  = float(data["tool_wear"])

    X_raw    = engineer_features(type_enc, air_temp, proc_temp, rpm, torque, tool_wear)
    X_scaled = scaler.transform(X_raw)

    anomaly_score = float(-model.decision_function(X_scaled)[0])
    is_anomaly    = bool(model.predict(X_scaled)[0] == -1)
    return anomaly_score, is_anomaly


# Endpoints
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "model": "isolation_forest",
                    "features": feature_names})


@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON with fields:
      type (L/M/H), air_temp, process_temp,
      rotational_speed, torque, tool_wear

    Returns:
      anomaly_score (higher = more anomalous),
      is_anomaly (true/false)
    """
    data = request.get_json(force=True)
    if data is None:
        return jsonify({"error": "No JSON body"}), 400

    required = ["type", "air_temp", "process_temp",
                "rotational_speed", "torque", "tool_wear"]
    missing = [f for f in required if f not in data]
    if missing:
        return jsonify({"error": f"Missing fields: {missing}"}), 400

    try:
        anomaly_score, is_anomaly = predict_one(data)
    except (ValueError, TypeError) as e:
        return jsonify({"error": str(e)}), 400

    logger.info(f"type={data['type']} air={data['air_temp']} "
                f"proc={data['process_temp']} rpm={data['rotational_speed']} "
                f"torque={data['torque']} wear={data['tool_wear']} → "
                f"score={anomaly_score:.4f} anomaly={is_anomaly}")

    return jsonify({
        "anomaly_score": round(anomaly_score, 5),
        "is_anomaly":    is_anomaly,
        "threshold":     0.0,
        "input":         data
    })


@app.route("/predict_batch", methods=["POST"])
def predict_batch():
    data = request.get_json(force=True)
    if not isinstance(data, list):
        return jsonify({"error": "Expected a JSON array"}), 400

    results = []
    for i, row in enumerate(data):
        try:
            s, is_anom = predict_one(row)
            results.append({"index": i, "anomaly_score": round(s, 5),
                            "is_anomaly": is_anom})
        except Exception as e:
            results.append({"index": i, "error": str(e)})

    return jsonify(results)


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)