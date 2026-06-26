# Anomaly Detection System — AI4I 2020 Predictive Maintenance

End-to-end machine learning pipeline for detecting anomalies in manufacturing enivronment using Isolation Forest.
Processes continuous sensor data through a REST API and flags anomalous machine behaviour in real time.

**Author:** Jana Petrovic

---

## Dataset

Download the AI4I 2020 Predictive Maintenance dataset from Kaggle and place the CSV file in a `data/` folder in the project root:

https://www.kaggle.com/datasets/stephanmatzka/predictive-maintenance-dataset-ai4i-2020

```
AnomalyDetection/
└── data/
    └── ai4i2020.csv
```

---

## Setup

```bash
git clone https://github.com/petrovicjana/AnomalyDetection.git
cd AnomalyDetection
pip install -r requirements.txt
```

---

## How to Run

All scripts are run from the project root. Folders `models/` and `outputs/` are created automatically.

**Step 1 — Preprocessing**
```bash
python preprocessing.py
```

**Step 2 — Train**
```bash
python train.py
```

**Step 3 — Evaluate**
```bash
python evaluate.py
```
Evaluation plots are saved to `outputs/`.

**Step 4 — Start the API (Terminal 1)**
```bash
python api.py
```
API runs on `http://localhost:5000`. Keep this terminal open.

**Step 5 — Start the stream simulator (Terminal 2)**
```bash
python simulate_stream.py
```
Sends a sensor reading every 1.5 seconds. Detections are printed with `*** ANOMALY DETECTED ***`.

---

## API Usage

**Health check:**
```bash
curl http://localhost:5000/health
```

**Single prediction:**
```bash
curl -X POST http://localhost:5000/predict \
  -H "Content-Type: application/json" \
  -d '{"type": "L", "air_temp": 298.5, "process_temp": 309.2,
       "rotational_speed": 1450, "torque": 42.0, "tool_wear": 180}'
```

**Response:**
```json
{
  "anomaly_score": 0.08234,
  "is_anomaly": false,
  "threshold": 0.0
}
```
