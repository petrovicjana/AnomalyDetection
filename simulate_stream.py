import requests, random, time

API_URL  = "http://localhost:5000/predict"
INTERVAL = 1.5

def normal_reading():
    return {
        "type":             random.choice(["L", "M", "H"]),
        "air_temp":         round(random.uniform(297, 303), 1),
        "process_temp":     round(random.uniform(308, 313), 1),
        "rotational_speed": round(random.uniform(1400, 1600)),
        "torque":           round(random.uniform(30, 50), 1),
        "tool_wear":        round(random.uniform(0, 190))
    }

def anomaly_reading():
    # Violates multiple failure conditions at once
    air = round(random.uniform(297, 303), 1)
    return {
        "type":             random.choice(["L", "M", "H"]),
        "air_temp":         air,
        "process_temp":     round(air + random.uniform(5.0, 7.0), 1),
        "rotational_speed": round(random.uniform(1000, 1200)),
        "torque":           round(random.uniform(70, 90), 1),
        "tool_wear":        round(random.uniform(220, 250))
    }

print("Streaming to API. Ctrl+C to stop.")
print("Anomaly injected every 8th reading (index ending in 5)\n")
print(f"{'Index':<8} {'Type':<6} {'TDiff':>6} {'RPM':>6} {'Torque':>7} "
      f"{'Wear':>6} {'Score':>8}  {'Result'}")
print("-" * 75)

i = 0
while True:
    is_injected = (i % 8 == 5)
    reading     = anomaly_reading() if is_injected else normal_reading()
    tdiff       = round(reading["process_temp"] - reading["air_temp"], 1)

    try:
        r          = requests.post(API_URL, json=reading, timeout=3).json()
        score      = r.get("anomaly_score", 0)
        is_anomaly = r.get("is_anomaly", False)

        if is_anomaly:
            result = "*** ANOMALY DETECTED ***"
        elif is_injected:
            result = "(injected — missed)"   # injected but not caught
        else:
            result = "normal"

        print(f"[{i:04d}]   {reading['type']:<4}  {tdiff:>5.1f}K "
              f"{reading['rotational_speed']:>6}  "
              f"{reading['torque']:>6.1f}  "
              f"{reading['tool_wear']:>6}  "
              f"{score:>8.4f}  {result}")

    except Exception as e:
        print(f"[{i:04d}] error: {e}")

    i += 1
    time.sleep(INTERVAL)