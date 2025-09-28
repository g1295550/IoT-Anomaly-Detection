# Smart Home IoT Anomaly Detection

## Overview

This project simulates IoT sensor data for a smart home and applies anomaly detection to identify unusual events such as:

- **Security breaches** (unexpected door/window openings at night)
- **Appliance failures** (fridge power drops to zero)
- **Environmental issues** (sudden spikes in humidity/temperature)

The pipeline:

1. Generate synthetic time-series data (`generate_data.py`)
2. Apply anomaly detection (`detect_anomalies.py`)
3. Output flagged anomalies

---

## Anomaly Definitions

- **Temperature spikes/drops** inconsistent with daily cycles
- **Fridge power loss** → power consumption flatlines at 0
- **Unexpected motion/door state** at odd hours (nighttime activity)
- **Humidity surge** in short time frame

---

## Approach

- Data simulation captures realistic **daily/weekly patterns**.
- For anomaly detection, we start with:
  - Baseline with **rolling z-score** for interpretable thresholds.

**Tradeoffs:**

- Statistical methods = simple & explainable, but may miss complex multi-sensor anomalies.
- Isolation Forest = more robust, but harder to tune.

---

## Setup & Usage

```bash
# 1. Create environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

# 3. Generate dataset
python generate_data.py

# 4. Run anomaly detection
python detect_anomalies.py
```
