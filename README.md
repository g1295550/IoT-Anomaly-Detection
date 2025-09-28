# Smart Home IoT Anomaly Detection

## Overview

<<<<<<< HEAD
This project simulates IoT sensor data for a smart home and applies anomaly detection to identify unusual events such as:
=======
This PoC simulates IoT sensor data for a smart home and applies anomaly detection to identify unusual events such as:
>>>>>>> dd2c491 (PoC)

- **Security breaches** (unexpected door/window openings at night)
- **Appliance failures** (fridge power drops to zero)
- **Environmental issues** (sudden spikes in humidity/temperature)

<<<<<<< HEAD
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
=======

## Synthetic Data Generation

The project generates realistic IoT sensor data for a smart home environment using a simulation approach that models both human behavior and environmental patterns.

### Normal Data Overview

- **Duration:** 6 months (180 days) starting from January 1, 2020
- **Frequency:** 1-minute intervals
- **Sensors:**
  - **Environmental:**  
    - `temperature`: Sydney-based seasonal variation  
    - `humidity`: Correlated with temperature  
    - `fridge_power`: Consumption based on environment
  - **Motion & Access:**  
    - `sensor_motion`: Apartment motion detection  
    - `sensor_window`: Window open/close states  
    - `sensor_door`: Door open/close states

### Simulation Architecture

- **Person Models:**  
  - Two residents (Alice and Bob) with distinct work schedules, sleep patterns, and interaction frequencies
  - Room occupancy and daily routines
- **Apartment-Level Integration:**  
  - Aggregates individual behaviors into apartment-wide sensor readings
  - Realistic motion detection and access patterns

### Key Features

- **Realistic Patterns:** Environmental data follows Sydney climate
- **Human Behavior:** Includes work, sleep, and daily routines
- **Temporal Correlation:** Sensors maintain realistic relationships
- **Configurable Parameters:** Adjustable timeframes, intervals, and behaviors


## Anomaly Injection

Anomaly injection simulates realistic IoT sensor failures and environmental disruptions for testing detection algorithms.

### Anomaly Types

1. **Basic Fixed-Value Anomalies**
   - **Method:** Sets sensors to fixed values during failure periods
   - **Examples:** Temperature stuck at -5°C, humidity at 0%, fridge power at 0W
   - **Detection:** Easier to identify, suitable for initial algorithm development

2. **Realistic Anomalies**
   - **Method:** Gradual changes mimicking real-world failures
   - **Examples:**  
     - Temperature drift (calibration issues)  
     - Humidity spikes (water leaks)  
     - Fridge efficiency degradation (dirty coils)  
     - Binary sensor false triggers/stuck states

### Available Anomaly Methods

- **Environmental Sensors:**  
  - `inject_temperature_drift_anomalies`  
  - `inject_temperature_spike_anomalies`  
  - `inject_humidity_sudden_change_anomalies`  
  - `inject_fridge_power_outage_anomalies`  
  - `inject_fridge_efficiency_degradation_anomalies`
- **Binary Sensors:**  
  - `inject_sensor_stuck_anomalies`  
  - `inject_sensor_false_trigger_anomalies`

### Anomaly Structure

- Each sensor has individual anomaly indicators:  
  - `is_anomaly_temperature`, `is_anomaly_humidity`, `is_anomaly_fridge_power`
  - `is_anomaly_sensor_window`, `is_anomaly_sensor_door`, `is_anomaly_sensor_motion`
  - `is_anomaly`: Overall indicator (TRUE if any sensor has anomaly)

### Generated Datasets

- `synthetic_sensors_basic_anomalies.csv`: Simple fixed-value failures
- `synthetic_sensors_realistic_anomalies.csv`: Complex realistic patterns



## Technical Approach

### Model Architecture

- **LSTM Autoencoder** with separate handling for continuous vs binary sensors
- **Mixed Loss Function**: MSE for continuous features, BCE for binary sensors
- **Per-feature Thresholds**: Optimized using F1-score on validation data

### Justification

- **LSTM**: Captures temporal dependencies in sensor data
- **Autoencoder**: Unsupervised learning suitable for rare anomalies
- **Per-feature approach**: Different sensors have different normal ranges

### Tradeoffs

- **Pros**: Handles temporal patterns, no labeled data required, interpretable errors
- **Cons**: Requires tuning, may miss novel anomaly types, computational overhead (slow training)



## Quick Start
>>>>>>> dd2c491 (PoC)

```bash
# 1. Create environment
python3 -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# 2. Install dependencies
pip install -r requirements.txt

<<<<<<< HEAD
# 3. Generate dataset
python generate_data.py

# 4. Run anomaly detection
python detect_anomalies.py
```
=======
# 3. Enter directory scripts
cd scripts

# 4. Generate dataset
python generate_data.py

# 5. Run notebook
# -> generate_anomalies.ipynb

# 6. Run anomaly detection
# -> detect_anomalies.ipynb
```


## Productionization Discussion

### Deployment Architecture

```
IoT Sensors → Message Queue → Batch Processing → Alert System
             (Kafka)        (Spark/Airflow)   (Email/SMS/Push notification)
```


To productionize the IoT anomaly detection system, consider both batch and real-time serving options:

- **Model Serving**:  
  - *Batch Processing*: Aggregate sensor data periodically (e.g., hourly or daily) and run anomaly detection in batches. Suitable for non-critical alerts and resource efficiency.
  - *Real-Time Processing*: Stream sensor data through a message queue (e.g., Kafka) and apply anomaly detection as data arrives. Enables immediate alerts for critical events.

- **Monitoring**:  
  - Track model performance using metrics like reconstruction error distributions and anomaly rates.
  - Implement dashboards to visualize trends and detect data drift or sudden changes in sensor behavior.

- **Retraining**:  
  - Schedule regular retraining (e.g., weekly or monthly) using the latest data to adapt to evolving patterns.
  - Set up automated triggers for retraining if performance degrades or data drift is detected.
  - Use A/B testing to compare new model versions before full deployment.

This approach ensures timely anomaly detection, robust monitoring, and continuous model improvement in a production environment.



## Results

### Statistical Methods - Performance

| Method          | Accuracy | Precision | Recall | F1-Score |
|-----------------|----------|-----------|--------|----------|
| Z-Score         | 96.62%   | 26.73%    | 3.15%  | 0.0564   |   
| IQR             | 90.53%   | 4.79%     | 10.34% | 0.0655   |
| Moving Average  | 96.82%   | 65.22%    | 1.80%  | 0.0351   |

Statistical methods tend to show limited overall performance. While they are more effective when dealing with continuous variables and **can discriminate well against fixed-value** defined anomalies, their performance drops significantly when applied to realistic datasets with more complex anomalies.

### Key Performance Issues Identified
- Low Recall Across All Methods: Missing 89–98% of actual anomalies  
- High False Negative Rates: Critical for anomaly detection systems  
- **Statistical Methods Limitations:** Struggle with realistic, gradual anomalies  

### LSTM Autoencoder Implementation
- **Architecture:** Multi-layer encoder-decoder with latent bottleneck  
- **Per-Feature Thresholds:** Individual thresholds optimized per sensor type  
- **Mixed Loss Function:** Handles both continuous and binary sensor data  
- **Temporal Modeling:** 180-step sequences for continuous, 60-step for binary features  

### LSTM Autoencoder - Performance

#### Clear Sensor Type Performance Stratification

#### Continuous Environmental Sensors (Success Story)
- Temperature and fridge power sensors achieve decent performance (F1: 0.61–0.78).  
- The model successfully learns  temporal patterns in environmental data.  
- High recall rates (84–100%) ensure that critical failures are rarely missed.  
- Performance improvement from basic to realistic datasets suggests robust pattern learning.  

#### Binary Behavioral Sensors (Critical Weakness)
- Motion, door, and window sensors show uniformly poor performance (F1: 0.02–0.09).  
- Sparse activation patterns (e.g., motion active only ~7%) create an insufficient training signal.  
- The autoencoder architecture is fundamentally mismatched to discrete event detection.  

This result has  implications for system design, indicating the need for hybrid architectures that better align model capabilities with data characteristics.




### Next Steps for Improvement
1. **Data Balancing Strategies**  
   - Weighted Loss Functions: Penalize false negatives more heavily  
   - Threshold Optimization: Use validation F1-score instead of percentile-based thresholds  

2. **Advanced Model Architectures**  
   - Transformer-based Models: Attention mechanisms for temporal patterns  
   - Ensemble Methods: Combine multiple detection approaches  

3. **Feature Engineering Improvements**  
   - Temporal Features: Hour, day-of-week, seasonal patterns  
   - Rolling Statistics: Use moving averages, standard deviations for neural network training

4. **Evaluation Enhancements**  
   - Per-Anomaly-Type Evaluation: Different metrics for different anomaly categories  
   - Cost-Sensitive Metrics: Weight false negatives vs false positives based on specific anomaly type 
   - Precision-Recall Curves: Better threshold selection  

5. **Threshold Optimization**  
   - F1-Score Maximization: Find optimal thresholds per feature  
   - ROC Analysis: Balance sensitivity vs specificity  
   - Adaptive Thresholds: Adjust based on recent data patterns  
   - Multi-Level Alerting: Different thresholds for different severity levels  


## AI Interaction

* VS Code + GitHub Copilot Usage

* Development was supported with VS Code and GitHub Copilot.

* Difficulty to capture inline interactions directly, a method for automatically saving transcripts should be explored.

* Copilot’s inline assistance was leveraged throughout the process.
>>>>>>> dd2c491 (PoC)
