# Copilot Instructions for IoT-Anomaly-Detection

## Project Overview

This repository simulates IoT sensor data for a smart home and applies anomaly detection to identify unusual events. The pipeline includes:

1. **Data Generation**: `scripts/generate_data.py` creates synthetic time-series data with realistic daily/weekly patterns and injected anomalies.
2. **Anomaly Detection**: Models like Isolation Forest analyze the data to flag anomalies.
3. **Exploration**: Jupyter notebooks in `notebooks/` are used for exploratory data analysis (EDA) and model comparisons.

## Key Files and Directories

- **`scripts/generate_data.py`**: Generates synthetic sensor data with configurable anomalies.
- **`data/`**: Stores generated datasets (e.g., `synthetic_sensors.csv`).
- **`notebooks/`**: Contains Jupyter notebooks for analysis and experimentation.
- **`requirements.txt`**: Lists Python dependencies (currently empty, update as needed).
- **`README.md`**: Provides setup instructions and project context.

## Development Guidelines

### Code Style

- Use Python 3.9+.
- Follow PEP8 standards.
- Document functions with clear docstrings.
- Use type hints where applicable.

### Data Generation

- The `generate_data.py` script simulates sensor data with anomalies. Key parameters include:
  - `start_date`, `end_date`: Define the data range.
  - `interval_minutes`: Frequency of data points.
  - `anomaly_params`: Configure anomaly types and probabilities.
- Example usage:
  ```bash
  python scripts/generate_data.py
  ```

### Anomaly Detection

- Default model: Isolation Forest.
- Compare with simpler methods like rolling z-scores.
- Output anomalies with timestamps and provide visualizations.

### Notebooks

- Use Jupyter notebooks for EDA and model evaluation.
- Document trade-offs and insights.

## AI Agent-Specific Instructions

- **Refactoring**: Prioritize modular, reusable code over monolithic scripts.
- **Anomaly Creativity**: When generating anomalies, ensure they are plausible (e.g., fridge power loss, unexpected motion at night).
- **Preserve Context**: Retain comments and prompts in files to maintain clarity.
- **Ask Questions**: If uncertain about implementation details, add clarifying comments.

## Setup and Usage

1. Create a virtual environment:
   ```bash
   python3 -m venv myenv
   source myenv/bin/activate
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```
3. Generate synthetic data:
   ```bash
   python scripts/generate_data.py
   ```
4. Run anomaly detection (script to be implemented):
   ```bash
   python detect_anomalies.py
   ```

## Notes

- Update `requirements.txt` as dependencies are added.
- Save trained models in the `models/` directory.
- Use `ai_transcripts/` for storing AI-generated insights or logs.
