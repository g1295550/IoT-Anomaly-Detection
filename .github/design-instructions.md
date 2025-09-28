# IoT Anomaly Detection - Design Instructions

## Project Overview

This repository implements a comprehensive IoT sensor data simulation and anomaly detection system for smart home environments. The system consists of three main components:

1. **Modular Data Simulation System** (`simulation/` module)
2. **Data Generation Pipeline** (`scripts/generate_data.py`)
3. **Anomaly Detection & Analysis** (Jupyter notebooks)

The architecture is designed to generate realistic multi-sensor IoT data with configurable anomaly injection, followed by sophisticated anomaly detection using both statistical and machine learning approaches.

---

## Architecture Overview

### Core Module Structure

#### 1. `simulation/` Package
A comprehensive simulation framework with the following modules:

- **`simulation/func.py`**: Core utility functions for time feature extraction and environmental data generation
- **`simulation/person.py`**: Person class modeling individual behavior patterns and sensor interactions  
- **`simulation/apartment.py`**: Apartment class aggregating multi-person sensor states
- **`simulation/anomaly.py`**: Anomaly injection system with configurable anomaly types
- **`simulation/__init__.py`**: Package initialization and exports

#### 2. `scripts/` Directory
- **`scripts/generate_data.py`**: Main data generation script that orchestrates the simulation

#### 3. `notebooks/` Directory
- **`notebooks/detect_anomalies.ipynb`**: Comprehensive anomaly detection using multiple approaches
- **`notebooks/generate_anomalies.ipynb`**: Anomaly injection and dataset generation
---

## Implementation Guidelines

### 1. Code Style & Standards

- **Python Version**: 3.9+
- **Style**: Follow PEP8 standards strictly
- **Documentation**: Comprehensive docstrings for all classes and functions
- **Type Hints**: Use throughout for better code maintainability
- **Modularity**: Small, focused functions and classes
- **Configurability**: Parameterize all constants and magic numbers

### 2. Data Generation System Implementation

#### Core Data Generation (`scripts/generate_data.py`)

**Main Function Structure:**
```python
def generate_sensor_data() -> pd.DataFrame:
    # 1. Create timeline
    # 2. Generate environmental sensors (temp, humidity, fridge_power)
    # 3. Create Person objects with behavior patterns
    # 4. Generate apartment-level aggregated sensors
    # 5. Return combined dataframe
```

**Configuration Constants:**
- `TIMESTAMP_START`: Start date (e.g., "2020-01-01 00:00")  
- `DAYS`: Duration in days (default: 180)
- `INTERVAL_MINUTES`: Data frequency (default: 1 minute)
- `DATA_PATH`: Output file path

#### Simulation Framework Implementation

**`simulation/func.py` - Core Functions:**

1. **`extract_time_features(timestamps)`**:
   - Converts timestamps to vectorized time features
   - Returns: months, days, hours, weekday, etc. as numpy arrays
   - Used throughout system for time-based calculations

2. **`generate_sydney_temp_from_timestamps(timestamps, seed)`**:
   - Generates realistic temperature data following Sydney climatology
   - Monthly temperature variations with daily cycles
   - Indoor temperature modeling with AC behavior

3. **`generate_humidity_from_timestamps(timestamps, temperature, seed)`**:
   - Correlates humidity with temperature and seasonal patterns
   - Sydney-specific humidity modeling

4. **`generate_fridge_power_from_arrays(timestamps, temperature, humidity, seed)`**:
   - Simulates fridge power consumption cycles
   - Temperature-dependent cooling behavior
   - Realistic idle/active power patterns

**`simulation/person.py` - Person Behavior Modeling:**

```python
class Person:
    def __init__(
        self,
        name: str,
        outside_hours_weekdays: List[Tuple[int, int]],
        outside_hours_weekend: List[Tuple[int, int]], 
        sleep_hours: Tuple[int, int],
        window_open_per_day: List[int],
        window_open_duration: Tuple[int, int],
        door_open_per_day: List[int],
        door_open_duration: Tuple[int, int],
        room_per_day: List[int],
        room_duration: Tuple[int, int],
        seed: Optional[int] = None
    ):
```

**Key Methods:**
- `generate()`: Returns (inside, sleeping, room1, windows, doors) states
- `set_timeline(timestamps)`: Configure person timeline
- Private methods for each behavior type: `_generate_inside()`, `_generate_sleeping()`, etc.

**`simulation/apartment.py` - Multi-Person Aggregation:**

```python
class Apartment:
    def __init__(
        self,
        timestamps: pd.Series,
        persons: List[Person],
        prob_movement: float = 0.2,
        min_duration: int = 1,
        max_duration: int = 10,
        seed: Optional[int] = None
    ):
```

**Key Methods:**
- `generate()`: Generate apartment-level motion patterns
- `get_window()`, `get_door()`, `get_motion()`: Aggregate person states
- `_aggregate_person_state()`: Logical OR aggregation across persons

#### Sensor Data Schema

**Environmental Sensors:**
- `temperature`: Indoor temperature (°C), 18-26°C normal range, Sydney seasonal patterns
- `humidity`: Relative humidity (%), 30-60% normal range, temperature-correlated  
- `fridge_power`: Power consumption (Watts), 15W idle, 50-150W active cycling

**Behavioral Sensors:**
- `sensor_window`: Binary (0=closed, 1=open), person-driven with time constraints
- `sensor_door`: Binary (0=closed, 1=open), person entry/exit patterns
- `sensor_motion`: Binary (0=no motion, 1=motion), aggregated person movement

### 3. Anomaly Injection System (`simulation/anomaly.py`)

**Core Anomaly Class:**
```python
class Anomalies:
    @staticmethod
    def inject_fixed_value_anomalies(
        df: pd.DataFrame,
        column: str, 
        anomaly_col: str,
        num_periods: int = None,
        period_length_range: tuple = (1, 10),
        anomaly_value: float = 0
    ) -> pd.DataFrame:
```

**Anomaly Types Implemented:**
- **Temperature Spikes**: 35-50°C extreme values
- **Humidity Surges**: 95-99% extreme humidity  
- **Fridge Power Loss**: 0-0.5W indicating malfunction
- **Unexpected Motion**: Motion during sleep hours
- **Door/Window Anomalies**: Open during restricted hours (00:30-05:30)

**Implementation Pattern:**
1. Preserve existing anomaly indicators
2. Random anomaly period selection  
3. Configurable duration and intensity
4. Multiple anomaly injection methods for different sensor types

### 4. Anomaly Detection Implementation

#### Statistical Methods
- **Z-Score Analysis**: Rolling window statistical anomaly detection
- **IQR Method**: Interquartile range based outlier detection  
- **Threshold-based**: Domain-specific rule-based detection

#### Machine Learning Approaches  
- **LSTM Autoencoder**: Deep learning temporal anomaly detection

#### Notebook Structure (`notebooks/detect_anomalies.ipynb`)
1. **Data Loading**: Multiple dataset variants (basic/realistic anomalies)
2. **Statistical Analysis**: Z-score, IQR, and threshold methods
3. **ML Model Training**: LSTM implementations  
4. **Evaluation**: Classification metrics
5. **Comparison**: Method performance analysis

### 5. Dependencies & Environment

**Core Requirements (`requirements.txt`):**

### 6. Data Pipeline & Usage

**Data Generation:**
```bash
cd scripts
python generate_data.py
```

**Generated Datasets:**
- `data/synthetic_sensors_2020.csv`: Base dataset (180 days)
- `data/synthetic_sensors_basic_anomalies.csv`: Simple anomaly injection
- `data/synthetic_sensors_realistic_anomalies.csv`: Complex realistic anomalies

**Anomaly Detection Workflow:**
1. Load generated datasets in Jupyter notebooks
2. Apply preprocessing and feature engineering
3. Train statistical and ML models  
4. Evaluate detection performance
5. Generate visualizations and reports
---


## Development Guidelines

### Code Quality Standards
- **Type Safety**: Use type hints consistently throughout
- **Error Handling**: Implement proper exception handling and validation
- **Performance**: Cache expensive computations (timeline generation)
- **Reproducibility**: Seed all random number generators
- **Documentation**: Comprehensive docstrings with parameter descriptions

### Testing & Validation
- **Unit Tests**: Test individual simulation components
- **Integration Tests**: Validate end-to-end data generation pipeline  
- **Data Quality**: Verify generated data ranges and distributions
- **Anomaly Validation**: Confirm anomaly injection correctness

### Configuration Management  
- **Parameterization**: Make all constants configurable
- **Default Values**: Provide sensible defaults for all parameters
- **Flexibility**: Support different simulation scenarios easily
- **Extensibility**: Design for easy addition of new sensor types

### Production Considerations
- **Scalability**: Design for larger datasets and longer time periods
- **Real-time**: Consider streaming data generation capabilities
- **Model Management**: Implement model saving/loading in `models/`
- **Monitoring**: Add data quality and drift detection capabilities

---

## How to Work With AI Assistants

- **Modular Development**: Build components incrementally and test each module
- **Preserve Context**: Maintain detailed comments and docstrings
- **Realistic Modeling**: Focus on plausible real-world scenarios for anomalies
- **Performance Focus**: Optimize for large dataset generation
- **Documentation**: Keep AI transcripts in `ai_transcripts/` for development history

