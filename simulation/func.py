import numpy as np
import pandas as pd
from typing import Union, Optional, Dict


def extract_time_features(
    timestamps: Union[pd.Series, pd.DatetimeIndex]
) -> Dict[str, np.ndarray]:
    """
    Converts a pd.Series or pd.DatetimeIndex of timestamps into
    NumPy arrays of time features for vectorized calculations.

    Returns a dict with:
        - 'months': 1–12
        - 'days': 1–31
        - 'days_in_month': 28–31
        - 'day_of_year': 1–366
        - 'hours': fractional hours (0–23.999)
        - 'minutes': 0–59
        - 'seconds': 0–59
        - 'weekday': 0 (Mon) – 6 (Sun)
        - 'dates': normalized datetime (midnight) for each timestamp
    """
    timestamps = pd.to_datetime(timestamps)

    if isinstance(timestamps, pd.Series):
        months = timestamps.dt.month.to_numpy()
        days = timestamps.dt.day.to_numpy()
        days_in_month = timestamps.dt.days_in_month.to_numpy()
        day_of_year = timestamps.dt.dayofyear.to_numpy()
        hours = (
            timestamps.dt.hour.to_numpy()
            + timestamps.dt.minute.to_numpy() / 60
            + timestamps.dt.second.to_numpy() / 3600
        )
        minutes = timestamps.dt.minute.to_numpy()
        seconds = timestamps.dt.second.to_numpy()
        weekday = timestamps.dt.weekday.to_numpy()
        dates = timestamps.dt.normalize()
    else:  # DatetimeIndex
        months = timestamps.month
        days = timestamps.day
        days_in_month = timestamps.days_in_month
        day_of_year = timestamps.dayofyear
        hours = (
            timestamps.hour
            + timestamps.minute / 60
            + timestamps.second / 3600
        )
        minutes = timestamps.minute
        seconds = timestamps.second
        weekday = timestamps.weekday
        dates = timestamps.normalize()

    return {
        'months': months,
        'days': days,
        'days_in_month': days_in_month,
        'day_of_year': day_of_year,
        'hours': hours,
        'minutes': minutes,
        'seconds': seconds,
        'weekday': weekday,
        'dates': dates
    }


# --- Sydney monthly temperature climatology (min, avg, max °C) ---
SYDNEY_MONTHLY_TEMP = np.array([
    [18.0, 23.5, 27.0],  # Jan
    [18.0, 23.5, 27.0],  # Feb
    [16.5, 22.5, 25.5],  # Mar
    [13.5, 19.5, 23.0],  # Apr
    [10.5, 16.5, 20.0],  # May
    [8.0, 14.0, 17.5],   # Jun
    [7.0, 13.0, 16.5],   # Jul
    [8.5, 14.5, 17.5],   # Aug
    [10.5, 17.0, 20.5],  # Sep
    [13.0, 19.5, 23.0],  # Oct
    [15.0, 21.0, 24.5],  # Nov
    [16.5, 22.5, 26.0],  # Dec
])


def generate_sydney_temp_from_timestamps(
    timestamps: Union[pd.Series, pd.DatetimeIndex],
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate indoor temperature series (°C) aligned with given timestamps,
    using Sydney monthly min/avg/max climatology.

    Parameters
    ----------
    timestamps : pd.Series or pd.DatetimeIndex
        Array of datetime values.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Indoor temperature values aligned with timestamps.
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(timestamps)
    tf = extract_time_features(timestamps)
    months = tf['months']
    hours = tf['hours']
    days = tf['dates']

    outdoor_min = SYDNEY_MONTHLY_TEMP[months - 1, 0]
    outdoor_avg = SYDNEY_MONTHLY_TEMP[months - 1, 1]
    outdoor_max = SYDNEY_MONTHLY_TEMP[months - 1, 2]

    unique_days = pd.Series(days).unique()
    day_fluctuations = np.random.uniform(-2.0, 2.0, size=len(unique_days))
    fluct_map = dict(zip(unique_days, day_fluctuations))
    daily_fluctuation = pd.Series(days).map(fluct_map).to_numpy()

    daily_midpoint = outdoor_avg + daily_fluctuation
    daily_amplitude = (outdoor_max - outdoor_min) / 2
    outdoor_temp = daily_midpoint + daily_amplitude * \
        np.sin(np.pi * (hours - 8) / 12)

    comfort_setpoint = 22.0
    indoor_temp = comfort_setpoint + 0.3 * (outdoor_temp - comfort_setpoint)
    indoor_temp += np.random.uniform(-1.0, 1.0, size=n_samples)

    return np.clip(np.round(indoor_temp, 2), 15.0, 30.0)


# --- Monthly base humidity for Sydney (approx %) ---
MONTHLY_BASE_HUMIDITY = np.array([
    72.0,  # Jan
    72.0,  # Feb
    68.0,  # Mar
    65.0,  # Apr
    62.0,  # May
    58.0,  # Jun
    56.0,  # Jul
    58.0,  # Aug
    62.0,  # Sep
    66.0,  # Oct
    68.0,  # Nov
    72.0,  # Dec
])


def generate_humidity_from_timestamps(
    timestamps: Union[pd.Series, pd.DatetimeIndex],
    temperatures: np.ndarray,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Generate indoor humidity (%RH) based on:
    - Monthly seasonal averages
    - Smooth daily cycle (inverse to temperature)
    - Correlation with temperature
    - Random noise

    Parameters
    ----------
    timestamps : pd.Series or pd.DatetimeIndex
        Array of datetime values.
    temperatures : np.ndarray
        Matching indoor temperature array.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Humidity values aligned with timestamps.
    """
    if seed is not None:
        np.random.seed(seed)

    n_samples = len(timestamps)
    tf = extract_time_features(timestamps)
    months = tf['months']
    hours = tf['hours']
    days = tf['dates']
    days_in_month = tf['days_in_month']

    current_month_base = MONTHLY_BASE_HUMIDITY[months - 1]
    next_month_base = MONTHLY_BASE_HUMIDITY[months % 12]
    frac_of_month = (pd.Series(days).dt.day - 1) / days_in_month
    base_humidity = current_month_base + \
        (next_month_base - current_month_base) * frac_of_month

    daily_amplitude = 5.0
    daily_cycle = daily_amplitude * np.sin(np.pi * (hours - 20) / 12)

    comfort_temp = 22.0
    temp_humidity_offset = (temperatures - comfort_temp) * -1.5

    target_humidity = base_humidity + daily_cycle + temp_humidity_offset
    noise = np.random.uniform(-3.0, 3.0, size=n_samples)
    humidity = target_humidity + noise

    humidity = np.clip(humidity, 30.0, 90.0)
    return np.round(humidity, 2)


def generate_fridge_power_from_arrays(
    timestamps: Union[pd.Series, pd.DatetimeIndex],
    temperatures: np.ndarray,
    humidities: Optional[np.ndarray] = None,
    seed: Optional[int] = None
) -> np.ndarray:
    """
    Simulate fridge power consumption (W) based on compressor cycles,
    temperature, humidity, and seasonal effects.

    Parameters
    ----------
    timestamps : pd.Series or pd.DatetimeIndex
        Array of datetime values.
    temperatures : np.ndarray
        Matching indoor temperature array.
    humidities : np.ndarray or None
        Optional humidity array.
    seed : int or None
        Random seed for reproducibility.

    Returns
    -------
    np.ndarray
        Fridge power values aligned with timestamps.
    """
    if seed is not None:
        np.random.seed(seed)

    tf = extract_time_features(timestamps)
    n_samples = len(timestamps)
    minutes = tf['hours'] * 60 + tf['minutes']
    cycle_position = minutes % 60

    power = np.zeros(n_samples)

    # Compressor ON for first 20 minutes
    on_phase = cycle_position < 20
    t_norm = cycle_position[on_phase] / 20  # 0 to 1

    peak_power = 140  # peak at start
    decay_slope = 40   # drops 40W over ON phase

    power[on_phase] = (
        peak_power
        - decay_slope * t_norm
        + np.random.uniform(-3, 3, on_phase.sum())
    )

    # Compressor OFF phase
    power[~on_phase] = 10 + np.random.uniform(-2, 2, (~on_phase).sum())

    # Seasonal factor on ON phase
    day_of_year = tf['day_of_year']
    month_angle = 2 * np.pi * day_of_year / 365.0
    seasonal_factor = 1.0 + 0.3 * np.sin(month_angle - np.pi / 2)
    power[on_phase] *= seasonal_factor[on_phase]

    # Temperature effect on ON phase
    temp_offset = np.maximum(temperatures - 4.0, 0)
    power[on_phase] += temp_offset[on_phase] * 5.0

    # Optional humidity effect
    if humidities is not None:
        humidity_effect = np.clip((humidities - 50) * 0.2, 0, 10)
        power[on_phase] += humidity_effect[on_phase]

    return np.round(power, 2)
