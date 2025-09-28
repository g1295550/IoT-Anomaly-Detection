"""
Synthetic IoT Sensor Data Generator for Smart Home Anomaly Detection

Generates time-series data with realistic daily/weekly patterns.
See README.md for usage and parameter details.
"""
# fmt: off
import sys
from typing import List
import numpy as np
import pandas as pd
from pathlib import Path

# Add simulation module to path
sys.path.append(str(Path(__file__).resolve().parent.parent))

from simulation.func import (
    extract_time_features,
    generate_sydney_temp_from_timestamps,
    generate_humidity_from_timestamps,
    generate_fridge_power_from_arrays,
)
from simulation.person import Person
from simulation.apartment import Apartment
# fmt: on


def create_timeline(start: str, days: int, interval_minutes: int) -> pd.Series:
    """Create a timeline of timestamps."""
    periods = days * 24 * (60 // interval_minutes)
    return pd.date_range(start, periods=periods, freq=f"{interval_minutes}min")


def create_persons(timestamps: pd.Series) -> List[Person]:
    """Instantiate and configure Person objects."""
    person1 = Person(
        name="Alice",
        outside_hours_weekdays=[(8, 17)],
        outside_hours_weekend=[(9, 12), (16, 20)],
        sleep_hours=(23, 6),
        window_open_per_day=[1, 2],
        window_open_duration=(15, 60),
        door_open_per_day=[1, 3],
        door_open_duration=(1, 3),
        room_per_day=[1, 3],
        room_duration=(15, 60),
        seed=123
    )
    person2 = Person(
        name="Bob",
        outside_hours_weekdays=[(7, 16)],
        outside_hours_weekend=[(9, 12), (13, 20)],
        sleep_hours=(22, 6),
        window_open_per_day=[0, 2],
        window_open_duration=(15, 60),
        door_open_per_day=[1, 3],
        door_open_duration=(15, 60),
        room_per_day=[1, 3],
        room_duration=(15, 60),
        seed=23
    )
    for person in [person1, person2]:
        person.set_timeline(timestamps)
    return [person1, person2]


def generate_sensor_data() -> pd.DataFrame:
    """Generate synthetic sensor data for the apartment."""
    timeline = create_timeline(TIMESTAMP_START, DAYS, INTERVAL_MINUTES)
    df = pd.DataFrame({"timestamp": timeline})

    # Environmental sensors
    df["temperature"] = generate_sydney_temp_from_timestamps(
        df["timestamp"], seed=42)
    df["humidity"] = generate_humidity_from_timestamps(
        df["timestamp"], df["temperature"], seed=123)
    df["fridge_power"] = generate_fridge_power_from_arrays(
        df["timestamp"], df["temperature"], df["humidity"], seed=42)

    # Person state features
    persons = create_persons(df["timestamp"])
    for idx, person in enumerate(persons, start=1):
        inside, sleeping, room1, window, door = person.generate()
        # df[f"person{idx}_inside"] = inside
        # df[f"person{idx}_sleeping"] = sleeping
        # df[f"person{idx}_room1_state"] = room1
        # df[f"person{idx}_window_state"] = window
        # df[f"person{idx}_door_state"] = door

    # Apartment-level sensors
    apartment = Apartment(df["timestamp"], persons,
                          prob_movement=0.3, min_duration=1, max_duration=10)
    apartment.generate()
    df["sensor_window"] = apartment.get_window()
    df["sensor_door"] = apartment.get_door()
    df["sensor_motion"] = apartment.get_motion()

    return df


def generate_2020():
    """Main entry point for data generation."""
    df = generate_sensor_data()
    Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic sensor data saved to {DATA_PATH}")


def generate_2025():
    """Main entry point for data generation."""
    df = generate_sensor_data()
    Path(DATA_PATH).parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(DATA_PATH, index=False)
    print(f"Synthetic sensor data saved to {DATA_PATH}")


if __name__ == "__main__":
    DATA_PATH = str(Path(__file__).resolve().parent.parent /
                    "data" / "synthetic_sensors_2020.csv")
    TIMESTAMP_START = "2020-01-01 00:00"
    DAYS = 180
    INTERVAL_MINUTES = 1
    generate_2020()

    DATA_PATH = str(Path(__file__).resolve().parent.parent /
                    "data" / "synthetic_sensors.csv")
    TIMESTAMP_START = "2025-01-01 00:00"
    DAYS = 90
    INTERVAL_MINUTES = 1
    generate_2025()
