from typing import Any
import pandas as pd

import matplotlib.pyplot as plt


def plot_person_states(
    day_number: int,
    timestamps: pd.Series,
    *state_series: pd.Series,
    state_labels: list[str] = None
) -> None:
    """
    Plot one or more person state series for a given day.

    Args:
        day_number (int): 1-based day index.
        timestamps (pd.Series): Series of datetime timestamps.
        *state_series (pd.Series): Any number of Series to plot.
        state_labels (list[str], optional): Labels for each state series.
    """
    start_time = timestamps.dt.normalize().unique()[day_number - 1]
    end_time = start_time + pd.Timedelta(days=1)
    mask = (timestamps >= start_time) & (timestamps < end_time)
    ts_day = timestamps[mask]

    if state_labels is None:
        state_labels = [f"State {i+1}" for i in range(len(state_series))]

    for idx, (series, label) in enumerate(zip(state_series, state_labels)):
        plt.figure(figsize=(15, 2))
        plt.plot(ts_day, series[mask], label=label, alpha=0.7, color=f"C{idx}")
        plt.xlabel("Timestamp")
        plt.ylabel(label)
        plt.title(f"{label} - Day {day_number}")
        plt.tight_layout()
        plt.show()
