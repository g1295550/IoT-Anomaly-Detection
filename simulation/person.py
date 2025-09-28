import numpy as np
import pandas as pd
from simulation.func import extract_time_features
from typing import List, Tuple, Optional


class Person:
    """
    Simulates a person's daily timeline and sensor-related states in a smart home.
    """

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
        seed: Optional[int] = None,
    ):
        self.name = name
        self.outside_hours_weekdays = outside_hours_weekdays
        self.outside_hours_weekend = outside_hours_weekend
        self.sleep_hours = sleep_hours
        self.window_open_per_day = window_open_per_day
        self.window_open_duration = window_open_duration
        self.door_open_per_day = door_open_per_day
        self.door_open_duration = door_open_duration
        self.room_per_day = room_per_day
        self.room_duration = room_duration
        self.rng = np.random.default_rng(seed)
        self.timestamps: Optional[pd.Series] = None
        self._timeline_cache: Optional[Tuple[np.ndarray, ...]] = None

    def set_timeline(self, timestamps: pd.Series) -> None:
        """
        Sets the timeline for the person.
        Args:
            timestamps: pd.Series of datetime objects.
        """
        self.timestamps = timestamps
        self._timeline_cache = None  # Reset cache if timeline changes

    def generate(self) -> Tuple[np.ndarray, ...]:
        """
        Generates and caches the timeline states for the person.
        Returns:
            Tuple of arrays: (inside, sleeping, room1, windows, doors)
        """
        if self._timeline_cache is not None:
            return self._timeline_cache

        if self.timestamps is None:
            raise ValueError(
                "Timestamps must be set using set_timeline before generating timeline.")

        tf = extract_time_features(self.timestamps)
        inside = self._generate_inside(tf)
        sleeping = self._generate_sleeping(tf)
        room1 = self._generate_room1(tf, inside, sleeping)
        windows = self._generate_window_state(tf, inside, sleeping)
        doors = self._generate_door_state(tf, inside, sleeping)
        self._timeline_cache = (inside, sleeping, room1, windows, doors)
        return self._timeline_cache

    def _generate_inside(self, tf: dict) -> np.ndarray:
        """
        Determines if the person is inside for each timestamp.
        Args:
            tf: Time features dict.
        Returns:
            np.ndarray of inside state (0/1).
        """
        inside = [
            0 if self._is_outside(hour, weekday) else 1
            for hour, weekday in zip(tf['hours'], tf['weekday'])
        ]
        return np.array(inside)

    def _is_outside(self, hour: int, weekday: int) -> bool:
        """
        Checks if the person is outside at a given hour and weekday.
        """
        hours_list = (
            self.outside_hours_weekdays if weekday < 5 else self.outside_hours_weekend
        )
        return any(start <= hour < end for (start, end) in hours_list)

    def _generate_sleeping(self, tf: dict) -> np.ndarray:
        """
        Determines if the person is sleeping for each timestamp.
        Args:
            tf: Time features dict.
        Returns:
            np.ndarray of sleeping state (0/1).
        """
        start, end = self.sleep_hours
        sleeping = [
            1 if (start <= hour < end if start <
                  end else hour >= start or hour < end) else 0
            for hour in tf['hours']
        ]
        return np.array(sleeping)

    def _generate_room1(self, tf: dict, inside: np.ndarray, sleeping: np.ndarray) -> np.ndarray:
        """
        Generates room1 occupancy state for the timeline.
        Args:
            tf: Time features dict.
            inside: np.ndarray of inside state (0/1).
            sleeping: np.ndarray of sleeping state (0/1).
        Returns:
            np.ndarray of room1 state (0/1).
        """
        n_minutes = len(tf['hours'])
        room1 = np.zeros(n_minutes, dtype=int)
        if n_minutes == 0:
            return room1

        for day_indices in self._get_day_indices(tf):
            n_periods = self.rng.integers(
                self.room_per_day[0], self.room_per_day[1] + 1)
            for _ in range(n_periods):
                valid = [i for i in day_indices if inside[i]
                         and not sleeping[i]]
                if not valid:
                    continue
                start_idx = self.rng.choice(valid)
                duration = self.rng.integers(
                    self.room_duration[0], self.room_duration[1] + 1)
                end_idx = min(start_idx + duration, day_indices[-1] + 1)
                for i in range(start_idx, end_idx):
                    if i >= n_minutes:
                        break
                    if inside[i] and not sleeping[i]:
                        room1[i] = 1
        return room1

    def _generate_window_state(self, tf: dict, inside: np.ndarray, sleeping: np.ndarray) -> np.ndarray:
        """
        Generates window state (0=closed, 1=open) for the timeline.
        Args:
            tf: Time features dict.
            inside: np.ndarray of inside state (0/1).
            sleeping: np.ndarray of sleeping state (0/1).
        Returns:
            np.ndarray of window state (0/1).
        """
        n = len(self.timestamps)
        window_state = np.zeros(n, dtype=int)
        for day_indices in self._get_day_indices(tf):
            n_opens = self.rng.integers(
                self.window_open_per_day[0], self.window_open_per_day[1] + 1)
            for _ in range(n_opens):
                valid = [i for i in day_indices if inside[i]
                         and not sleeping[i]]
                if not valid:
                    continue
                start_idx = self.rng.choice(valid)
                duration = self.rng.integers(
                    self.window_open_duration[0], self.window_open_duration[1] + 1)
                for i in range(start_idx, min(start_idx + duration, day_indices[-1] + 1)):
                    if not inside[i] or sleeping[i]:
                        break
                    window_state[i] = 1
        return window_state

    def _generate_door_state(self, tf: dict, inside: np.ndarray, sleeping: np.ndarray) -> np.ndarray:
        """
        Generates door state (0=closed, 1=open) for the timeline.
        Args:
            tf: Time features dict.
            inside: np.ndarray of inside state (0/1).
            sleeping: np.ndarray of sleeping state (0/1).
        Returns:
            np.ndarray of door state (0/1).
        """
        n = len(self.timestamps)
        door_state = np.zeros(n, dtype=int)
        # Door opens on inside state change (entry/exit)
        door_state[1:] = (inside[1:] != inside[:-1]).astype(int)
        # Random door opens per day (when inside and not sleeping)
        for day_indices in self._get_day_indices(tf):
            n_opens = self.rng.integers(
                self.door_open_per_day[0], self.door_open_per_day[1] + 1)
            for _ in range(n_opens):
                valid = [i for i in day_indices if inside[i]
                         and not sleeping[i]]
                if not valid:
                    continue
                start_idx = self.rng.choice(valid)
                duration = self.rng.integers(
                    self.door_open_duration[0], self.door_open_duration[1] + 1)
                for i in range(start_idx, min(start_idx + duration, day_indices[-1] + 1)):
                    if i >= n:
                        break
                    door_state[i] = 1
        return door_state

    def _get_day_indices(self, tf: dict) -> List[np.ndarray]:
        """
        Helper to get indices for each day in the timeline.
        Args:
            tf: Time features dict.
        Returns:
            List of np.ndarray of indices for each day.
        """
        days = np.unique(tf['dates'])
        return [np.where(tf['dates'] == day)[0] for day in days if np.any(tf['dates'] == day)]

    def get_room1_state(self) -> Optional[np.ndarray]:
        """Returns room1 state if generated."""
        return self._timeline_cache[2] if self._timeline_cache else None

    def get_window_state(self) -> Optional[np.ndarray]:
        """Returns window state if generated."""
        return self._timeline_cache[3] if self._timeline_cache else None

    def get_door_state(self) -> Optional[np.ndarray]:
        """Returns door state if generated."""
        return self._timeline_cache[4] if self._timeline_cache else None
