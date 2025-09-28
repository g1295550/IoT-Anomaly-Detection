import numpy as np
import pandas as pd
from typing import List, Optional
from simulation.person import Person


class Apartment:
    """
    Aggregates window, door, and motion states for all persons in the apartment.
    """

    def __init__(
        self,
        timestamps: pd.Series,
        persons: Optional[List[Person]] = None,
        prob_movement: float = 0.2,
        min_duration: int = 1,
        max_duration: int = 10,
        seed: Optional[int] = None
    ):
        """
        Initialize Apartment.

        Args:
            timestamps (pd.Series): Time indices for simulation.
            persons (List[Person], optional): List of Person objects.
            prob_movement (float): Probability of movement per time step.
            min_duration (int): Minimum duration of a motion event.
            max_duration (int): Maximum duration of a motion event.
            seed (int, optional): Random seed for reproducibility.
        """
        self.timestamps = timestamps
        self.persons = persons if persons is not None else []
        self.prob_movement = prob_movement
        self.min_duration = min_duration
        self.max_duration = max_duration
        self.seed = seed

        self._motion_state: Optional[np.ndarray] = None

        if self.seed is not None:
            np.random.seed(self.seed)

    def generate(self) -> None:
        """Generate apartment-level motion state."""
        self._motion_state = self._generate_motion()

    def get_window(self) -> np.ndarray:
        """
        Aggregate window open states across all persons.

        Returns:
            np.ndarray: Binary array, 1 if any window is open at each timestamp.
        """
        return self._aggregate_person_state(lambda p: p.get_window_state())

    def get_door(self) -> np.ndarray:
        """
        Aggregate door open states across all persons.

        Returns:
            np.ndarray: Binary array, 1 if any door is open at each timestamp.
        """
        return self._aggregate_person_state(lambda p: p.get_door_state())

    def get_motion(self) -> Optional[np.ndarray]:
        """
        Get apartment-level motion state.

        Returns:
            Optional[np.ndarray]: Binary array, 1 if motion detected at each timestamp.
        """
        return self._motion_state

    def _aggregate_person_state(self, state_func) -> np.ndarray:
        """
        Aggregate a binary state across all persons using logical OR.

        Args:
            state_func (Callable): Function to get state from a person.

        Returns:
            np.ndarray: Aggregated binary state.
        """
        state = np.zeros(len(self.timestamps), dtype=int)
        for person in self.persons:
            person_state = state_func(person)
            state = np.logical_or(state, person_state)
        return state.astype(int)

    def _generate_motion(self) -> np.ndarray:
        """
        Generate apartment-level motion based on room occupancy and random movement periods.

        Returns:
            np.ndarray: Binary array, 1 if motion detected at each timestamp.
        """
        motion = np.zeros(len(self.timestamps), dtype=int)
        for person in self.persons:
            person_motion = self._simulate_person_motion(person)
            motion = np.logical_or(motion, person_motion)
        return motion.astype(int)

    def _simulate_person_motion(self, person: Person) -> np.ndarray:
        """
        Simulate motion events for a single person.

        Args:
            person (Person): Person object.

        Returns:
            np.ndarray: Binary motion array for the person.
        """
        room1_state = person.get_room1_state()
        person_motion = np.zeros(len(self.timestamps), dtype=int)
        i = 0
        while i < len(self.timestamps):
            if room1_state[i] > 0 and np.random.rand() < self.prob_movement:
                duration = np.random.randint(
                    self.min_duration, self.max_duration + 1)
                end_idx = min(i + duration, len(self.timestamps))
                person_motion[i:end_idx] = 1
                i = end_idx
            else:
                i += 1
        return person_motion.astype(int)
