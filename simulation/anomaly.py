import numpy as np
import random
import pandas as pd


class Anomalies:
    """Class for injecting anomalies into sensor data."""

    @staticmethod
    def inject_fixed_value_anomalies(
        df: pd.DataFrame,
        column: str,
        anomaly_col: str,
        num_periods: int = None,
        period_length_range: tuple = (1, 10),
        anomaly_value: float = 0
    ) -> pd.DataFrame:
        """
        Injects random periods of fixed value anomalies into a specified column of a DataFrame.
        Adds an anomaly indicator column (1 for anomaly, 0 for normal).
        PRESERVES existing anomalies in the anomaly indicator column.

        Args:
            df (pd.DataFrame): The dataframe containing sensor data.
            column (str): Name of the column to inject anomalies into.
            anomaly_col (str): Name of the column to indicate anomaly periods (1=anomaly, 0=normal).
            num_periods (int, optional): Number of anomaly periods to inject. Random [1,30] if None.
            period_length_range (tuple): Min/max length (in minutes) of each anomaly period.
            anomaly_value (float): Value to set during anomaly (default 0).

        Returns:
            pd.DataFrame: DataFrame with modified column and anomaly indicator column.
        """
        result = df.copy()
        s = result[column].copy()

        # Initialize or get existing anomaly indicator column
        if anomaly_col in result.columns:
            anomaly_indicator = result[anomaly_col].copy()
        else:
            anomaly_indicator = pd.Series(0, index=df.index)

        n = len(s)
        if num_periods is None:
            num_periods = random.randint(1, 30)

        for _ in range(num_periods):
            period_length = random.randint(
                period_length_range[0], period_length_range[1])
            start_idx = random.randint(0, n - period_length)
            s.iloc[start_idx:start_idx + period_length] = anomaly_value
            anomaly_indicator.iloc[start_idx:start_idx + period_length] = 1

        result[column] = s
        result[anomaly_col] = anomaly_indicator
        return result

    @staticmethod
    def inject_temperature_drift_anomalies(
        df: pd.DataFrame,
        column: str,
        anomaly_col: str,
        num_periods: int = None,
        period_length_range: tuple = (30, 120),
        drift_magnitude_range: tuple = (3, 8)
    ) -> pd.DataFrame:
        """
        Injects gradual temperature drift anomalies (sensor degradation/calibration issues).

        Args:
            df (pd.DataFrame): The dataframe containing sensor data.
            column (str): Name of the temperature column.
            anomaly_col (str): Name of the anomaly indicator column.
            num_periods (int, optional): Number of drift periods. Random [1,10] if None.
            period_length_range (tuple): Min/max length (in minutes) of each drift period.
            drift_magnitude_range (tuple): Min/max temperature drift in degrees.

        Returns:
            pd.DataFrame: DataFrame with temperature drifts and anomaly indicators.
        """
        result = df.copy()
        s = result[column].copy()

        # Initialize or get existing anomaly indicator column
        if anomaly_col in result.columns:
            anomaly_indicator = result[anomaly_col].copy()
        else:
            anomaly_indicator = pd.Series(0, index=df.index)

        n = len(s)
        if num_periods is None:
            num_periods = random.randint(1, 10)

        for _ in range(num_periods):
            period_length = random.randint(
                period_length_range[0], period_length_range[1])
            start_idx = random.randint(0, n - period_length)

            # Random drift direction and magnitude
            drift_magnitude = random.uniform(
                drift_magnitude_range[0], drift_magnitude_range[1])
            drift_direction = random.choice([-1, 1])

            # Apply gradual drift
            for i in range(period_length):
                if start_idx + i < n:
                    progress = i / period_length  # 0 to 1
                    drift_amount = drift_direction * drift_magnitude * progress
                    s.iloc[start_idx + i] += drift_amount
                    anomaly_indicator.iloc[start_idx + i] = 1

        result[column] = s
        result[anomaly_col] = anomaly_indicator
        return result

    @staticmethod
    def inject_temperature_spike_anomalies(
        df: pd.DataFrame,
        column: str,
        anomaly_col: str,
        num_spikes: int = None,
        spike_duration_range: tuple = (1, 5),
        spike_magnitude_range: tuple = (5, 15)
    ) -> pd.DataFrame:
        """
        Injects sudden temperature spikes (HVAC malfunction, direct sunlight, etc.).

        Args:
            df (pd.DataFrame): The dataframe containing sensor data.
            column (str): Name of the temperature column.
            anomaly_col (str): Name of the anomaly indicator column.
            num_spikes (int, optional): Number of spikes. Random [5,25] if None.
            spike_duration_range (tuple): Min/max duration of each spike in minutes.
            spike_magnitude_range (tuple): Min/max spike magnitude in degrees.

        Returns:
            pd.DataFrame: DataFrame with temperature spikes and anomaly indicators.
        """
        result = df.copy()
        s = result[column].copy()

        if anomaly_col in result.columns:
            anomaly_indicator = result[anomaly_col].copy()
        else:
            anomaly_indicator = pd.Series(0, index=df.index)

        n = len(s)
        if num_spikes is None:
            num_spikes = random.randint(5, 25)

        for _ in range(num_spikes):
            duration = random.randint(
                spike_duration_range[0], spike_duration_range[1])
            start_idx = random.randint(0, n - duration)

            spike_magnitude = random.uniform(
                spike_magnitude_range[0], spike_magnitude_range[1])
            spike_direction = random.choice([-1, 1])

            # Apply spike with gradual onset and decay
            for i in range(duration):
                if start_idx + i < n:
                    # Bell curve for spike shape
                    progress = i / duration
                    intensity = np.exp(-((progress - 0.5) * 6)
                                       ** 2)  # Peak at middle
                    spike_amount = spike_direction * spike_magnitude * intensity
                    s.iloc[start_idx + i] += spike_amount
                    anomaly_indicator.iloc[start_idx + i] = 1

        result[column] = s
        result[anomaly_col] = anomaly_indicator
        return result

    @staticmethod
    def inject_humidity_sudden_change_anomalies(
        df: pd.DataFrame,
        column: str,
        anomaly_col: str,
        num_periods: int = None,
        period_length_range: tuple = (10, 60),
        change_magnitude_range: tuple = (15, 35)
    ) -> pd.DataFrame:
        """
        Injects sudden humidity changes (water leak, ventilation failure, etc.).

        Args:
            df (pd.DataFrame): The dataframe containing sensor data.
            column (str): Name of the humidity column.
            anomaly_col (str): Name of the anomaly indicator column.
            num_periods (int, optional): Number of change periods. Random [3,15] if None.
            period_length_range (tuple): Min/max length of each period in minutes.
            change_magnitude_range (tuple): Min/max humidity change percentage.

        Returns:
            pd.DataFrame: DataFrame with humidity changes and anomaly indicators.
        """
        result = df.copy()
        s = result[column].copy()

        if anomaly_col in result.columns:
            anomaly_indicator = result[anomaly_col].copy()
        else:
            anomaly_indicator = pd.Series(0, index=df.index)

        n = len(s)
        if num_periods is None:
            num_periods = random.randint(3, 15)

        for _ in range(num_periods):
            period_length = random.randint(
                period_length_range[0], period_length_range[1])
            start_idx = random.randint(0, n - period_length)

            change_magnitude = random.uniform(
                change_magnitude_range[0], change_magnitude_range[1])
            change_direction = random.choice([-1, 1])

            # Apply sudden change with some variation
            base_change = change_direction * change_magnitude
            for i in range(period_length):
                if start_idx + i < n:
                    # Add some noise to make it more realistic
                    noise = random.uniform(-2, 2)
                    new_value = s.iloc[start_idx + i] + base_change + noise
                    # Clamp to realistic humidity range
                    s.iloc[start_idx + i] = max(0, min(100, new_value))
                    anomaly_indicator.iloc[start_idx + i] = 1

        result[column] = s
        result[anomaly_col] = anomaly_indicator
        return result

    @staticmethod
    def inject_fridge_power_outage_anomalies(
        df: pd.DataFrame,
        column: str,
        anomaly_col: str,
        num_outages: int = None,
        outage_duration_range: tuple = (5, 180),
        recovery_duration_range: tuple = (10, 30)
    ) -> pd.DataFrame:
        """
        Injects fridge power outage anomalies (power cuts, unplugged, etc.).

        Args:
            df (pd.DataFrame): The dataframe containing sensor data.
            column (str): Name of the fridge power column.
            anomaly_col (str): Name of the anomaly indicator column.
            num_outages (int, optional): Number of outages. Random [2,8] if None.
            outage_duration_range (tuple): Min/max outage duration in minutes.
            recovery_duration_range (tuple): Min/max recovery duration in minutes.

        Returns:
            pd.DataFrame: DataFrame with power outages and anomaly indicators.
        """
        result = df.copy()
        s = result[column].copy()

        if anomaly_col in result.columns:
            anomaly_indicator = result[anomaly_col].copy()
        else:
            anomaly_indicator = pd.Series(0, index=df.index)

        n = len(s)
        if num_outages is None:
            num_outages = random.randint(2, 8)

        for _ in range(num_outages):
            outage_duration = random.randint(
                outage_duration_range[0], outage_duration_range[1])
            recovery_duration = random.randint(
                recovery_duration_range[0], recovery_duration_range[1])
            total_duration = outage_duration + recovery_duration

            start_idx = random.randint(0, n - total_duration)

            # Power outage phase (near zero consumption)
            for i in range(outage_duration):
                if start_idx + i < n:
                    # Minimal standby power
                    s.iloc[start_idx + i] = random.uniform(0, 5)
                    anomaly_indicator.iloc[start_idx + i] = 1

            # Recovery phase (high power consumption as fridge restarts)
            for i in range(recovery_duration):
                if start_idx + outage_duration + i < n:
                    recovery_power = random.uniform(
                        200, 280)  # High startup power
                    s.iloc[start_idx + outage_duration + i] = recovery_power
                    anomaly_indicator.iloc[start_idx + outage_duration + i] = 1

        result[column] = s
        result[anomaly_col] = anomaly_indicator
        return result

    @staticmethod
    def inject_fridge_efficiency_degradation_anomalies(
        df: pd.DataFrame,
        column: str,
        anomaly_col: str,
        num_periods: int = None,
        period_length_range: tuple = (120, 480),
        efficiency_loss_range: tuple = (1.2, 2.0)
    ) -> pd.DataFrame:
        """
        Injects fridge efficiency degradation (dirty coils, door seal issues, etc.).

        Args:
            df (pd.DataFrame): The dataframe containing sensor data.
            column (str): Name of the fridge power column.
            anomaly_col (str): Name of the anomaly indicator column.
            num_periods (int, optional): Number of degradation periods. Random [1,5] if None.
            period_length_range (tuple): Min/max duration in minutes.
            efficiency_loss_range (tuple): Min/max multiplier for power consumption.

        Returns:
            pd.DataFrame: DataFrame with efficiency degradation and anomaly indicators.
        """
        result = df.copy()
        s = result[column].copy()

        if anomaly_col in result.columns:
            anomaly_indicator = result[anomaly_col].copy()
        else:
            anomaly_indicator = pd.Series(0, index=df.index)

        n = len(s)
        if num_periods is None:
            num_periods = random.randint(1, 5)

        for _ in range(num_periods):
            period_length = random.randint(
                period_length_range[0], period_length_range[1])
            start_idx = random.randint(0, n - period_length)

            efficiency_multiplier = random.uniform(
                efficiency_loss_range[0], efficiency_loss_range[1])

            for i in range(period_length):
                if start_idx + i < n:
                    # Only increase power if fridge is actively running (power > 50W)
                    if s.iloc[start_idx + i] > 50:
                        s.iloc[start_idx + i] *= efficiency_multiplier
                    anomaly_indicator.iloc[start_idx + i] = 1

        result[column] = s
        result[anomaly_col] = anomaly_indicator
        return result

    @staticmethod
    def inject_sensor_stuck_anomalies(
        df: pd.DataFrame,
        column: str,
        anomaly_col: str,
        num_periods: int = None,
        period_length_range: tuple = (30, 300),
        stuck_value: int = None
    ) -> pd.DataFrame:
        """
        Injects stuck sensor anomalies for binary sensors (stuck open/closed).

        Args:
            df (pd.DataFrame): The dataframe containing sensor data.
            column (str): Name of the sensor column.
            anomaly_col (str): Name of the anomaly indicator column.
            num_periods (int, optional): Number of stuck periods. Random [2,8] if None.
            period_length_range (tuple): Min/max duration in minutes.
            stuck_value (int, optional): Value to stick at (0 or 1). Random if None.

        Returns:
            pd.DataFrame: DataFrame with stuck sensor values and anomaly indicators.
        """
        result = df.copy()
        s = result[column].copy()

        if anomaly_col in result.columns:
            anomaly_indicator = result[anomaly_col].copy()
        else:
            anomaly_indicator = pd.Series(0, index=df.index)

        n = len(s)
        if num_periods is None:
            num_periods = random.randint(2, 8)

        for _ in range(num_periods):
            period_length = random.randint(
                period_length_range[0], period_length_range[1])
            start_idx = random.randint(0, n - period_length)

            if stuck_value is None:
                current_stuck_value = random.choice([0, 1])
            else:
                current_stuck_value = stuck_value

            for i in range(period_length):
                if start_idx + i < n:
                    s.iloc[start_idx + i] = current_stuck_value
                    anomaly_indicator.iloc[start_idx + i] = 1

        result[column] = s
        result[anomaly_col] = anomaly_indicator
        return result

    @staticmethod
    def inject_sensor_false_trigger_anomalies(
        df: pd.DataFrame,
        column: str,
        anomaly_col: str,
        num_triggers: int = None,
        trigger_duration_range: tuple = (1, 3)
    ) -> pd.DataFrame:
        """
        Injects false trigger anomalies for binary sensors (brief spurious activations).

        Args:
            df (pd.DataFrame): The dataframe containing sensor data.
            column (str): Name of the sensor column.
            anomaly_col (str): Name of the anomaly indicator column.
            num_triggers (int, optional): Number of false triggers. Random [10,50] if None.
            trigger_duration_range (tuple): Min/max duration of each trigger in minutes.

        Returns:
            pd.DataFrame: DataFrame with false triggers and anomaly indicators.
        """
        result = df.copy()
        s = result[column].copy()

        if anomaly_col in result.columns:
            anomaly_indicator = result[anomaly_col].copy()
        else:
            anomaly_indicator = pd.Series(0, index=df.index)

        n = len(s)
        if num_triggers is None:
            num_triggers = random.randint(10, 50)

        for _ in range(num_triggers):
            duration = random.randint(
                trigger_duration_range[0], trigger_duration_range[1])
            start_idx = random.randint(0, n - duration)

            # Only create false trigger if sensor is currently inactive (0)
            if s.iloc[start_idx] == 0:
                for i in range(duration):
                    if start_idx + i < n:
                        s.iloc[start_idx + i] = 1
                        anomaly_indicator.iloc[start_idx + i] = 1

        result[column] = s
        result[anomaly_col] = anomaly_indicator
        return result
