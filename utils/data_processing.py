import pandas as pd
import numpy as np

# TODO: add the blink detector to the EyeTrackingProcessor class 

class EyeTrackingProcessor:
    """
    A class for processing eye-tracking and mouse movement data from TSV files.
    It provides functionalities for reading, cleaning, resampling, and chunking tasks.
    """

    # ------------------------- 1. DATA LOADING & VALIDATION -------------------------
    
    def read_tsv(self, path: str) -> pd.DataFrame:
        """Read a TSV file and validate task labeling."""
        df = pd.read_csv(path, sep='\t')
        
        # Validate task counts
        expected_tasks = [f"Task {i}" for i in range(1, 7)] + [f"Task {i} end" for i in range(1, 7)]
        task_counts = df['Event'].value_counts()
        
        for task in expected_tasks:
            if task_counts.get(task, 0) != 6:
                print(f"Warning: {task} in {path} has {task_counts.get(task, 0)} occurrences instead of 6.")
        
        return df

    def load_data(self, paths: list[str]) -> list[pd.DataFrame]:
        """Load multiple TSV files into a list of DataFrames."""
        return [self.read_tsv(path) for path in paths]

    # ------------------------- 2. TASK IDENTIFICATION & FEATURE EXTRACTION -------------------------

    def task_range_finder(self, df: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
        """Find start and end times of tasks within a DataFrame."""
        event_df = df[df['Event'].str.contains('Task', na=False)].sort_values(by="Recording timestamp")
        task_ranges = {}
        task_stack = {}

        for _, row in event_df.iterrows():
            event, timestamp = row["Event"], row["Recording timestamp"]
            if "end" not in event:
                task_stack[event] = timestamp
            else:
                task_type = event.replace(" end", "")
                if task_type in task_stack:
                    task_ranges.setdefault(task_type, []).append((task_stack.pop(task_type), timestamp))

        return task_ranges

    def get_features(self, dfs: list[pd.DataFrame], tasks: list[str], features: list[str]) -> pd.DataFrame:
        """Extract features for tasks from multiple DataFrames."""
        full_dataset = []
        for df in dfs:
            sub_dataset = []
            task_ranges = self.task_range_finder(df)
            for task, periods in task_ranges.items():
                for i, (start, end) in enumerate(periods):
                    task_data = df.loc[(df["Recording timestamp"] >= start) & (df["Recording timestamp"] <= end), features].copy()
                    task_data["Task_id"] = int(task[-1])
                    task_data["Task_execution"] = i
                    sub_dataset.append(task_data)
            if sub_dataset:
                full_dataset.append(pd.concat(sub_dataset))
        return pd.concat(full_dataset, ignore_index=True)
    
    # ------------------------- 3. BLINK IDENTIFICATION -------------------------
    def detect_blinks(self, df: pd.DataFrame, blink_threshold: int= 1e5)-> pd.DataFrame:
        df = df.sort_values(by=["Participant name", "Task_id", "Task_execution", "Recording timestamp"]).reset_index(drop=True)

        # Identify rows where gaze data is missing
        df["Missing Gaze"] = df["Gaze point X"].isna() | df["Gaze point Y"].isna()
        df["Time Diff"] = df["Recording timestamp"].diff()

        # Identify blink start (when missing gaze starts) and blink end (when gaze reappears)
        df["Blink Start"] = df["Missing Gaze"] & ~df["Missing Gaze"].shift(1, fill_value=False)
        df["Blink End"] = ~df["Missing Gaze"] & df["Missing Gaze"].shift(1, fill_value=False)

        # Assign blink IDs
        df["Blink ID"] = df["Blink Start"].cumsum()

        # Compute blink durations
        blink_durations = df.groupby("Blink ID")["Time Diff"].sum().reset_index()
        blinks_detected = blink_durations[blink_durations["Time Diff"] > blink_threshold]

        # Create a binary blink mask column in the original dataframe
        df["Blink"] = df["Blink ID"].isin(blinks_detected["Blink ID"]).astype(int)

        # Drop unnecessary intermediate columns
        df.drop(columns=["Missing Gaze", "Time Diff", "Blink Start", "Blink End", "Blink ID"], inplace=True)
        
        return df, blinks_detected

    # ------------------------- 4. DATA RESAMPLING -------------------------

    def resample_task(self, df: pd.DataFrame, interpolate_cols: list[str], new_timestamps: np.ndarray) -> pd.DataFrame:
        """Resample task data to a given set of timestamps."""
        resampled_df = pd.DataFrame({'Recording timestamp': new_timestamps})

        for col in interpolate_cols:
            if col in df.columns:
                resampled_df[col] = np.interp(new_timestamps, df['Recording timestamp'], df[col])

        return resampled_df

    def resample_tasks_fixed_time(self, df: pd.DataFrame, interpolate_cols: list[str], timestep: float = 0.001) -> pd.DataFrame:
        """Resample all tasks to a fixed time step."""
        resampled_tasks = []
        unique_tasks = df[['Participant name', 'Task_id', 'Task_execution']].drop_duplicates()

        for _, row in unique_tasks.iterrows():
            subset = df[(df["Participant name"] == row["Participant name"]) &
                        (df["Task_id"] == row["Task_id"]) & 
                        (df["Task_execution"] == row["Task_execution"])]

            if not subset.empty:
                min_time, max_time = subset['Recording timestamp'].min(), subset['Recording timestamp'].max()
                new_timestamps = np.arange(min_time, max_time, timestep * 1e6)
                resampled = self.resample_task(subset, interpolate_cols, new_timestamps)
                resampled["Participant name"], resampled["Task_id"], resampled["Task_execution"] = row
                resampled_tasks.append(resampled)

        return pd.concat(resampled_tasks, ignore_index=True)

    def resample_tasks_fixed_points(self, df: pd.DataFrame, interpolate_cols: list[str], num_points: int = 1000) -> pd.DataFrame:
        """Resample tasks to a fixed number of points."""
        resampled_tasks = []
        unique_tasks = df[['Participant name', 'Task_id', 'Task_execution']].drop_duplicates()

        for _, row in unique_tasks.iterrows():
            subset = df[(df["Participant name"] == row["Participant name"]) &
                        (df["Task_id"] == row["Task_id"]) & 
                        (df["Task_execution"] == row["Task_execution"])].sort_values("Recording timestamp")

            if not subset.empty:
                original_indices = np.linspace(0, 1, len(subset))
                new_indices = np.linspace(0, 1, num_points)
                resampled = pd.DataFrame({'Recording timestamp': np.interp(new_indices, original_indices, subset['Recording timestamp'])})

                for col in interpolate_cols:
                    if col in subset.columns:
                        resampled[col] = np.interp(new_indices, original_indices, subset[col])

                resampled["Participant name"], resampled["Task_id"], resampled["Task_execution"] = row
                resampled_tasks.append(resampled)

        return pd.concat(resampled_tasks, ignore_index=True)

    # ------------------------- 4. DATA CLEANING & PADDING -------------------------

    def apply_outside_screen_placeholder(self, df: pd.DataFrame, x_lim: tuple[int, int], y_lim: tuple[int, int], placeholder_value=np.nan) -> pd.DataFrame:
        """Replace values outside screen boundaries with a placeholder."""
        for axis, limits in zip(["X", "Y"], [x_lim, y_lim]):
            for col in [f"Gaze point {axis}", f"Mouse position {axis}"]:
                df.loc[(df[col] < limits[0]) | (df[col] > limits[1]), col] = placeholder_value
        return df

    # def pad_tasks(self, df: pd.DataFrame, pad_value=np.nan) -> pd.DataFrame:
    #     """Pad tasks to match the longest task length."""
    #     max_len = df.groupby(["Participant name", "Task_id", "Task_execution"]).size().max()
    #     padded_data = df.groupby(["Participant name", "Task_id", "Task_execution"]).apply(
    #         lambda group: group.reindex(range(max_len), fill_value=pad_value)
    #     ).reset_index(drop=True)
    #     return padded_data
    
    def pad_tasks(self, df: pd.DataFrame, pad_value=np.nan) -> pd.DataFrame:
        """Pad tasks to match the longest task length while keeping metadata."""
        max_len = df.groupby(["Participant name", "Task_id", "Task_execution"]).size().max()
        
        padded_dfs = []
        for (participant, task, execution), group in df.groupby(["Participant name", "Task_id", "Task_execution"]):
            current_len = len(group)
            pad_needed = max_len - current_len  # How many rows to add
            
            if pad_needed > 0:
                # Create a DataFrame for the padding
                pad_df = pd.DataFrame({
                    "Participant name": [participant] * pad_needed,
                    "Task_id": [task] * pad_needed,
                    "Task_execution": [execution] * pad_needed,
                })
                
                # Fill all other feature columns with `pad_value`
                for col in df.columns:
                    if col not in ["Participant name", "Task_id", "Task_execution"]:
                        pad_df[col] = pad_value
                
                # Concatenate original and padded data
                group = pd.concat([group, pad_df], ignore_index=True)
            
            padded_dfs.append(group)
        
        # Combine all padded groups
        padded_data = pd.concat(padded_dfs, ignore_index=True)
        
        return padded_data
    
    

class GazeMetricsProcessor:
    """
    A class for computing gaze-related metrics from eye-tracking data.
    It calculates fixation statistics, saccades, velocity, acceleration, blink rate, and gaze dispersion.
    """

    def __init__(self, df):
        """Initialize with a DataFrame containing a single task execution."""
        self.df = df.sort_values(by="Recording timestamp").reset_index(drop=True)

    # ------------------------- FIXATION COMPUTATION -------------------------
    
    def compute_fixation_statistics(self, time_threshold=100000, radius_threshold=50):
        """Compute fixation count, total duration, and average duration with a time constraint."""
        dx = self.df["Gaze point X"].diff().fillna(0)
        dy = self.df["Gaze point Y"].diff().fillna(0)
        displacement = np.sqrt(dx**2 + dy**2)

        fixation_mask = displacement < radius_threshold
        fixation_start_time = self.df["Recording timestamp"].where(fixation_mask & ~fixation_mask.shift(1, fill_value=False))
        fixation_end_time = self.df["Recording timestamp"].where(~fixation_mask & fixation_mask.shift(1, fill_value=False))

        fixation_start_time = fixation_start_time.dropna().to_numpy()
        fixation_end_time = fixation_end_time.dropna().to_numpy()

        min_len = min(len(fixation_start_time), len(fixation_end_time))
        fixation_durations = (fixation_end_time[:min_len] - fixation_start_time[:min_len])

        valid_fixations = fixation_durations >= time_threshold
        fixation_count = valid_fixations.sum()
        total_fixation_duration = fixation_durations[valid_fixations].sum() / 1e6  # Convert to seconds
        avg_fixation_duration = total_fixation_duration / fixation_count if fixation_count > 0 else 0

        return fixation_count, total_fixation_duration, avg_fixation_duration

    # ------------------------- SACCADE COMPUTATION -------------------------

    def compute_saccade_statistics(self, radius_threshold=50):
        """Compute saccade count, amplitude, and velocity."""
        dx = self.df["Gaze point X"].diff().fillna(0)
        dy = self.df["Gaze point Y"].diff().fillna(0)
        displacement = np.sqrt(dx**2 + dy**2)
        time_diff = self.df["Recording timestamp"].diff().fillna(0) / 1e6  # Convert to seconds

        saccade_mask = displacement >= radius_threshold
        saccade_count = saccade_mask.sum()
        avg_saccade_amplitude = displacement[saccade_mask].mean() if saccade_count > 0 else 0
        avg_saccade_velocity = (displacement[saccade_mask] / time_diff.replace(0, np.nan)).mean() if saccade_count > 0 else 0

        return saccade_count, avg_saccade_amplitude, avg_saccade_velocity

    # ------------------------- VELOCITY & ACCELERATION COMPUTATION -------------------------

    def compute_velocity_acceleration(self):
        """Compute gaze velocity and acceleration."""
        dx = self.df["Gaze point X"].diff().fillna(0)
        dy = self.df["Gaze point Y"].diff().fillna(0)
        displacement = np.sqrt(dx**2 + dy**2)
        time_diff = self.df["Recording timestamp"].diff().fillna(0) / 1e6  # Convert to seconds

        velocity = displacement / time_diff.replace(0, np.nan)
        acceleration = velocity.diff().fillna(0) / time_diff.replace(0, np.nan)

        avg_velocity = velocity.mean()
        avg_acceleration = acceleration.mean()

        return avg_velocity, avg_acceleration

    # ------------------------- BLINK RATE COMPUTATION -------------------------

    def compute_blink_rate(self):
        """Compute blink rate (blinks per second)."""
        total_time = (self.df["Recording timestamp"].max() - self.df["Recording timestamp"].min()) / 1e6  # Convert to seconds
        blink_count = self.df["Blink"].sum()
        blink_rate = blink_count / total_time if total_time > 0 else 0
        return blink_rate

    # ------------------------- GAZE DISPERSION COMPUTATION -------------------------

    def compute_gaze_dispersion(self):
        """Compute gaze dispersion using bounding box area."""
        if len(self.df) > 3:
            gaze_dispersion = (self.df["Gaze point X"].max() - self.df["Gaze point X"].min()) * \
                              (self.df["Gaze point Y"].max() - self.df["Gaze point Y"].min())
        else:
            gaze_dispersion = 0
        return gaze_dispersion

    # ------------------------- COMPUTE ALL METRICS PER TASK -------------------------

    def compute_all_metrics(self):
        """Compute all gaze/mouse-related metrics per task execution."""
        fixation_count, total_fix_duration, avg_fix_duration = self.compute_fixation_statistics()
        saccade_count, avg_saccade_amp, avg_saccade_vel = self.compute_saccade_statistics()
        avg_velocity, avg_acceleration = self.compute_velocity_acceleration()
        blink_rate = self.compute_blink_rate()
        gaze_dispersion = self.compute_gaze_dispersion()

        return {
            "Fixation Count": fixation_count,
            "Total Fixation Duration (s)": total_fix_duration,
            "Avg Fixation Duration (s)": avg_fix_duration,
            "Saccade Count": saccade_count,
            "Avg Saccade Amplitude (px)": avg_saccade_amp,
            "Avg Saccade Velocity (px/s)": avg_saccade_vel,
            "Avg Gaze Velocity (px/s)": avg_velocity,
            "Avg Gaze Acceleration (px/s²)": avg_acceleration,
            "Blink Rate (blinks/s)": blink_rate,
            "Gaze Dispersion (area)": gaze_dispersion
        }

class MouseMetricsProcessor:
    """
    A class for computing mouse movement-related metrics, including velocity, acceleration,
    movement frequency, idle times, click counts, path patterns, total distance, stops, 
    average speed, and movement bursts.
    """

    def __init__(self, df):
        """Initialize with a DataFrame containing a single task execution."""
        self.df = df.sort_values(by="Recording timestamp").reset_index(drop=True)

    # ------------------------- VELOCITY & ACCELERATION COMPUTATION -------------------------
    
    def compute_velocity_acceleration(self):
        """Compute mouse velocity and acceleration."""
        dx = self.df["Mouse position X"].diff().fillna(0)
        dy = self.df["Mouse position Y"].diff().fillna(0)
        displacement = np.sqrt(dx**2 + dy**2)
        time_diff = self.df["Recording timestamp"].diff().fillna(0) / 1e6  # Convert to seconds

        velocity = displacement / time_diff.replace(0, np.nan)
        acceleration = velocity.diff().fillna(0) / time_diff.replace(0, np.nan)

        avg_velocity = velocity.mean()
        avg_acceleration = acceleration.mean()

        return avg_velocity, avg_acceleration

    # ------------------------- FREQUENCY OF MOVEMENTS & IDLE TIMES -------------------------

    def compute_movement_frequency_idle_time(self, idle_threshold=0.5):
        """
        Compute movement frequency (how often the mouse moves) and idle times (when the mouse doesn't move).

        Args:
            idle_threshold (float): Time in seconds considered as idle if no movement happens.

        Returns:
            movement_frequency (movements per second), idle_time (total idle duration in seconds)
        """
        time_diff = self.df["Recording timestamp"].diff().fillna(0) / 1e6  # Convert to seconds
        movement_mask = (self.df["Mouse position X"].diff().abs() > 0) | (self.df["Mouse position Y"].diff().abs() > 0)

        movement_frequency = movement_mask.sum() / time_diff.sum()
        
         # Identify idle periods (where mouse does not move for at least idle_threshold seconds)
        idle_periods = time_diff[~movement_mask]  # Time differences where no movement happened
        total_idle_time = idle_periods[idle_periods >= idle_threshold].sum()  # Sum only idle periods longer than the threshold


        return movement_frequency, total_idle_time

    # ------------------------- CLICK / KEYBOARD COUNT COMPUTATION -------------------------

    def compute_click_count(self):
        """Compute the number of mouse clicks (if Event data is available)."""
        if "Event" in self.df.columns:
            if "MouseEvent" in self.df["Event"].value_counts():
                return self.df["Event"].value_counts()["MouseEvent"].item()
        return 0  # Default if click data isn't present
    
    def compute_keyboard_count(self):
        """Compute the number of keyboard clicks (if Event data is available)."""
        if "Event" in self.df.columns:
            if "KeyboardEvent" in self.df["Event"].value_counts():
                return self.df["Event"].value_counts()["KeyboardEvent"].item()
        return 0  # Default if keyboard data isn't present

    # ------------------------- PATH PATTERNS & DIRECTION CHANGES -------------------------

    def compute_path_patterns(self, angle_threshold:float=30):
        """
        Compute path complexity: number of direction changes in mouse movement.
        Args:
            angle_threshold (float): Threshold angle considered for significant direction changes

        Returns:
            direction_changes (int): Count of significant changes in movement direction.
        """
        dx = self.df["Mouse position X"].diff().fillna(0)
        dy = self.df["Mouse position Y"].diff().fillna(0)
        angles = np.arctan2(dy, dx)
        direction_changes = np.sum(np.abs(np.diff(angles)) > np.radians(angle_threshold))  # Count significant direction changes

        return direction_changes

    # ------------------------- TOTAL DISTANCE TRAVELED & NUMBER OF STOPS -------------------------

    def compute_total_distance_and_stops(self, stop_threshold=5):
        """
        Compute the total distance traveled by the mouse and the number of times the mouse stops.

        Args:
            stop_threshold (int): Speed (px/s) below which the movement is considered a stop.

        Returns:
            total_distance (float): Total distance traveled by the mouse.
            num_stops (int): Number of times mouse movement speed drops below threshold.
        """
        dx = self.df["Mouse position X"].diff().fillna(0)
        dy = self.df["Mouse position Y"].diff().fillna(0)
        displacement = np.sqrt(dx**2 + dy**2)
        time_diff = self.df["Recording timestamp"].diff().fillna(0) / 1e6  # Convert to seconds
        velocity = displacement / time_diff.replace(0, np.nan)

        total_distance = displacement.sum()
        num_stops = (velocity < stop_threshold).sum()

        return total_distance, num_stops

    # ------------------------- BURSTS OF MOVEMENT VS STILLNESS -------------------------

    def compute_movement_bursts(self, burst_threshold=50, stillness_threshold=5):
        """
        Identify periods of bursts (rapid movements) vs stillness (very slow movement).

        Args:
            burst_threshold (int): Speed above which a movement is considered a burst.
            stillness_threshold (int): Speed below which movement is considered stillness.

        Returns:
            movement_bursts (int): Count of burst movement periods.
            stillness_periods (int): Count of stillness periods.
        """
        dx = self.df["Mouse position X"].diff().fillna(0)
        dy = self.df["Mouse position Y"].diff().fillna(0)
        displacement = np.sqrt(dx**2 + dy**2)
        time_diff = self.df["Recording timestamp"].diff().fillna(0) / 1e6  # Convert to seconds
        velocity = displacement / time_diff.replace(0, np.nan)

        movement_bursts = (velocity > burst_threshold).sum()
        stillness_periods = (velocity < stillness_threshold).sum()

        return movement_bursts, stillness_periods

    # ------------------------- COMPUTE ALL METRICS PER TASK -------------------------

    def compute_all_metrics(self):
        """Compute all mouse-related metrics per task execution."""
        avg_velocity, avg_acceleration = self.compute_velocity_acceleration()
        movement_freq, idle_time = self.compute_movement_frequency_idle_time()
        click_count = self.compute_click_count()
        keyboard_count = self.compute_keyboard_count()
        direction_changes = self.compute_path_patterns()
        total_distance, num_stops = self.compute_total_distance_and_stops()
        movement_bursts, stillness_periods = self.compute_movement_bursts()

        return {
            "Avg Mouse Velocity (px/s)": avg_velocity,
            "Avg Mouse Acceleration (px/s²)": avg_acceleration,
            "Movement Frequency (movements/s)": movement_freq,
            "Total Idle Time (s)": idle_time,
            "Click Count": click_count,
            "Keyboard Count": keyboard_count,
            "Path Direction Changes": direction_changes,
            "Total Distance Traveled (px)": total_distance,
            "Number of Stops": num_stops,
            "Movement Bursts": movement_bursts,
            "Stillness Periods": stillness_periods
        }
