import pandas as pd
import numpy as np

class EyeTrackingProcessor:
    """
    A class for processing eye-tracking and mouse movement data from TSV files.
    It provides functionalities for reading, cleaning, resampling, and chunking tasks.
    """
    
    def __init__(self):
        self.timestamp_col = None
        
    # ------------------------- 0. HELPER METHODS -------------------------
    
    def detect_timestamp_column(self, df: pd.DataFrame):
        """Detect and store the timestamp column."""
        if self.timestamp_col is None:
            for col in df.columns:
                if col.startswith("Recording timestamp"):
                    self.timestamp_col = col
                    break
        if self.timestamp_col is None:
            raise ValueError("No suitable timestamp column found in DataFrame.")

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

    def task_range_finder_old(self, df: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
        """Find start and end times of tasks within a DataFrame and warn for unmatched starts."""
        self.detect_timestamp_column(df)
        event_df = df[df['Event'].str.contains('Task', na=False)].sort_values(by=self.timestamp_col)
        task_ranges = {}
        task_stack = {}

        for _, row in event_df.iterrows():
            event, timestamp = row["Event"], row[self.timestamp_col]
            if "end" not in event:
                task_stack[event] = timestamp
            else:
                task_type = event.replace(" end", "")
                if task_type in task_stack:
                    task_ranges.setdefault(task_type, []).append((task_stack.pop(task_type), timestamp))
                else:
                    print(f"⚠️ Warning: End event '{event}' at {timestamp} without matching start.")

        # After processing, check for unmatched starts
        if task_stack:
            print("⚠️ Warning: The following tasks have a start but no corresponding end:")
            for task, start_time in task_stack.items():
                print(f"   - {task} started at {start_time}")

        return task_ranges
    
    def task_range_finder(self, df: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
        """Find start and end times of tasks within a DataFrame.
        
        Handles overlapping or nested starts of the same task using per-task stacks.
        """
        self.detect_timestamp_column(df)

        event_df = df[df['Event'].str.contains('Task', na=False)].sort_values(by=self.timestamp_col)
        task_ranges = {}
        task_stack = {}

        for _, row in event_df.iterrows():
            event, timestamp = row["Event"], row[self.timestamp_col]

            if event.endswith("end"):
                task_type = event.replace(" end", "")
                if task_type in task_stack and task_stack[task_type]:
                    start_time = task_stack[task_type].pop()
                    task_ranges.setdefault(task_type, []).append((start_time, timestamp))
                else:
                    print(f"⚠️ Unmatched 'end' for {task_type} at {timestamp}")
            else:
                task_type = event  # e.g., "Task 3"
                task_stack.setdefault(task_type, []).append(timestamp)

        # Warn about unmatched starts
        for task_type, stack in task_stack.items():
            for unmatched_start in stack:
                print(f"⚠️ Unmatched 'start' for {task_type} at {unmatched_start}")

        return task_ranges

    # def get_features_old(self, dfs: list[pd.DataFrame], features: list[str]) -> pd.DataFrame:
    #     """Extract features for tasks from multiple DataFrames."""
        
    #     full_dataset = []
    #     for df in dfs:
            
    #         sub_dataset = []
    #         task_ranges = self.task_range_finder(df) # Detection of timestamp column per df
    #         for task, periods in task_ranges.items():
    #             for i, (start, end) in enumerate(periods):
    #                 task_data = df.loc[(df[self.timestamp_col] >= start) & (df[self.timestamp_col] <= end), features].copy()
    #                 task_data["Task_id"] = int(task[-1])
    #                 task_data["Task_execution"] = i
    #                 sub_dataset.append(task_data)
    #         if sub_dataset:
    #             full_dataset.append(pd.concat(sub_dataset))
    #     return pd.concat(full_dataset, ignore_index=True)
    
    def get_features(
        self,
        dfs: list[pd.DataFrame],
        features: list[str]
    ) -> dict[str, pd.DataFrame]:
        """
        Extracts per-task chunks from each DataFrame (for each participant) and returns a dict where:
        - key = "participant_task_execution" id
        - value = corresponding dataframe chunk (can overlap)

        Overlapping task intervals are handled: a row can appear in multiple chunks.
        """

        task_chunks = {}

        for df in dfs:
            participant = df["Participant name"].iloc[0] if "Participant name" in df.columns else 0
            task_ranges = self.task_range_finder(df)  # ranges for each tasks

            for task, periods in task_ranges.items():
                for i, (start, end) in enumerate(periods):
                    # Extract the task slice
                    mask = (df[self.timestamp_col] >= start) & (df[self.timestamp_col] <= end)
                    task_data = df.loc[mask, features].copy()

                    # Create unique ID
                    task_id = int(task.split()[-1])
                    uid = f"{participant}_{task_id}_{i}"

                    # Assign metadata
                    task_data["Task_id"] = task_id
                    task_data["Task_execution"] = i
                    task_data["Participant name"] = participant
                    task_data["id"] = uid

                    task_chunks[uid] = task_data

        return task_chunks
    
    # ------------------------- 3. BLINK IDENTIFICATION -------------------------
    # def detect_blinks_old(self, df: pd.DataFrame, blink_threshold: int= 1e5)-> pd.DataFrame:
    #     df = df.sort_values(by=["Participant name", "Task_id", "Task_execution", self.timestamp_col]).reset_index(drop=True)

    #     # Identify rows where gaze data is missing
    #     df["Missing Gaze"] = df["Gaze point X"].isna() | df["Gaze point Y"].isna()
    #     df["Time Diff"] = df["Recording timestamp"].diff()

    #     # Identify blink start (when missing gaze starts) and blink end (when gaze reappears)
    #     df["Blink Start"] = df["Missing Gaze"] & ~df["Missing Gaze"].shift(1, fill_value=False)
    #     df["Blink End"] = ~df["Missing Gaze"] & df["Missing Gaze"].shift(1, fill_value=False)

    #     # Assign blink IDs
    #     df["Blink ID"] = df["Blink Start"].cumsum()

    #     # Compute blink durations
    #     blink_durations = df.groupby("Blink ID")["Time Diff"].sum().reset_index()
    #     blinks_detected = blink_durations[blink_durations["Time Diff"] > blink_threshold]

    #     # Create a binary blink mask column in the original dataframe
    #     df["Blink"] = df["Blink ID"].isin(blinks_detected["Blink ID"]).astype(int)

    #     # Drop unnecessary intermediate columns
    #     df.drop(columns=["Missing Gaze", "Time Diff", "Blink Start", "Blink End", "Blink ID"], inplace=True)
        
    #     return df, blinks_detected
    
    def detect_blinks(self, task_chunks: dict[str, pd.DataFrame], blink_threshold: int = 100):
        """
        Detect blinks in a dictionary of task-based DataFrame chunks.

        Parameters:
        - task_chunks: dict where keys are task IDs and values are DataFrames
        - blink_threshold: duration threshold in milliseconds

        Returns:
        - updated_chunks: dict with "Blink" column added to each DataFrame
        - all_blinks: DataFrame summarizing detected blinks across all tasks
        """

        updated_chunks = {}
        all_blinks = {}

        for task_id, df in task_chunks.items():
            df = df.copy()

            # Detect gaze columns
            x_col = next((col for col in df.columns if "Gaze point X" in col), None)
            y_col = next((col for col in df.columns if "Gaze point Y" in col), None)

            if not x_col or not y_col:
                print(f"⚠️ Skipping {task_id}: Gaze columns not found.")
                continue

            # Sort to ensure temporal order
            df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)

            # Normalize timestamps if needed
            time_diff = df[self.timestamp_col].diff()
            median_diff = time_diff.dropna().median()

            if median_diff > 10000:
                df[self.timestamp_col] = df[self.timestamp_col] / 1000  # µs → ms

            # Blink detection logic
            df["Missing Gaze"] = df[x_col].isna() | df[y_col].isna()
            df["Time Diff"] = df[self.timestamp_col].diff()
            df["Blink Start"] = df["Missing Gaze"] & ~df["Missing Gaze"].shift(1, fill_value=False)
            df["Blink End"] = ~df["Missing Gaze"] & df["Missing Gaze"].shift(1, fill_value=False)
            df["Blink ID"] = df["Blink Start"].cumsum()

            # Filter for actual blinks
            blink_durations = df.groupby("Blink ID")["Time Diff"].sum().reset_index()
            valid_blinks = blink_durations.loc[blink_durations["Time Diff"] > blink_threshold].copy()
            # Classify blink types
            valid_blinks["Attention State"] = valid_blinks["Time Diff"].apply(
                lambda dur: "Blink" if dur <= 400 else "Loss of attention"
            )
            valid_blinks["Loss of Attention"] = valid_blinks["Time Diff"] > 400
            df["Blink"] = df["Blink ID"].isin(valid_blinks["Blink ID"]).astype(int)
            
            df["Loss of Attention"] = False
            long_blink_ids = valid_blinks.loc[valid_blinks["Loss of Attention"], "Blink ID"]
            df.loc[df["Blink ID"].isin(long_blink_ids), "Loss of Attention"] = True
            
            all_blinks[task_id] = valid_blinks

            # Clean up intermediate columns
            df.drop(columns=["Missing Gaze", "Time Diff", "Blink Start", "Blink End", "Blink ID"], inplace=True)
            updated_chunks[task_id] = df

        return updated_chunks, all_blinks
    
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
    Compute gaze-related metrics from a *single task execution* DataFrame.
    Handles variable column names (units in brackets) and auto time-unit normalization.
    """

    def __init__(self, df: pd.DataFrame, timestamp_unit: str = "auto"):
        """
        Parameters
        ----------
        df : pd.DataFrame
            Data for one task execution (already filtered if needed).
        timestamp_unit : {"auto", "ms", "us", "s"}
            Force unit if known; otherwise infer.
        """
        self.original_df = df
        self.timestamp_unit_arg = timestamp_unit

        # Resolve/standardize column names we need
        self.ts_col = self._find_col(df, "Recording timestamp")
        self.gx_col = self._find_col(df, "Gaze point X")
        self.gy_col = self._find_col(df, "Gaze point Y")

        # Compute time-unit scaling
        self.seconds_per_unit = self._infer_time_scale(df[self.ts_col], timestamp_unit)

        # Work on a copy sorted by timestamp
        self.df = df.sort_values(by=self.ts_col).reset_index(drop=True).copy()

    # ------------------------- HELPERS FUCNTIONS -------------------------
    @staticmethod
    def _find_col(df: pd.DataFrame, prefix: str, required: bool = True):
        """
        Return first column whose name starts with `prefix`.
        Example: prefix='Gaze point X' will match 'Gaze point X [DACS px]'.
        """
        for col in df.columns:
            if col.startswith(prefix):
                return col
        if required:
            raise ValueError(f"Required column starting with '{prefix}' not found.")
        return None

    def _infer_time_scale(self, ts: pd.Series, unit_arg: str) -> float:
        """
        Determine how many *seconds* correspond to 1 raw timestamp unit.
        """
        name = ts.name or ""
        name_lower = name.lower()

        # Unit given
        if unit_arg == "ms":
            return 1e-3
        if unit_arg in ("us", "µs"):
            return 1e-6
        if unit_arg == "s":
            return 1.0

        # Detect unit in name
        if "[ms" in name_lower:
            return 1e-3
        if "[us" in name_lower or "[µs" in name_lower:
            return 1e-6

        # Detect from median sample interval magnitude
        diffs = ts.sort_values().diff().dropna()
        if diffs.empty:
            return 1e-3  # by default
        med = float(diffs.median())

        # Heuristic thresholds
        # typical eye-tracking sample period ~16 ms (60 Hz) or ~4 ms (250 Hz) etc.
        if med > 5000:        # >5k units between samples → likely microseconds
            return 1e-6
        elif med > 0.5:       # >0.5 units likely ms-scale
            return 1e-3
        else:                 # else already seconds
            return 1.0

    def _to_seconds(self, s: pd.Series) -> pd.Series:
        return s * self.seconds_per_unit

    def _time_diff_seconds(self) -> pd.Series:
        return self._to_seconds(self.df[self.ts_col].diff().fillna(0))

    # ------------------------- FIXATION COMPUTATION -------------------------
    
    def compute_fixation_statistics(self, time_threshold=100, radius_threshold=50):
            """
            Parameters
            ----------
            time_threshold : numeric
                Minimum fixation duration in **milliseconds**.
            radius_threshold : numeric
                Max movement (px) to remain in fixation.

            Returns
            -------
            fixation_count, total_fixation_duration_s, avg_fixation_duration_s
            """
            gx = self.df[self.gx_col]
            gy = self.df[self.gy_col]

            dx = gx.diff().fillna(0)
            dy = gy.diff().fillna(0)
            displacement = np.sqrt(dx**2 + dy**2)

            fixation_mask = displacement < radius_threshold

            ts = self.df[self.ts_col]
            start_series = ts.where(fixation_mask & ~fixation_mask.shift(1, fill_value=False))
            end_series   = ts.where(~fixation_mask & fixation_mask.shift(1, fill_value=False))

            start_vals = start_series.dropna().to_numpy()
            end_vals   = end_series.dropna().to_numpy()

            # Align
            n = min(len(start_vals), len(end_vals))
            durations_raw = end_vals[:n] - start_vals[:n]

            # Convert to ms then seconds
            durations_ms = durations_raw * (self.seconds_per_unit * 1000.0)
            valid = durations_ms >= time_threshold

            fixation_count = int(valid.sum())
            total_fix_dur_s = float((durations_ms[valid].sum()) / 1000.0)
            avg_fix_dur_s = total_fix_dur_s / fixation_count if fixation_count else 0.0

            return fixation_count, total_fix_dur_s, avg_fix_dur_s

    # ------------------------- SACCADE COMPUTATION -------------------------

    def compute_saccade_statistics(self, radius_threshold=50):
        """
        Returns
        -------
        saccade_count, avg_saccade_amplitude_px, avg_saccade_velocity_px_per_s
        """
        gx = self.df[self.gx_col]
        gy = self.df[self.gy_col]

        dx = gx.diff().fillna(0)
        dy = gy.diff().fillna(0)
        displacement = np.sqrt(dx**2 + dy**2)

        dt_s = self._time_diff_seconds().replace(0, np.nan)

        saccade_mask = displacement >= radius_threshold
        saccade_count = int(saccade_mask.sum())

        if saccade_count:
            amp = displacement[saccade_mask].mean()
            vel = (displacement[saccade_mask] / dt_s[saccade_mask]).mean()
        else:
            amp = 0.0
            vel = 0.0

        return saccade_count, amp, vel

    # ------------------------- VELOCITY & ACCELERATION COMPUTATION -------------------------

    def compute_velocity_acceleration(self):
        """
        Returns
        -------
        velocity_s : pd.Series of px/s
        acceleration_s2 : pd.Series of px/s^2
        """
        gx = self.df[self.gx_col]
        gy = self.df[self.gy_col]

        dx = gx.diff().fillna(0)
        dy = gy.diff().fillna(0)
        disp = np.sqrt(dx**2 + dy**2)

        dt_s = self._time_diff_seconds().replace(0, np.nan)

        velocity = disp / dt_s
        acceleration = velocity.diff().fillna(0) / dt_s

        return velocity, acceleration

    # ------------------------- BLINK RATE COMPUTATION -------------------------

    def compute_blink_rate(self):
        """
        Blink rate = (# rows flagged as blink) / total duration (seconds).
        If Blink column missing, returns 0.
        """
        if "Blink" not in self.df.columns:
            return 0.0

        ts = self.df[self.ts_col]
        total_time_s = (ts.max() - ts.min()) * self.seconds_per_unit
        if total_time_s <= 0:
            return 0.0

        blink_count = int(self.df["Blink"].sum())
        return blink_count / total_time_s

    # ------------------------- GAZE DISPERSION COMPUTATION -------------------------

    def compute_gaze_dispersion(self):
        """Bounding-box area of gaze coordinates (px^2)."""
        gx = self.df[self.gx_col]
        gy = self.df[self.gy_col]
        if len(self.df) > 3:
            return (gx.max() - gx.min()) * (gy.max() - gy.min())
        return 0.0

    # ------------------------- COMPUTE ALL METRICS PER TASK -------------------------

    def compute_all_metrics(self):
        fix_n, fix_tot_s, fix_avg_s = self.compute_fixation_statistics()
        sac_n, sac_amp_px, sac_vel = self.compute_saccade_statistics()
        vel_s, acc_s2 = self.compute_velocity_acceleration()
        blink_rate = self.compute_blink_rate()
        dispersion = self.compute_gaze_dispersion()

        return {
            "Fixation Count": fix_n,
            "Total Fixation Duration (s)": fix_tot_s,
            "Avg Fixation Duration (s)": fix_avg_s,
            "Saccade Count": sac_n,
            "Avg Saccade Amplitude (px)": sac_amp_px,
            "Avg Saccade Velocity (px/s)": sac_vel,
            "Avg Gaze Velocity (px/s)": vel_s.mean(),
            "Avg Gaze Acceleration (px/s²)": acc_s2.mean(),
            "Blink Rate (blinks/s)": blink_rate,
            "Gaze Dispersion (area_px²)": dispersion,
            "Seconds per raw time unit": self.seconds_per_unit,
            "Timestamp column": self.ts_col,
            "Gaze X column": self.gx_col,
            "Gaze Y column": self.gy_col,
        }

class MouseMetricsProcessor:
    """
    Computes mouse movement-related metrics from a DataFrame with a single task execution.
    Automatically handles variations in column names (with/without units) and timestamp units.
    """

    def __init__(self, df: pd.DataFrame, timestamp_unit: str = "auto"):
        self.original_df = df
        self.timestamp_unit_arg = timestamp_unit

        self.ts_col = self._find_col(df, "Recording timestamp")
        self.mx_col = self._find_col(df, "Mouse position X")
        self.my_col = self._find_col(df, "Mouse position Y")

        self.seconds_per_unit = self._infer_time_scale(df[self.ts_col], timestamp_unit)
        self.df = df.sort_values(by=self.ts_col).reset_index(drop=True).copy()

    def _find_col(self, df: pd.DataFrame, prefix: str):
        for col in df.columns:
            if col.startswith(prefix):
                return col
        raise ValueError(f"Required column starting with '{prefix}' not found.")

    def _infer_time_scale(self, ts: pd.Series, unit_arg: str) -> float:
        name = ts.name or ""
        name_lower = name.lower()

        if unit_arg == "ms":
            return 1e-3
        if unit_arg in ("us", "µs", "μs"):
            return 1e-6
        if unit_arg == "s":
            return 1.0

        if "[ms" in name_lower:
            return 1e-3
        if "[us" in name_lower or "[µs" in name_lower or "[μs" in name_lower:
            return 1e-6

        diffs = ts.sort_values().diff().dropna()
        if diffs.empty:
            return 1e-3  # fallback
        med = float(diffs.median())
        if med > 5000:
            return 1e-6
        elif med > 0.5:
            return 1e-3
        else:
            return 1.0

    def _to_seconds(self, s: pd.Series) -> pd.Series:
        return s * self.seconds_per_unit

    def _time_diff_seconds(self) -> pd.Series:
        return self._to_seconds(self.df[self.ts_col].diff().fillna(0))

    # ------------------------- VELOCITY & ACCELERATION -------------------------
    def compute_velocity_acceleration(self):
        dx = self.df[self.mx_col].diff().fillna(0)
        dy = self.df[self.my_col].diff().fillna(0)
        disp = np.sqrt(dx**2 + dy**2)
        dt = self._time_diff_seconds().replace(0, np.nan)
        velocity = disp / dt
        acceleration = velocity.diff().fillna(0) / dt
        return velocity, acceleration

    # ------------------------- MOVEMENT FREQUENCY & IDLE TIMES -------------------------
    def compute_movement_frequency_idle_time(self, idle_threshold=0.5):
        dt = self._time_diff_seconds()
        moved = (self.df[self.mx_col].diff().abs() > 0) | (self.df[self.my_col].diff().abs() > 0)
        movement_frequency = moved.sum() / dt.sum()
        idle_periods = dt[~moved]
        total_idle_time = idle_periods[idle_periods >= idle_threshold].sum()
        return movement_frequency, total_idle_time

    # ------------------------- CLICK & KEYBOARD EVENT COUNTS -------------------------
    def compute_click_count(self):
        if "Event" in self.df.columns:
            return int((self.df["Event"] == "MouseEvent").sum())
        return 0

    def compute_keyboard_count(self):
        if "Event" in self.df.columns:
            return int((self.df["Event"] == "KeyboardEvent").sum())
        return 0

    # ------------------------- DIRECTION CHANGES -------------------------
    def compute_path_patterns(self, angle_threshold=30):
        dx = self.df[self.mx_col].diff().fillna(0)
        dy = self.df[self.my_col].diff().fillna(0)
        angles = np.arctan2(dy, dx)
        angle_diff = np.abs(np.diff(angles))
        changes = np.sum(angle_diff > np.radians(angle_threshold))
        return int(changes)

    # ------------------------- DISTANCE & STOPS -------------------------
    def compute_total_distance_and_stops(self, stop_threshold=5):
        dx = self.df[self.mx_col].diff().fillna(0)
        dy = self.df[self.my_col].diff().fillna(0)
        disp = np.sqrt(dx**2 + dy**2)
        dt = self._time_diff_seconds().replace(0, np.nan)
        velocity = disp / dt
        total_distance = disp.sum()
        num_stops = (velocity < stop_threshold).sum()
        return total_distance, int(num_stops)

    # ------------------------- MOVEMENT BURSTS & STILLNESS -------------------------
    def compute_movement_bursts(self, burst_threshold=50, stillness_threshold=5):
        dx = self.df[self.mx_col].diff().fillna(0)
        dy = self.df[self.my_col].diff().fillna(0)
        disp = np.sqrt(dx**2 + dy**2)
        dt = self._time_diff_seconds().replace(0, np.nan)
        velocity = disp / dt
        bursts = (velocity > burst_threshold).sum()
        stillness = (velocity < stillness_threshold).sum()
        return int(bursts), int(stillness)

    # ------------------------- AGGREGATE -------------------------
    def compute_all_metrics(self):
        velocity, acceleration = self.compute_velocity_acceleration()
        movement_freq, idle_time = self.compute_movement_frequency_idle_time()
        click_count = self.compute_click_count()
        keyboard_count = self.compute_keyboard_count()
        direction_changes = self.compute_path_patterns()
        total_distance, num_stops = self.compute_total_distance_and_stops()
        bursts, stillness = self.compute_movement_bursts()

        return {
            "Avg Mouse Velocity (px/s)": velocity.mean(),
            "Avg Mouse Acceleration (px/s²)": acceleration.mean(),
            "Movement Frequency (movements/s)": movement_freq,
            "Total Idle Time (s)": idle_time,
            "Click Count": click_count,
            "Keyboard Count": keyboard_count,
            "Path Direction Changes": direction_changes,
            "Total Distance Traveled (px)": total_distance,
            "Number of Stops": num_stops,
            "Movement Bursts": bursts,
            "Stillness Periods": stillness,
            "Seconds per raw time unit": self.seconds_per_unit,
            "Timestamp column": self.ts_col,
            "Mouse X column": self.mx_col,
            "Mouse Y column": self.my_col,
        }