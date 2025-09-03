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
        
    @staticmethod
    def extract_atco_task_roots(paths: list[str]) -> list[str]:
        """Collect all unique task roots across all files."""
        all_roots = set()

        for path in paths:
            df = pd.read_csv(path, sep="\t")
            for event in df["Event"].unique():
                if (
                    isinstance(event, str)
                    and " - " in event
                    and (event.endswith("start") or event.endswith("end"))
                    and not event.startswith("Session -")
                    and not event.startswith("Conflict detection -")
                ):
                    root = event.split(" - ")[0]
                    all_roots.add(root)

        return sorted(all_roots)
    
    @staticmethod
    def build_global_task_map(task_roots: list[str]) -> dict[str, str]:
        """Map each task root to a consistent Task N label."""
        return {root: f"Task {i}" for i, root in enumerate(task_roots)}
    
    @staticmethod
    def apply_global_task_mapping(df: pd.DataFrame, task_map: dict[str, str]) -> pd.DataFrame:
        """Apply the global task map to the 'Event' column."""
        def map_event(event):
            if isinstance(event, str) and " - " in event:
                root, suffix = event.split(" - ", maxsplit=1)
                task_label = task_map.get(root)
                if task_label:
                    return task_label if suffix == "start" else f"{task_label} end"
            return event

        df["Event"] = df["Event"].apply(map_event)
        return df

    # ------------------------- 1. DATA LOADING & VALIDATION -------------------------
    
    # def read_tsv_old(self, path: str) -> pd.DataFrame:
    #     """Read a TSV file and validate task labeling."""
    #     df = pd.read_csv(path, sep='\t')
        
    #     # Validate task counts
    #     expected_tasks = [f"Task {i}" for i in range(1, 7)] + [f"Task {i} end" for i in range(1, 7)]
    #     task_counts = df['Event'].value_counts()
        
    #     for task in expected_tasks:
    #         if task_counts.get(task, 0) != 6:
    #             print(f"Warning: {task} in {path} has {task_counts.get(task, 0)} occurrences instead of 6.")
        
    #     return df
    
    # def read_tsv(self, path: str) -> pd.DataFrame:
    #     """Read a TSV file and convert task labeling."""
    #     df = pd.read_csv(path, sep='\t')
        
    #     # Session is not a task
    #     # Conflict detection is only start. Find a way to measure it. 
    #     atco_tasks = [
    #         event for event in df.Event.unique()
    #         if (
    #             isinstance(event, str)
    #             and " - " in event
    #             and (event.endswith("start") or event.endswith("end"))
    #             and not event.startswith("Session -")
    #             and not event.startswith("Conflict detection -")
    #         )
    #     ]
        
    #     # Mapping like Tasks were defined in ms_df
    #     def map_events_to_tasks_inplace(df, event_column, task_events):
    #         task_roots = sorted(set(event.split(" - ")[0] for event in task_events if " - " in event))
    #         root_to_task = {root: f"Task {i+1}" for i, root in enumerate(task_roots)}

    #         def map_event(event):
    #             if isinstance(event, str) and " - " in event:
    #                 root, suffix = event.split(" - ", maxsplit=1)
    #                 task_label = root_to_task.get(root)
    #                 if task_label:
    #                     return task_label if suffix == "start" else f"{task_label} end"
    #             return event
            
    #         df[event_column] = df[event_column].apply(map_event)
    #         return root_to_task
        
    #     atco_tasks_map = map_events_to_tasks_inplace(df, "Event", atco_tasks)
        
    #     return df, atco_tasks_map

    # def load_data(self, paths: list[str]) -> list[pd.DataFrame]:
    #     """Load multiple TSV files into a list of DataFrames, and map ATCO tasks"""
    #     dfs = []
    #     atco_tasks_maps = []
    #     for i, path in enumerate(paths):
    #         df, atco_map = self.read_tsv(path)
    #         if "Participant name" not in df.columns:
    #             df["Participant name"] = i
    #         dfs.append(df)
    #         atco_tasks_maps.append(atco_map)
    #     return dfs, atco_tasks_maps
    
    
    
    def load_data(self, paths: list[str]) -> tuple[list[pd.DataFrame], dict[str, str]]:
        """Load multiple TSV files with consistent ATCO task mapping."""
        task_roots = self.extract_atco_task_roots(paths)
        global_task_map = self.build_global_task_map(task_roots)

        dfs = []
        for i, path in enumerate(paths):
            df = pd.read_csv(path, sep="\t")
            df = self.apply_global_task_mapping(df, global_task_map)
            if "Participant name" not in df.columns:
                df["Participant name"] = i
            dfs.append(df)

        return dfs, global_task_map

    # ------------------------- 2. TASK IDENTIFICATION & FEATURE EXTRACTION -------------------------

    # def task_range_finder_old(self, df: pd.DataFrame) -> dict[str, list[tuple[int, int]]]:
    #     """Find start and end times of tasks within a DataFrame and warn for unmatched starts."""
    #     self.detect_timestamp_column(df)
    #     event_df = df[df['Event'].str.contains('Task', na=False)].sort_values(by=self.timestamp_col)
    #     task_ranges = {}
    #     task_stack = {}

    #     for _, row in event_df.iterrows():
    #         event, timestamp = row["Event"], row[self.timestamp_col]
    #         if "end" not in event:
    #             task_stack[event] = timestamp
    #         else:
    #             task_type = event.replace(" end", "")
    #             if task_type in task_stack:
    #                 task_ranges.setdefault(task_type, []).append((task_stack.pop(task_type), timestamp))
    #             else:
    #                 print(f"⚠️ Warning: End event '{event}' at {timestamp} without matching start.")

    #     # After processing, check for unmatched starts
    #     if task_stack:
    #         print("⚠️ Warning: The following tasks have a start but no corresponding end:")
    #         for task, start_time in task_stack.items():
    #             print(f"   - {task} started at {start_time}")

    #     return task_ranges
    
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
    
    
    def get_fixed_window_chunks(
        self,
        dfs: list[pd.DataFrame],
        features: list[str],
        window_ms: int = 3000,
        step_ms: int | None = None,
        min_presence: float = 0.5,
    ) -> dict[str, pd.DataFrame]:
        """
        Slice each participant stream into fixed-length windows and assign a task label
        based on maximum overlap with any Task occurrence (start/end pair).

        - window_ms: window length in milliseconds (e.g., 3000 for 3s).
        - step_ms: hop size in milliseconds. Default = window_ms (no overlap).
        - min_presence: minimum fraction of the window that must be covered by the winning task
        to be accepted (0.5 = majority). Windows below this are dropped.

        Output format:
        dict[id] -> DataFrame slice with columns `features` plus the metadata
        ["Task_id", "Task_execution", "Participant name", "id"]
        """
        chunks: dict[str, pd.DataFrame] = {}
        step_ms = step_ms or window_ms

        for df in dfs:
            # Participant id/name
            participant = df["Participant name"].iloc[0] if "Participant name" in df.columns else 0

            # Timestamp detection + normalization (ms)
            self.detect_timestamp_column(df)
            df_sorted = df.sort_values(by=self.timestamp_col).reset_index(drop=True)
            median_diff = df_sorted[self.timestamp_col].diff().dropna().median()
            if pd.notna(median_diff) and median_diff > 10000:
                # µs -> ms
                df_sorted[self.timestamp_col] = df_sorted[self.timestamp_col] / 1000.0

            # Build task ranges (per original occurrences)
            task_ranges = self.task_range_finder(df_sorted)  # {"Task N": [(start,end), ...]}
            # Precompute a flat list with (task_name, occ_index, start, end) for quick overlap checks
            flat_ranges: list[tuple[str, int, float, float]] = []
            for task_name, periods in task_ranges.items():
                for occ_idx, (st, en) in enumerate(periods):
                    flat_ranges.append((task_name, occ_idx, float(st), float(en)))

            if not flat_ranges:
                # No tasks found -> skip this participant entirely
                continue

            # Time domain and window anchors
            t_min = float(df_sorted[self.timestamp_col].min())
            t_max = float(df_sorted[self.timestamp_col].max())
            if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
                continue

            window_starts = np.arange(t_min, t_max - window_ms + 1e-9, step_ms)
            # Per-task counters to keep ID uniqueness: participant_task_occurrence
            # Occurrence here counts windows per Task_id (not original start/end index).
            per_task_window_counter: dict[int, int] = {}

            for w_start in window_starts:
                w_end = w_start + window_ms

                # Find the overlapping task occurrence with the largest overlap
                best = None  # (overlap_ms, task_name, occ_idx)
                for task_name, occ_idx, st, en in flat_ranges:
                    overlap = max(0.0, min(w_end, en) - max(w_start, st))
                    if overlap > 0.0:
                        if (best is None) or (overlap > best[0]):
                            best = (overlap, task_name, occ_idx)

                if best is None:
                    # No task overlaps this window -> skip (between tasks)
                    continue

                overlap_ms, best_task_name, best_occ_idx = best
                if overlap_ms / window_ms < min_presence:
                    # Not enough majority coverage -> skip
                    continue

                # Extract the window slice
                mask = (df_sorted[self.timestamp_col] >= w_start) & (df_sorted[self.timestamp_col] < w_end)
                window_df = df_sorted.loc[mask, features].copy()
                if window_df.empty:
                    # No samples landed in this window (edge case) -> skip
                    continue

                # Task metadata
                task_id = int(best_task_name.split()[-1])  # "Task N" -> N

                # Maintain window occurrence per task to build unique ids
                window_occurrence = per_task_window_counter.get(task_id, 0)
                per_task_window_counter[task_id] = window_occurrence + 1

                # Attach metadata
                window_df["Task_id"] = task_id
                # IMPORTANT:
                # - Keep Task_execution = original start/end occurrence index (stable semantics)
                # - The ID uses the *window* occurrence counter to stay unique per window.
                window_df["Task_execution"] = best_occ_idx
                window_df["Participant name"] = participant
                uid = f"{participant}_{task_id}_{window_occurrence}"
                window_df["id"] = uid

                chunks[uid] = window_df

        return chunks

    
    # ------------------------- 3. BLINK IDENTIFICATION -------------------------
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
    
    def resample_single_task_chunk(self, df: pd.DataFrame, interpolate_cols: list[str], mode: str, param) -> pd.DataFrame:
        """
        Resample a single task DataFrame chunk based on time, number of points, or custom timestamps.

        Parameters:
        - df (pd.DataFrame): Input DataFrame representing a task segment.
        - interpolate_cols (list[str]): Columns to interpolate (e.g., gaze or mouse positions).
        - mode (str): Resampling mode. Options:
            - "time": fixed time step in milliseconds (float).
            - "points": fixed number of points (int).
            - "custom": specific timestamps to interpolate to (np.ndarray).
        - param (float | int | np.ndarray): Parameter for the selected mode.

        Returns:
        - pd.DataFrame: Resampled DataFrame with the same structure, interpolated columns, and metadata preserved.
        """

        df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)

        # Normalize timestamp units if necessary
        time_diff = df[self.timestamp_col].diff().dropna().median()
        if time_diff > 10000:
            df[self.timestamp_col] = df[self.timestamp_col] / 1000  # Convert µs → ms

        # Generate new timestamps
        if mode == "time":
            min_time, max_time = df[self.timestamp_col].min(), df[self.timestamp_col].max()
            new_timestamps = np.arange(min_time, max_time, param)
        elif mode == "points":
            original_indices = np.linspace(0, 1, len(df))
            new_indices = np.linspace(0, 1, param)
            new_timestamps = np.interp(new_indices, original_indices, df[self.timestamp_col])
        elif mode == "custom":
            new_timestamps = param
        else:
            raise ValueError(f"Invalid resampling mode: {mode}")

        # Interpolate columns
        resampled_df = pd.DataFrame({self.timestamp_col: new_timestamps})
        for col in interpolate_cols:
            if col in df.columns:
                resampled_df[col] = np.interp(new_timestamps, df[self.timestamp_col], df[col])

        # Add metadata
        for meta_col in ["id", "Participant name", "Task_id", "Task_execution"]:
            if meta_col in df.columns:
                resampled_df[meta_col] = df[meta_col].iloc[0]

        return resampled_df

    def resample_task_chunks(self, task_chunks: dict[str, pd.DataFrame], interpolate_cols: list[str], mode: str, param) -> dict[str, pd.DataFrame]:
        """
        Resample a dictionary of task chunks using a specified method.

        Parameters:
        - task_chunks (dict[str, pd.DataFrame]): Dictionary where each key is a task ID and each value is a task DataFrame.
        - interpolate_cols (list[str]): List of columns to interpolate numerically.
        - mode (str): Resampling strategy:
            - "time": fixed step size (param in milliseconds).
            - "points": fixed number of points.
            - "custom": pass explicit timestamp array.
        - param (float | int | np.ndarray): Parameter for the selected mode:
            - timestep in ms (if mode == "time")
            - number of points (if mode == "points")
            - timestamp array (if mode == "custom")

        Returns:
        - dict[str, pd.DataFrame]: Dictionary of resampled task DataFrames, same structure as input.
        """
        resampled_chunks = {}

        for task_id, df in task_chunks.items():
            try:
                resampled_df = self.resample_single_task_chunk(df, interpolate_cols, mode, param)
                resampled_chunks[task_id] = resampled_df
            except Exception as e:
                print(f"⚠️ Could not resample {task_id}: {e}")

        return resampled_chunks


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