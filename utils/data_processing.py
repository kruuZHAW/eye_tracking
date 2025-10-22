import pandas as pd
import numpy as np
from pathlib import Path

class EyeTrackingProcessor:
    """
    A class for processing eye-tracking and mouse movement data from TSV files.
    It provides functionalities for reading, cleaning, resampling, and chunking tasks.
    """
    
    def __init__(self):
        self.timestamp_col = None
        self.task_map: dict[str, str] | None = None
        
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
    def extract_atco_task_roots(dfs: list[pd.DataFrame]) -> list[str]:
        """Collect all unique task roots across all scenarios dataframes."""
        roots = set()
        for df in dfs:
            if "Event" not in df.columns:
                continue
            for event in df["Event"].dropna().unique():
                if (
                    isinstance(event, str)
                    and " - " in event
                    and (event.endswith("start") or event.endswith("end"))
                    and not event.startswith("Session -")
                    and not event.startswith("Conflict detection -")
                ):
                    roots.add(event.split(" - ")[0])
        return sorted(roots)
    
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
    
    def load_data(
        self,
        file_index: list[dict],
        want_columns: list[str] | None = None,
    ) -> tuple[list[pd.DataFrame], dict[str, str]]:
        """
        Read multiple Parquet files, add participant/scenario ids, and map tasks.
        file_index: [{'path': Path, 'participant_id': str, 'scenario_id': str}, ...]
        want_columns: columns to read from parquet for speed (optional).
        """
        dfs: list[pd.DataFrame] = []
        # Always request these if present
        base_needed = {"Event", "Participant name", "epoch_ms", "Recording timestamp [ms]"}
        read_cols = None
        if want_columns:
            read_cols = list(set(want_columns) | base_needed)

        for item in file_index:
            p = item["path"]
            # read only necessary columns (pyarrow engine strongly recommended)
            df = pd.read_parquet(p, columns=read_cols) if read_cols else pd.read_parquet(p)

            # attach ids
            df = df.copy()
            df["participant_id"] = str(item["participant_id"])
            df["scenario_id"] = str(item["scenario_id"])

            # If no Participant name column, fill it with participant_id
            if "Participant name" not in df.columns:
                df["Participant name"] = str(item["participant_id"])

            dfs.append(df)

        # Build task map across all files (if Event exists)
        task_roots = self.extract_atco_task_roots(dfs)
        task_map = self.build_global_task_map(task_roots) if task_roots else {}
        self.task_map = task_map

        # Apply mapping
        dfs = [self.apply_global_task_mapping(df, task_map) for df in dfs]

        # Detect timestamp once (on first non-empty df)
        for df in dfs:
            if not df.empty:
                self.detect_timestamp_column(df)
                break

        return dfs, task_map

    # ------------------------- 2. TASK IDENTIFICATION & FEATURE EXTRACTION -------------------------
    
    def task_range_finder(self, df: pd.DataFrame) -> tuple[dict[str, list[tuple[int, int]]], pd.DataFrame]:
        """Find start and end times of tasks within a DataFrame and record unmatched markers."""
        self.detect_timestamp_column(df)

        participant = df["participant_id"].iloc[0] if "participant_id" in df.columns else df["Participant name"].iloc[0]
        scenario_id = df["scenario_id"].iloc[0]

        event_df = df[df['Event'].str.contains('Task', na=False)].sort_values(by=self.timestamp_col)
        task_ranges: dict[str, list[tuple[int, int]]] = {}
        task_stack: dict[str, list[int]] = {}
        unmatched_records: list[dict] = []

        for _, row in event_df.iterrows():
            event, timestamp, epoch_ms = row["Event"], row[self.timestamp_col], row["epoch_ms"]

            if event.endswith("end"):
                task_type = event.replace(" end", "")
                if task_type in task_stack and task_stack[task_type]:
                    start_ts, start_ep = task_stack[task_type].pop()
                    task_ranges.setdefault(task_type, []).append((start_ts, timestamp))
                else:
                    # record unmatched end
                    unmatched_records.append({
                        "participant": participant,
                        "scenario_id": scenario_id,
                        "task": task_type,
                        "marker": "end",
                        "timestamp": timestamp,
                        "epoch_ms": epoch_ms,
                        "timestamp_utc": pd.to_datetime(epoch_ms, unit="ms", utc=True, errors="coerce").tz_localize(None),
                    })
                    print(f"⚠️ Unmatched 'end' for {task_type} at {timestamp}")
            else:
                task_type = event  
                task_stack.setdefault(task_type, []).append((timestamp, epoch_ms))

        # record unmatched starts
        for task_type, stack in task_stack.items():
            for unmatched_start, unmatched_epoch in stack:
                unmatched_records.append({
                    "participant": participant,
                    "scenario_id": scenario_id,
                    "task": task_type,
                    "marker": "start",
                    "timestamp": unmatched_start,
                    "epoch_ms": unmatched_epoch,
                    "timestamp_utc": pd.to_datetime(unmatched_epoch, unit="ms", utc=True, errors="coerce").tz_localize(None),
                })
                print(f"⚠️ Unmatched 'start' for {task_type} at {unmatched_start}")

        return task_ranges, pd.DataFrame(unmatched_records)

    def get_features(
        self,
        dfs: list[pd.DataFrame],
        features: list[str], 
        unmatched_excel_path: str | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Extracts per-task chunks from each DataFrame (for each participant) and returns a dict where:
        - key = "participant_task_execution" id
        - value = corresponding dataframe chunk (can overlap)

        Overlapping task intervals are handled: a row can appear in multiple chunks.
        Write an Excel file that identifies all missing markers
        """

        task_chunks: dict[str, pd.DataFrame] = {}
        writer = None
        used_sheet_names: set[str] = set()

        if unmatched_excel_path:
            Path(unmatched_excel_path).parent.mkdir(parents=True, exist_ok=True)
            writer = pd.ExcelWriter(unmatched_excel_path, engine="xlsxwriter")

        try:
            for df in dfs:
                participant = df["participant_id"].iloc[0] if "participant_id" in df.columns else df["Participant name"].iloc[0]
                scenario_id = df["scenario_id"].iloc[0]
                print(f"Finding tasks for participant {participant} Scenario {scenario_id}")

                task_ranges, unmatched_df = self.task_range_finder(df)

                # --- Write unmatched markers sheet (INSIDE the loop) ---
                if writer is not None:
                    sheet_base = f"{participant}_{scenario_id}"
                    sheet_name = sheet_base[:31]
                    suffix = 1
                    while sheet_name in used_sheet_names:
                        truncated = sheet_base[: max(0, 31 - (len(str(suffix)) + 1))]
                        sheet_name = f"{truncated}_{suffix}"
                        suffix += 1
                    used_sheet_names.add(sheet_name)

                    if unmatched_df.empty:
                        unmatched_df = pd.DataFrame([{
                            "participant": participant,
                            "scenario_id": scenario_id,
                            "task": None,
                            "marker": None,
                            "timestamp": None,
                            "epoch_ms": None,
                            "timestamp_utc": None,
                            "note": "No unmatched markers",
                        }])

                    unmatched_df.sort_values(by=["timestamp"], inplace=True, na_position="last")
                    unmatched_df.to_excel(writer, sheet_name=sheet_name, index=False)

                # --- Existing chunk extraction (INSIDE the loop) ---
                for task, periods in task_ranges.items():
                    for i, (start, end) in enumerate(periods):
                        mask = (df[self.timestamp_col] >= start) & (df[self.timestamp_col] <= end)
                        task_data = df.loc[mask, features].copy()

                        task_id = int(task.split()[-1])  # task is normalized, e.g., "Task 3"
                        uid = f"{participant}_{scenario_id}_{task_id}_{i}"

                        task_data["Task_id"] = task_id
                        task_data["Task_execution"] = i
                        task_data["Participant name"] = participant
                        task_data["Scenario_id"] = scenario_id
                        task_data["id"] = uid

                        task_chunks[uid] = task_data
        finally:
            if writer is not None:
                writer.close()

        return task_chunks
    
    
    def get_fixed_window_chunks(
        self,
        dfs: list[pd.DataFrame],
        features: list[str],
        window_ms: int = 3000,
        step_ms: int | None = None,
        min_presence: float = 0.5,
        idle_id: int = -1,
        label_idle: bool = True,
    ) -> dict[str, pd.DataFrame]:
        """
        Slice each participant stream into fixed-length windows and assign a task label
        based on maximum overlap with task ranges found.
        If there is no overlap (or a small), between the window and the task, it is labeled as "idling"

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
            participant = df["participant_id"].iloc[0] if "participant_id" in df.columns else df["Participant name"].iloc[0]
            
            # Scenario id
            scenario_id = df["scenario_id"].iloc[0]

            # Timestamp detection + normalization (ms)
            self.detect_timestamp_column(df)
            df_sorted = df.sort_values(by=self.timestamp_col).reset_index(drop=True)
            median_diff = df_sorted[self.timestamp_col].diff().dropna().median()
            if pd.notna(median_diff) and median_diff > 10000:
                # µs -> ms
                df_sorted[self.timestamp_col] = df_sorted[self.timestamp_col] / 1000.0

            # Build task ranges (per original occurrences)
            print(f"Finding tasks for participant {participant} Scenario {scenario_id}")
            task_ranges, _ = self.task_range_finder(df_sorted)  # {"Task N": [(start,end), ...]}
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

                # Extract the window slice
                mask = (df_sorted[self.timestamp_col] >= w_start) & (df_sorted[self.timestamp_col] < w_end)
                window_df = df_sorted.loc[mask, features].copy()
                if window_df.empty:
                    # No samples landed in this window (edge case) -> skip
                    continue
                
                if best is None or (best[0] / window_ms) < min_presence:
                    # ---- IDLING WINDOW ----
                    if not label_idle:
                        continue  # If not idle libel, then windows with no tasks are discarded
                    task_id = idle_id
                    task_exec = -1
                else:
                    # ---- TASKED WINDOW ----
                    _, best_task_name, best_occ_idx = best
                    task_id = int(best_task_name.split()[-1])
                    task_exec = best_occ_idx
                    # increase per-task counter (occurences) for unique window IDs ONLY for non-idle
                    window_occurrence = per_task_window_counter.get(task_id, 0)
                    per_task_window_counter[task_id] = window_occurrence + 1

                if task_id == idle_id:
                    idle_count = per_task_window_counter.get(task_id, 0)
                    per_task_window_counter[task_id] = idle_count + 1
                    uid = f"{participant}_{scenario_id}_{task_id}_{idle_count}"
                else:
                    uid = f"{participant}_{scenario_id}_{task_id}_{window_occurrence}"

                # Attach metadata
                window_df["Task_id"] = task_id
                window_df["Task_execution"] = task_exec
                window_df["Participant name"] = participant
                window_df["Scenario_id"] = scenario_id
                window_df["id"] = uid

                chunks[uid] = window_df

        return chunks
    
    # ------------------------- HELPER TASKS BOUNDAIRES -------------------------
    def collect_task_boundaries(
        self,
        dfs: list[pd.DataFrame],
        *,
        export_path: str | None = None,   # optional: write a CSV/Parquet
    ) -> pd.DataFrame:
        """
        Return one row per detected task occurrence with start/end timestamps.

        Columns:
        participant_id, scenario_id, task_label, task_id, execution,
        start_ts, end_ts, duration_ms, rows_in_range, uid
        """
        rows: list[dict] = []
        inv_map = {v: k for k, v in (self.task_map or {}).items()}

        for df in dfs:
            if df.empty:
                continue

            # IDs
            participant = df["participant_id"].iloc[0] if "participant_id" in df.columns else df["Participant name"].iloc[0]
            scenario_id = df["scenario_id"].iloc[0]

            # Make sure we know the timestamp column
            self.detect_timestamp_column(df)

            # Get ranges
            task_ranges, _ = self.task_range_finder(df)  # dict: "Task N" -> [(start, end), ...]

            # Build rows
            for task_code, periods in task_ranges.items():
                atco_label = inv_map.get(task_code, task_code)
                # numeric id from "Task N"
                try:
                    task_id = int(task_code.split()[-1])
                except Exception:
                    task_id = None

                for exec_idx, (start, end) in enumerate(periods):
                    mask = (df[self.timestamp_col] >= start) & (df[self.timestamp_col] <= end)
                    rows_in_range = int(mask.sum())
                    
                    start_epoch = df[df[self.timestamp_col] == start].epoch_ms.iloc[0]
                    end_epoch = df[df[self.timestamp_col] == end].epoch_ms.iloc[0]

                    rows.append({
                        "participant_id": str(participant),
                        "scenario_id": str(scenario_id),
                        "task_label": atco_label,
                        "task_code": task_code,        # e.g., "Task 3"
                        "task_id": task_id,              # e.g., 3
                        "execution": exec_idx,           # occurrence index within (participant, scenario, task)
                        "start_epoch_ms": int(start_epoch),
                        "end_epoch_ms": int(end_epoch),
                        "start_utc": pd.to_datetime(start_epoch, unit="ms", utc=True, errors="coerce").tz_localize(None),
                        "end_utc": pd.to_datetime(end_epoch, unit="ms", utc=True, errors="coerce").tz_localize(None),
                        "duration_ms": int(end - start),
                        "rows_in_range": rows_in_range,
                        "uid": f"{participant}_{scenario_id}_{task_id}_{exec_idx}",
                    })

        boundaries = pd.DataFrame(rows).sort_values(
            ["participant_id", "scenario_id", "task_id", "execution"]
        ).reset_index(drop=True)

        if export_path:
            if export_path.endswith(".parquet"):
                boundaries.to_parquet(export_path, index=False)
            else:
                boundaries.to_csv(export_path, index=False)

        return boundaries

    
    # ------------------------- 3. BLINK IDENTIFICATION -------------------------
    def detect_blinks(self, task_chunks: dict[str, pd.DataFrame], blink_threshold: int = 100):
        """
        Detect blinks in a dictionary of task-based DataFrame chunks.
         - A blink is a contiguous run where either gaze X or Y is NaN.
         - We mark Blink=True ONLY on rows inside runs whose duration > blink_threshold (ms).
         - Runs > 400 ms are also flagged as Loss of Attention.

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

            # Sort by time and (if needed) convert µs -> ms
            df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)
            med_diff = df[self.timestamp_col].diff().dropna().median()
            if pd.notna(med_diff) and med_diff > 10_000:
                df[self.timestamp_col] = df[self.timestamp_col] / 1_000.0  # µs → ms

            # Missing gaze mask
            df["Missing Gaze"] = df[x_col].isna() | df[y_col].isna()
            
            # Run IDs: increment when Missing Gaze changes (False->True or True->False)
            df["_run"] = df["Missing Gaze"].ne(df["Missing Gaze"].shift(fill_value=False)).cumsum()
            
            # Duration per run (sum of diffs) using only missing-gaze rows
            # If a run has a single row, its diff-sum is 0 (sensible for duration).
            dur_by_run = (
                df.loc[df["Missing Gaze"], [self.timestamp_col, "_run"]]
                .groupby("_run")[self.timestamp_col]
                .apply(lambda s: (s.diff().fillna(0)).sum())
            )

            # Which runs count as blinks/attention loss
            valid_runs = dur_by_run[dur_by_run > blink_threshold].index if not dur_by_run.empty else pd.Index([])
            long_runs  = dur_by_run[dur_by_run > 400].index if not dur_by_run.empty else pd.Index([])

            # Initialize flags
            df["Blink"] = False
            df["Loss of Attention"] = False

            # Mark only the rows inside the selected runs
            mask_missing = df["Missing Gaze"]
            if len(valid_runs) > 0:
                df.loc[mask_missing & df["_run"].isin(valid_runs), "Blink"] = True
            if len(long_runs) > 0:
                df.loc[mask_missing & df["_run"].isin(long_runs), "Loss of Attention"] = True

            # Build a compact per-run summary (only for missing runs)
            if not dur_by_run.empty:
                run_bounds = (
                    df.loc[df["Missing Gaze"], [self.timestamp_col, "_run"]]
                    .groupby("_run")[self.timestamp_col]
                    .agg(start_ts="first", end_ts="last")
                )
                summary = (
                    dur_by_run.to_frame("duration_ms")
                    .join(run_bounds, how="left")
                    .assign(
                        blink=lambda t: t["duration_ms"] > blink_threshold,
                        loss_of_attention=lambda t: t["duration_ms"] > 400,
                    )
                    .reset_index(names="run_id")
                )
                all_blinks[task_id] = summary.loc[summary["blink"] | summary["loss_of_attention"]].copy()
            else:
                all_blinks[task_id] = pd.DataFrame(columns=["run_id", "duration_ms", "start_ts", "end_ts", "blink", "loss_of_attention"])

            # Clean up helpers
            df.drop(columns=["_run", "Missing Gaze"], inplace=True)

            updated_chunks[task_id] = df
            
            # df["Time Diff"] = df[self.timestamp_col].diff()
            # df["Blink Start"] = df["Missing Gaze"] & ~df["Missing Gaze"].shift(1, fill_value=False)
            # df["Blink End"] = ~df["Missing Gaze"] & df["Missing Gaze"].shift(1, fill_value=False)
            # df["Blink ID"] = df["Blink Start"].cumsum()

            # # Filter for actual blinks
            # blink_durations = df.groupby("Blink ID")["Time Diff"].sum().reset_index()
            # valid_blinks = blink_durations.loc[blink_durations["Time Diff"] > blink_threshold].copy()
            # # Classify blink types
            # valid_blinks["Attention State"] = valid_blinks["Time Diff"].apply(
            #     lambda dur: "Blink" if dur <= 400 else "Loss of attention"
            # )
            # valid_blinks["Loss of Attention"] = valid_blinks["Time Diff"] > 400
            # df["Blink"] = df["Blink ID"].isin(valid_blinks["Blink ID"]).astype(int)
            
            # df["Loss of Attention"] = False
            # long_blink_ids = valid_blinks.loc[valid_blinks["Loss of Attention"], "Blink ID"]
            # df.loc[df["Blink ID"].isin(long_blink_ids), "Loss of Attention"] = True
            
            # all_blinks[task_id] = valid_blinks

            # # Clean up intermediate columns
            # df.drop(columns=["Missing Gaze", "Time Diff", "Blink Start", "Blink End", "Blink ID"], inplace=True)
            # updated_chunks[task_id] = df

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
        
        # Check which interpolate_cols are non-numeric
        bad = [c for c in interpolate_cols if c in df.columns and not pd.api.types.is_numeric_dtype(df[c])]
        if len(bad) > 0:
            print("Non-numeric columns among interpolate_cols:", bad)

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