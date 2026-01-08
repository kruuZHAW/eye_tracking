import pandas as pd
import numpy as np
from pathlib import Path
import pymovements as pm

from collections import defaultdict

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

    # ------------------------- 2. TASK IDENTIFICATION & FILTERING -------------------------
    
    def task_range_finder(self, 
                          df: pd.DataFrame,
                          ) -> tuple[dict[str, list[tuple[int, int]]], pd.DataFrame]:
        """Find start and end times of tasks within a DataFrame and record unmatched markers.
        
        Apply filtering for extremely long or short tasks. 
        
        """
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
    
    def compute_task_duration_bounds(
        self,
        dfs: list[pd.DataFrame],
        *,
        min_duration_ms: int = 500,
        iqr_multiplier: float = 3.0,
        global_max_ms: int | None = None,
    ) -> dict[int, tuple[int, int]]:
        """
        Compute global per-task duration bounds across all participants/scenarios.

        Returns
        -------
        duration_bounds_by_task : dict[int, (lower_ms, upper_ms)]
            For each task_id, a (lower, upper) duration in ms.

        Steps:
        - Step 1: enforce min_duration_ms (drop shorter durations)
        - Step 2: per-task upper bound = Q3 + iqr_multiplier * IQR
        - Optional: clip upper bound to global_max_ms if provided.
        """
        per_task_durations: dict[int, list[int]] = defaultdict(list)

        # First pass: collect all durations per task_id
        for df in dfs:
            if df.empty:
                continue

            # Ensure timestamp column known
            self.detect_timestamp_column(df)

            task_ranges, _ = self.task_range_finder(df)
            for task_name, periods in task_ranges.items():
                try:
                    task_id = int(task_name.split()[-1])
                except Exception:
                    # If not of the form "Task N", skip
                    continue

                for start, end in periods:
                    dur = int(end - start)
                    per_task_durations[task_id].append(dur)

        # Second pass: compute bounds
        duration_bounds_by_task: dict[int, tuple[int, int]] = {}

        for task_id, durs in per_task_durations.items():
            arr = np.asarray(durs, dtype=float)

            # Step 1: minimal duration
            arr = arr[arr >= min_duration_ms]
            if arr.size == 0:
                # no valid durations left for this task
                continue

            # Step 2: IQR-based upper cutoff
            Q1 = np.percentile(arr, 25)
            Q3 = np.percentile(arr, 75)
            IQR = Q3 - Q1
            upper = Q3 + iqr_multiplier * IQR

            # Optional global hard cap
            if global_max_ms is not None:
                upper = min(upper, global_max_ms)

            # Lower bound: at least min_duration_ms
            lower = max(min_duration_ms, int(Q1 - 1.5 * IQR))
            duration_bounds_by_task[task_id] = (int(lower), int(upper))

        return duration_bounds_by_task
    
    def apply_duration_bounds_to_ranges(
        self,
        task_ranges: dict[str, list[tuple[int, int]]],
        duration_bounds_by_task: dict[int, tuple[int, int]],
    ) -> dict[str, list[tuple[int, int]]]:
        """
        Filter a single participant/scenario's task_ranges using global per-task
        duration bounds.

        Parameters
        ----------
        task_ranges : dict[str, list[(start_ms, end_ms)]]
            Output of task_range_finder for one df.
        duration_bounds_by_task : dict[int, (lower_ms, upper_ms)]
            Output of compute_task_duration_bounds.

        Returns
        -------
        filtered_task_ranges : dict[str, list[(start_ms, end_ms)]]
        """
        if not duration_bounds_by_task:
            return task_ranges  

        filtered_task_ranges: dict[str, list[tuple[int, int]]] = {}

        for task_name, periods in task_ranges.items():
            try:
                task_id = int(task_name.split()[-1])
            except Exception:
                # Unknown naming, keep as-is
                filtered_task_ranges[task_name] = periods
                continue

            bounds = duration_bounds_by_task.get(task_id)
            if bounds is None:
                # No bounds for this task_id → keep all periods
                filtered_task_ranges[task_name] = periods
                continue

            lower, upper = bounds
            kept: list[tuple[int, int]] = []
            for (start, end) in periods:
                dur = int(end - start)
                if dur >= lower and dur <= upper:
                    kept.append((start, end))

            if kept:
                filtered_task_ranges[task_name] = kept

        return filtered_task_ranges

    # ------------------------- 3. TASK WINDOWS CONSTRUCTION -------------------------
    
    def get_full_tasks(
        self,
        dfs: list[pd.DataFrame],
        features: list[str], 
        unmatched_excel_path: str | None = None,
        *,
        filter_outliers: bool = False,
        min_duration_ms: int = 500,
        iqr_multiplier: float = 3.0,
        global_max_ms: int | None = None,
    ) -> dict[str, pd.DataFrame]:
        """
        Extracts per-task chunks from each DataFrame (for each participant) and returns a dict where:
        - key = "participant_task_execution" id
        - value = corresponding dataframe chunk (can overlap)
        
        If filter_outliers=True, apply compute_task_duration_bounds and apply_duration_bounds_to_ranges:

        Overlapping task intervals are handled: a row can appear in multiple chunks.
        Write an Excel file that identifies all missing markers
        """

        task_chunks: dict[str, pd.DataFrame] = {}
        writer = None
        used_sheet_names: set[str] = set()
        
        # ---- Compute global bounds filtering once, if requested ----
        if filter_outliers:
            duration_bounds_by_task = self.compute_task_duration_bounds(
                dfs,
                min_duration_ms=min_duration_ms,
                iqr_multiplier=iqr_multiplier,
                global_max_ms=global_max_ms,
            )
        else:
            duration_bounds_by_task = {}

        if unmatched_excel_path:
            Path(unmatched_excel_path).parent.mkdir(parents=True, exist_ok=True)
            writer = pd.ExcelWriter(unmatched_excel_path, engine="xlsxwriter")

        try:
            for df in dfs:
                participant = df["participant_id"].iloc[0] if "participant_id" in df.columns else df["Participant name"].iloc[0]
                scenario_id = df["scenario_id"].iloc[0]
                print(f"Finding tasks for participant {participant} Scenario {scenario_id}")

                task_ranges, unmatched_df = self.task_range_finder(df)
                # Apply filtering if required
                if filter_outliers and duration_bounds_by_task:
                    task_ranges = self.apply_duration_bounds_to_ranges(
                        task_ranges,
                        duration_bounds_by_task,
                    )

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
    
    
    def get_multiscale_window_chunks(
        self,
        dfs: list[pd.DataFrame],
        features: list[str],
        window_short_ms: int = 3000,
        window_mid_ms: int = 10000,
        window_long_ms: int = 25000,
        task_margin_ms: int = 2000,
        step_ms: int = 3000,
        idle_id: int = -1,
        label_idle: bool = True,
        *,
        filter_outliers: bool = False,
        min_duration_ms: int = 500,
        iqr_multiplier: float = 3.0,
        global_max_ms: int | None = None,
    ) -> dict[str, dict[str, pd.DataFrame]]:
        """
        Slice each participant stream into *multi-scale*, end-anchored windows.
        For each anchor time t (discretized every `step_ms`), we build:
        - short window: [t - short_ms, t)
        - mid   window: [t - mid_ms,   t)
        - long  window: [t - long_ms,  t)
        All three windows share the same label, determined from the task
        *active at time t* (if any), else idle.

        - window_sort_ms: short window length in milliseconds (e.g., 3000 for 3s).
        - window_mid_ms: medium window length in milliseconds.
        - window_long_ms: long window length in milliseconds.
        - task_margin_ms: margin in milliseconds for the assignement of the task
        - step_ms: hop size in milliseconds.

        Output format:
        chunks[sample_id] = {
            "short": df_short,
            "mid":   df_mid,
            "long":  df_long,
        }
        where each df_* has the selected `features` + metadata columns ["Task_id", "Task_execution", "Participant name", "id", "Task_*_proportion"]
        """
        chunks: dict[str, dict[str, pd.DataFrame]] = {}
        
        if filter_outliers:
            duration_bounds_by_task = self.compute_task_duration_bounds(
                dfs,
                min_duration_ms=min_duration_ms,
                iqr_multiplier=iqr_multiplier,
                global_max_ms=global_max_ms,
            )
        else:
            duration_bounds_by_task = {}

        for df in dfs:
            # Participant id,  Scenario id
            participant = df["participant_id"].iloc[0] if "participant_id" in df.columns else df["Participant name"].iloc[0]
            scenario_id = df["scenario_id"].iloc[0]

            # Timestamp detection + normalization (ms)
            self.detect_timestamp_column(df)
            df_sorted = df.sort_values(by=self.timestamp_col).reset_index(drop=True)
            median_diff = df_sorted[self.timestamp_col].diff().dropna().median()
            if pd.notna(median_diff) and median_diff > 10000:
                # µs -> ms
                df_sorted[self.timestamp_col] = df_sorted[self.timestamp_col] / 1000.0

            # Build task ranges
            print(f"Finding tasks for participant {participant} Scenario {scenario_id}")
            task_ranges, _ = self.task_range_finder(df_sorted)  # {"Task N": [(start,end), ...]}
            if filter_outliers and duration_bounds_by_task:
                task_ranges = self.apply_duration_bounds_to_ranges(
                    task_ranges,
                    duration_bounds_by_task,
                )
            
            # Precompute a flat list with (task_name, occ_index, start, end) for quick overlap checks
            flat_ranges: list[tuple[str, int, float, float]] = []
            for task_name, periods in task_ranges.items():
                for occ_idx, (st, en) in enumerate(periods):
                    flat_ranges.append((task_name, occ_idx, float(st), float(en)))

            if not flat_ranges:
                # No tasks found -> skip this participant entirely
                continue

            # Time domain and anchor times
            t_min = float(df_sorted[self.timestamp_col].min())
            t_max = float(df_sorted[self.timestamp_col].max())
            if not np.isfinite(t_min) or not np.isfinite(t_max) or t_max <= t_min:
                continue
            
            anchor_start = t_min + window_long_ms #the longest window fit entirely in [t_min, t_max]
            if anchor_start > t_max:
                continue
            anchor_times = np.arange(anchor_start, t_max + 1e-9, step_ms)
            
            # Occurrence counter to give unique ids per (participant, scenario, task_id).
            per_task_sample_counter: dict[int, int] = {}
            
            ts_col = self.timestamp_col

            for t_anchor in anchor_times:
                t_label = t_anchor
                
                # 1) Determine label from active task at t_label
                task_name_at_t = None
                occ_idx_at_t = None
                
                # Find first task interval containing t_label
                for task_name, occ_idx, start, end in flat_ranges:
                    # Apply task_margin to avoid labelling windows with tasks that just began
                    if start <= t_label - task_margin_ms < end:
                        task_name_at_t = task_name
                        occ_idx_at_t = occ_idx
                        break
                    
                if task_name_at_t is None:
                    if not label_idle:
                        continue
                    task_id = idle_id
                    task_exec = -1
                else:
                    task_id = int(task_name_at_t.split()[-1])
                    task_exec = occ_idx_at_t
                    
                # 2) build short / mid / long windows ending at t_anchor
                def slice_window(length_ms:int) -> pd.DataFrame:
                    start = t_anchor - length_ms
                    mask = (df_sorted[ts_col] >= start) & (df_sorted[ts_col] < t_anchor) #exlusive end
                    return df_sorted.loc[mask, features].copy()
                short_df = slice_window(window_short_ms)
                mid_df = slice_window(window_mid_ms)
                long_df = slice_window(window_long_ms)
                # If one of them is empty, discard
                if short_df.empty or mid_df.empty or long_df.empty:
                    continue
                
                # 3) Build a unique sample ID
                sample_index = per_task_sample_counter.get(task_id, 0)
                per_task_sample_counter[task_id] = sample_index + 1
                uid = f"{participant}_{scenario_id}_{task_id}_{sample_index}"

                # 4) Attach metadata to each window
                # OPTIONAL TODO: ADD THE PROPORTION OF EACH TASK PRESENT IN THE WINDOW
                for wdf in (short_df, mid_df, long_df):
                    if wdf.empty:
                        continue
                    wdf["Task_id"] = task_id
                    wdf["Task_execution"] = task_exec
                    wdf["Participant name"] = participant
                    wdf["Scenario_id"] = scenario_id
                    wdf["id"] = uid
                    # Optional: store anchor time or label time for later
                    wdf["anchor_time"] = t_anchor
                    wdf["label_time"] = t_label

                chunks[uid] = {
                    "short": short_df,
                    "mid":   mid_df,
                    "long":  long_df,
                }


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

    
    # ------------------------- 4. HELPERS BLINK IDENTIFICATION -------------------------
    
    def _detect_blinks_in_scenario(self, df: pd.DataFrame, blink_threshold: int = 100):
        """
        Detect blinks in a single continuous DataFrame: Aimed to be applied on full scenario DF.

        - A blink is a contiguous run where either gaze X or Y is NaN.
        - We mark Blink=True ONLY on rows inside runs whose duration > blink_threshold (ms).
        - Runs > 400 ms are also flagged as Loss of Attention.

        Returns
        -------
        updated_df : pd.DataFrame
            Same as input but with 'Blink' and 'Loss of Attention' columns added.
        summary_df : pd.DataFrame
            One row per missing-gaze run (that is blink or loss_of_attention).
        """
        if df.empty:
            # return empty summary but with columns
            df = df.copy()
            df["Blink"] = False
            df["Loss of Attention"] = False
            summary = pd.DataFrame(
                columns=["run_id", "duration_ms", "start_ts", "end_ts", "blink", "loss_of_attention"]
            )
            return df, summary

        df = df.copy()

        # Detect gaze columns
        x_col = next((col for col in df.columns if "Gaze point X" in col), None)
        y_col = next((col for col in df.columns if "Gaze point Y" in col), None)

        if not x_col or not y_col:
            # No gaze columns: just add flags as False and empty summary
            df["Blink"] = False
            df["Loss of Attention"] = False
            summary = pd.DataFrame(
                columns=["run_id", "duration_ms", "start_ts", "end_ts", "blink", "loss_of_attention"]
            )
            return df, summary

        self.detect_timestamp_column(df)
        df = df.sort_values(by=self.timestamp_col).reset_index(drop=True)
        med_diff = df[self.timestamp_col].diff().dropna().median()
        if pd.notna(med_diff) and med_diff > 10_000:
            df[self.timestamp_col] = df[self.timestamp_col] / 1_000.0  # µs → ms

        # Missing gaze mask
        df["Missing Gaze"] = df[x_col].isna() | df[y_col].isna()

        # Blink run IDs: increment when Missing Gaze changes (False->True or True->False)
        df["_run"] = df["Missing Gaze"].ne(df["Missing Gaze"].shift(fill_value=False)).cumsum()

        # Duration per run (sum of diffs) using only missing-gaze rows
        dur_by_run = (
            df.loc[df["Missing Gaze"], [self.timestamp_col, "_run"]]
            .groupby("_run")[self.timestamp_col]
            .apply(lambda s: (s.diff().fillna(0)).sum())
        )

        # Which runs count as blinks/attention loss
        if dur_by_run.empty:
            valid_runs = pd.Index([])
            long_runs = pd.Index([])
        else:
            valid_runs = dur_by_run[dur_by_run > blink_threshold].index
            long_runs  = dur_by_run[dur_by_run > 400].index

        # Initialize flags
        df["Blink"] = False
        df["Loss of Attention"] = False

        # Mark only rows inside the selected runs
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
            summary = summary.loc[summary["blink"] | summary["loss_of_attention"]].copy()
        else:
            summary = pd.DataFrame(
                columns=["run_id", "duration_ms", "start_ts", "end_ts", "blink", "loss_of_attention"]
            )

        # Clean up helpers
        df.drop(columns=["_run", "Missing Gaze"], inplace=True)

        return df, summary
    
    def detect_blinks_in_streams(
        self,
        dfs: list[pd.DataFrame],
        blink_threshold: int = 100,
    ) -> tuple[list[pd.DataFrame], dict[str, pd.DataFrame]]:
            """
            Detect blinks on a list of continuous scenario/participant DataFrames.

            This is intended for use *before* windowing (e.g. before get_multiscale_window_chunks or get_full_tasks),
            so that the Blink/Loss of Attention flags are present in the raw streams and
            automatically carried into any windows/chunks.

            Returns
            -------
            updated_dfs : list[pd.DataFrame]
                Same structure as input `dfs`, each df with Blink/Loss of Attention columns.
            all_blinks : dict[str, pd.DataFrame]
                Mapping from a scenario key to its blink summary DataFrame.
                Key format: "<participant>_<scenario>" if those columns exist, else "stream_i".
            """
            updated_dfs: list[pd.DataFrame] = []
            all_blinks: dict[str, pd.DataFrame] = {}

            for i, df in enumerate(dfs):
                if df.empty:
                    updated_dfs.append(df)
                    all_blinks[f"stream_{i}"] = pd.DataFrame(
                        columns=["run_id", "duration_ms", "start_ts", "end_ts", "blink", "loss_of_attention"]
                    )
                    continue

                # Build a readable key
                if "participant_id" in df.columns:
                    participant = str(df["participant_id"].iloc[0])
                else:
                    participant = str(df.get("Participant name", pd.Series(["unknown"])).iloc[0])

                scenario_id = str(df.get("scenario_id", pd.Series(["unknown"])).iloc[0])
                key = f"{participant}_{scenario_id}"

                updated_df, summary_df = self._detect_blinks_in_scenario(df, blink_threshold=blink_threshold)
                updated_dfs.append(updated_df)
                all_blinks[key] = summary_df

            return updated_dfs, all_blinks

    
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
        
        # Not necessary as the resolution is at most 120HZ wich maks one nan already a blink (above 0.1s)
        # if interpolate_short_gaps:
        #     self._interpolate_short_gaps(max_gap_len=2)

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
    
    def _estimate_sampling_rate_hz(self) -> float:
        ts_s = self._to_seconds(self.df[self.ts_col])
        dt = ts_s.diff().dropna()
        dt = dt[dt > 0]
        if dt.empty:
            return 120.0
        return float(1.0 / dt.median())
    
    def _ms_to_samples(self, ms: float) -> int:
        sr = self._estimate_sampling_rate_hz()
        return max(1, int(np.ceil((ms / 1000.0) * sr)))
    
    def _interpolate_short_gaps(self, max_gap_len: int = 2):
        # Only interpolate very small runs of NaNs in gaze, not longer ones which are blinks
        for col in [self.gx_col, self.gy_col]:
            s = self.df[col]

            # limit = max number of consecutive NaNs we allow interpolation over
            self.df[col] = s.interpolate(
                method="linear",
                limit=max_gap_len,
                limit_direction="both"
        )
    
    def _valid_segments(self):
        """Yield (start_idx, segment_df) where gx/gy are both not nans."""
        valid = self.df[self.gx_col].notna() & self.df[self.gy_col].notna()

        if not valid.any():
            return

        # contiguous runs
        run_id = valid.ne(valid.shift(fill_value=False)).cumsum()
        for _, chunk in self.df[valid].groupby(run_id[valid]):
            start_idx = int(chunk.index[0])
            yield start_idx, chunk

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
        
    def compute_fixation_statistics_pm_idt(self, time_threshold=100, dispersion_threshold_px=50):
        """
        I-DT via pymovements. Thresholds:
          - time_threshold in ms 
          - dispersion_threshold_px in px 
        """

        dur_samples = self._ms_to_samples(time_threshold)
        sr_hz = self._estimate_sampling_rate_hz()

        fix_durations_s = []

        for start_idx, seg in self._valid_segments():
            
            if len(seg) < max(2, dur_samples):
                continue
            
            positions = seg[[self.gx_col, self.gy_col]].to_numpy(dtype=float)

            # Run I-DT on this contiguous valid segment
            res = pm.events.idt(
                positions=positions,
                dispersion_threshold=float(dispersion_threshold_px),
                minimum_duration=int(dur_samples),
            )

            # Each fixation has onset/offset in *sample indices* relative to this segment
            for fix_dur in res.fixations["duration"]:
                fix_durations_s.append(fix_dur / sr_hz)

        fixation_count = len(fix_durations_s)
        total_fix_dur_s = float(np.sum(fix_durations_s)) if fixation_count else 0.0
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

    def compute_saccade_statistics_pm(
        self,
        radius_threshold: float = 50,
        min_duration_ms: float = 20,
        threshold_factor: float = 6.0,
        min_vel_std: float = 1e-8,   # variance gate for pm
    ):
        """
        If velocity variance looks good -> use pymovements microsaccades detector.
        Otherwise -> fall back to the simple displacement-threshold method.

        Returns
        -------
        saccade_count, avg_saccade_amplitude_px, avg_saccade_velocity_px_per_s
        """
        sr_hz = self._estimate_sampling_rate_hz()
        min_samples = max(2, int(np.ceil((min_duration_ms / 1000.0) * sr_hz)))

        # --------- quick global variance gate ----------
        # Build a velocity array from all valid points (ignoring NaNs).
        valid = self.df[self.gx_col].notna() & self.df[self.gy_col].notna()
        if valid.sum() < (min_samples + 1):
            return self.compute_saccade_statistics(radius_threshold=radius_threshold)

        pos_all = self.df.loc[valid, [self.gx_col, self.gy_col]].to_numpy(dtype=float)
        v_all = np.diff(pos_all, axis=0) * sr_hz

        # If either axis has ~zero variance, pm microsaccades will often fail
        if (np.nanstd(v_all[:, 0]) < min_vel_std) or (np.nanstd(v_all[:, 1]) < min_vel_std):
            return self.compute_saccade_statistics(radius_threshold=radius_threshold)

        # --------- run pymovements on segments ----------
        amps = []
        mean_vels = []

        for _, seg in self._valid_segments():
            if len(seg) < (min_samples + 1):
                continue

            pos = seg[[self.gx_col, self.gy_col]].to_numpy(dtype=float)
            v = np.diff(pos, axis=0) * sr_hz

            # segment-level gate (prevents the [0, 500] kind of error)
            if (np.nanstd(v[:, 0]) < min_vel_std) or (np.nanstd(v[:, 1]) < min_vel_std):
                # if ANY segment is degenerate, just use the simple method for consistency
                return self.compute_saccade_statistics(radius_threshold=radius_threshold)

            try:
                sacc = pm.events.microsaccades(
                    velocities=v,
                    threshold="engbert2015",
                    threshold_factor=float(threshold_factor),
                    minimum_duration=int(min_samples),
                )
            except ValueError:
                # pm refused (usually due to variance/threshold issues) -> simple fallback
                return self.compute_saccade_statistics(radius_threshold=radius_threshold)

            # Extract events (handle both output shapes)
            if hasattr(sacc, "saccades"):
                onsets = np.asarray(sacc.saccades["onset"], dtype=int)
                durs   = np.asarray(sacc.saccades["duration"], dtype=int)
                for onset_v, dur in zip(onsets, durs):
                    if dur < min_samples:
                        continue
                    offset_v = onset_v + dur

                    onset_p = onset_v
                    offset_p = min(offset_v, len(pos) - 1)

                    amp = float(np.linalg.norm(pos[offset_p] - pos[onset_p]))
                    speed = np.linalg.norm(v[onset_v:offset_v], axis=1)
                    mean_speed = float(np.nanmean(speed)) if speed.size else 0.0

                    amps.append(amp)
                    mean_vels.append(mean_speed)
            else:
                for ev in sacc:
                    onset_v = int(ev.onset)
                    offset_v = int(ev.offset)
                    if (offset_v - onset_v) < min_samples:
                        continue

                    onset_p = onset_v
                    offset_p = min(offset_v + 1, len(pos) - 1)

                    amp = float(np.linalg.norm(pos[offset_p] - pos[onset_p]))
                    speed = np.linalg.norm(v[onset_v:offset_v], axis=1)
                    mean_speed = float(np.nanmean(speed)) if speed.size else 0.0

                    amps.append(amp)
                    mean_vels.append(mean_speed)

        # If pm found nothing, return zeros (don't fall back; that would mix definitions)
        saccade_count = len(amps)
        avg_amp = float(np.mean(amps)) if saccade_count else 0.0
        avg_vel = float(np.mean(mean_vels)) if saccade_count else 0.0
        return saccade_count, avg_amp, avg_vel


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
        # fix_n, fix_tot_s, fix_avg_s = self.compute_fixation_statistics()
        fix_n, fix_tot_s, fix_avg_s = self.compute_fixation_statistics_pm_idt()
        # sac_n, sac_amp_px, sac_vel = self.compute_saccade_statistics()
        sac_n, sac_amp_px, sac_vel = self.compute_saccade_statistics_pm()
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
            # "Seconds per raw time unit": self.seconds_per_unit,
            # "Timestamp column": self.ts_col,
            # "Gaze X column": self.gx_col,
            # "Gaze Y column": self.gy_col,
        }