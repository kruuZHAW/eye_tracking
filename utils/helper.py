from typing import Union, Dict, Tuple, Optional, Iterable
from collections import defaultdict
from pathlib import Path
import os
import json

import pandas as pd

from utils.data_processing_gaze_data import EyeTrackingProcessor
from utils.task_data_io import PARQUET_ET_NAME, list_parquet_files

def load_and_process_et(
    root_dir: Union[str, Path],
    columns: list[str],
    filter_outliers: bool,
    interpolate_cols: list[str],
    fill_cols: list[str],
    window_short_ms: int = None,
    window_mid_ms: int = None,
    window_long_ms: int = None,
    task_margin_ms: int = None,
    step_ms: int = None, 
    participants: Optional[Iterable[str]] = None,
    scenarios: Optional[Iterable[Union[str, int]]] = None,
    time_resampling: bool = False,
):
    # Get all data files
    et_file_index, _ = list_parquet_files(root_dir, participants=participants, scenarios=scenarios)
    if not et_file_index:
        raise FileNotFoundError(f"No {PARQUET_ET_NAME} found under {root_dir}")

    # Option: Only download the necessary columns
    needed = set(columns) | {
        "Event", "Participant name", "epoch_ms", "Recording timestamp [ms]"
    }
    
    # Load data
    processor = EyeTrackingProcessor()
    scenarios, atco_task_map = processor.load_data(et_file_index, want_columns=list(needed))
    
    # Blink detection
    scenarios_with_blinks, blink_summaries = processor.detect_blinks_in_streams(scenarios)

    ###### Chunking Strategy ######
    if (window_short_ms is not None) & (window_mid_ms is not None) & (window_long_ms is not None):
        chunks = processor.get_multiscale_window_chunks(
            scenarios_with_blinks,
            features=columns + ["Blink", "Loss of Attention"],
            window_short_ms = window_short_ms,
            window_mid_ms = window_mid_ms,
            window_long_ms = window_long_ms,
            task_margin_ms=task_margin_ms,
            step_ms=step_ms,
            filter_outliers=filter_outliers,
        )
    else:
        chunks = processor.get_full_tasks(scenarios_with_blinks, 
                                          columns + ["Blink", "Loss of Attention"], 
                                          unmatched_excel_path="unmatched_markers.xlsx", 
                                          filter_outliers=filter_outliers)

    # Time-step resampling
    # TODO: debug resampling 'dict' object has no attribute 'sort_values'
    # if time_resampling:
    #     resampled_chunks_time = processor.resample_task_chunks(
    #         chunks, interpolate_cols, mode="time", param=10
    #     )
    #     # post-process: Blink back to bool + fill others
    #     for task_id, chunk in resampled_chunks_time.items():
    #         chunk["Blink"] = chunk["Blink"] > 0.5
    #         for col in fill_cols:
    #             chunk[col] = chunk[col].ffill().bfill()
    #     return resampled_chunks_time, blink_summaries, atco_task_map

    return chunks,blink_summaries, atco_task_map

def load_asd_scenario_data(root_dir: Union[str, Path]) -> dict[str, pd.DataFrame]:
    """
    Load ASD events parquet files on the scenario level
    """
    
    dfs: dict[str, pd.DataFrame] = {}
    
    _, asd_files = list_parquet_files(root_dir)
        
    for item in asd_files:
        p = item["path"]
        df = pd.read_parquet(p)

        df = df.copy()
        df["participant_id"] = str(item["participant_id"])
        df["scenario_id"] = str(item["scenario_id"])
        id = f"{df["participant_id"].iloc[0]}_{df["scenario_id"].iloc[0]}"

        dfs[id] = df

    return dfs

def save_processed_data(save_dir: Union[str, Path], 
                        chunks: dict, 
                        blinks: dict = None, 
                        task_map: dict = None):
    os.makedirs(save_dir, exist_ok=True)

    # Save each chunk dataframe
    chunks_dir = os.path.join(save_dir, "chunks")
    os.makedirs(chunks_dir, exist_ok=True)

    for chunk_id, df in chunks.items():
        df.to_parquet(os.path.join(chunks_dir, f"{chunk_id}.parquet"))

    # Save blinks if provided
    if blinks is not None:
        blinks_dir = os.path.join(save_dir, "blinks")
        os.makedirs(blinks_dir, exist_ok=True)
        for blink_id, df in blinks.items():
            df.to_parquet(os.path.join(blinks_dir, f"{blink_id}.parquet"))

    # Save task map if provided
    if task_map is not None:
        with open(os.path.join(save_dir, "atco_task_map.json"), "w") as f:
            json.dump(task_map, f)
        
def load_processed_data(save_dir: Union[str, Path]):
    chunks_dir = os.path.join(save_dir, "chunks")
    blinks_dir = os.path.join(save_dir, "blinks")
    task_map_path = os.path.join(save_dir, "atco_task_map.json")

    chunks = {}
    blinks = None
    task_map = None

    # Load chunks (required)
    for file in os.listdir(chunks_dir):
        if file.endswith(".parquet"):
            chunk_id = file.replace(".parquet", "")
            chunks[chunk_id] = pd.read_parquet(os.path.join(chunks_dir, file))

    # Load blinks if available
    if os.path.exists(blinks_dir):
        blinks = {}
        for file in os.listdir(blinks_dir):
            if file.endswith(".parquet"):
                blink_id = file.replace(".parquet", "")
                blinks[blink_id] = pd.read_parquet(os.path.join(blinks_dir, file))

    # Load task map if available
    if os.path.exists(task_map_path):
        with open(task_map_path, "r") as f:
            task_map = json.load(f)

    return chunks, blinks, task_map

def find_overlapping_tasks(task_chunks: dict[str, pd.DataFrame]) -> dict[int, list[tuple[str, str]]]:
    # Organize tasks per participant
    participant_tasks = defaultdict(list)

    for task_id, df in task_chunks.items():
        if df.empty:
            continue

        participant = df["Participant name"].iloc[0]
        start_time = df["Recording timestamp [ms]"].min()
        end_time = df["Recording timestamp [ms]"].max()
        participant_tasks[participant].append((task_id, start_time, end_time))

    # Detect overlaps/nesting
    overlaps_per_participant = defaultdict(list)

    for participant, tasks in participant_tasks.items():
        n = len(tasks)
        for i in range(n):
            id1, start1, end1 = tasks[i]
            for j in range(i + 1, n):
                id2, start2, end2 = tasks[j]

                # Check for overlap
                if start1 <= end2 and start2 <= end1:
                    overlaps_per_participant[participant].append((id1, id2))

    return overlaps_per_participant

def drop_chunks_with_nan_et(multiscale_chunks: dict[str, dict[str, pd.DataFrame]], 
                                threshold: float = 1.0, 
                                drop_if_all_missing: bool = True
                                ) -> dict[str, dict[str, pd.DataFrame]]:
    """
    For a multi-level dict of windows:
        { uid: {"short": df_short, "mid": df_mid, "long": df_long} }

    Drop a uid if *any* of its windows ("short", "mid", or "long") has
    a proportion of Nan values greater than *thresolhod* in the eye tracking data.

    Notes
    -----
    - Zeros are treated as valid values (only NaNs matter).
    - If a window DataFrame is empty, it is treated as invalid -> sample dropped.
    - If none of `feature_cols` are present in a window, that window is ignored
      for the all-NaN check (you can tighten this if you want).
    """
    cleaned_chunks: dict[str, dict[str, pd.DataFrame]] = {}
    dropped_ids: list[str] = []

    window_keys = ("short", "mid", "long")

    for uid, windows in multiscale_chunks.items():
        drop = False
        
        for wname in window_keys:
            df = windows.get(wname)
            
            # Already checked during built, but safe proof
            if df is None or df.empty:
                drop = True
                break
            
            sub = df[['Gaze point X [DACS px]', 'Gaze point Y [DACS px]']]
            
            # Proportion where both gaze X and Y are nans
            joint_nan = (sub.isna().all(axis=1)).mean()
            if joint_nan > threshold:
                drop = True
                break
        
        if drop:
            dropped_ids.append(uid)
        else:
            cleaned_chunks[uid] = windows
        
    if dropped_ids:
        print(
        f"Dropped {len(dropped_ids)} multi-scale samples due to a proportion of Nans greater than {threshold} in eye-tracking window(s): "
        f"{dropped_ids[:10]}{' ...' if len(dropped_ids) > 10 else ''}"
    )
    else:
        print("No multi-scale samples dropped (threshold={threshold}).")
        
    return cleaned_chunks