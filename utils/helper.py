from typing import Union, Dict, Tuple, Optional, Iterable
from collections import defaultdict
from pathlib import Path
import os
import json

import pandas as pd

from utils.data_processing import EyeTrackingProcessor
from utils.task_data_io import PARQUET_NAME, list_parquet_files

def load_and_process(
    root_dir: Union[str, Path],
    columns: list[str],
    interpolate_cols: list[str],
    fill_cols: list[str],
    participants: Optional[Iterable[str]] = None,
    scenarios: Optional[Iterable[Union[str, int]]] = None,
    time_resampling: bool = True,
    fixed_window_ms: int | None = 3000,
    window_step_ms: int | None = None,
    min_task_presence: float = 0.5,
):
    # Get all data files
    file_index = list_parquet_files(root_dir, participants=participants, scenarios=scenarios)
    if not file_index:
        raise FileNotFoundError(f"No {PARQUET_NAME} found under {root_dir}")

    # Option: Only download the necessary columns
    needed = set(columns) | {
        "Event", "Participant name", "epoch_ms", "Recording timestamp [ms]"
    }
    
    # Load data
    processor = EyeTrackingProcessor()
    all_data, atco_task_map = processor.load_data(file_index, want_columns=list(needed))

    ###### Chunking Strategy ######
    if fixed_window_ms is not None:
        chunks = processor.get_fixed_window_chunks(
            all_data,
            features=columns,
            window_ms=fixed_window_ms,
            step_ms=window_step_ms,
            min_presence=min_task_presence,
        )
    else:
        chunks = processor.get_features(all_data, columns)

    # Blink detection
    chunks, blinks = processor.detect_blinks(chunks)

    # Time-step resampling
    if time_resampling:
        resampled_chunks_time = processor.resample_task_chunks(
            chunks, interpolate_cols, mode="time", param=10
        )
        # post-process: Blink back to bool + fill others
        for task_id, chunk in resampled_chunks_time.items():
            chunk["Blink"] = chunk["Blink"] > 0.5
            for col in fill_cols:
                chunk[col] = chunk[col].ffill().bfill()
        return resampled_chunks_time, blinks, atco_task_map

    return chunks, blinks, atco_task_map

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

def drop_chunks_with_all_zero_features(task_chunks: dict[str, pd.DataFrame], threshold: float = 1.0) -> dict[str, pd.DataFrame]:
    """
    Drops any DataFrame from the dict where at least one feature column has a great proportion of zero.
    
    Parameters
    ----------
    task_chunks : dict[str, pd.DataFrame]
        Dictionary with task ID as key and task data as value.
    threshold: float
        Proportion of zeros required to drop a feature
    
    Returns
    -------
    dict[str, pd.DataFrame]
        Filtered dictionary with problematic chunks removed.
    """
    feature_cols = ["Gaze point X", "Gaze point Y", "Mouse position X", "Mouse position Y"]
    cleaned_chunks = {}
    dropped_ids = []
    for task_id, df in task_chunks.items():
            present_cols = [col for col in feature_cols if col in df.columns]
            drop = False

            for col in present_cols:
                zero_ratio = (df[col] == 0).mean()  # proportion of zeros
                if zero_ratio >= threshold:
                    drop = True
                    break

            if drop:
                dropped_ids.append(task_id)
            else:
                cleaned_chunks[task_id] = df

    if dropped_ids:
        print(f"Dropped {len(dropped_ids)} chunks (threshold={threshold}):", dropped_ids)
    else:
        print(f"No chunks dropped (threshold={threshold}).")

    return cleaned_chunks