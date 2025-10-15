import os
import sys
from typing import Union, Dict, Tuple
import json
from pathlib import Path
# sys.path.append(str(Path('~/git_folder/eye_tracking/').expanduser()))
sys.path.append(str(Path('~/git/eye_tracking/').expanduser()))

import pandas as pd
from utils.data_processing import GazeMetricsProcessor, MouseMetricsProcessor
from utils.helper import load_and_process, find_overlapping_tasks, drop_chunks_with_all_zero_features

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table

import xgboost as xgb
import joblib 
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter(action="ignore", category=UserWarning)

# ------------------------- HELPER FUNCTION -------------------------

def detect_timestamp_column(df: pd.DataFrame) -> str:
    candidates = [col for col in df.columns if "Recording timestamp" in col]
    if not candidates:
        raise ValueError("No 'Recording timestamp' column found.")
    return candidates[0] 

def extract_tsfresh_features_from_chunks(
    task_chunks: dict[str, pd.DataFrame],
    columns_to_extract: list[str],
    pval_threshold: float = 0.05,
    n_jobs: int = 4
) -> pd.DataFrame:
    """
    Extract and select TSFresh features from a dictionary of task chunks.

    Parameters:
    - task_chunks: dict with `id` as key and corresponding df as value
    - columns_to_extract: list of features to use in TSFresh
    - pval_threshold: p-value threshold for relevance filtering
    - n_jobs: number of parallel jobs for TSFresh

    Returns:
    - final_features: DataFrame of filtered, selected features with `id` as index
    """
    
    # Step 0: Build labels per chunk from the ORIGINAL dict
    task_labels = pd.Series(
        {uid: int(df["Task_id"].iloc[0]) for uid, df in task_chunks.items()},
        name="Task_id"
    )

    
    # Step 1: Recombine into one DataFrame
    # Concatenate everything in one single df for TSFresh
    # Rows are dupplicated if we have overlapping tasks
    value_cols = columns_to_extract
    full_df = pd.concat(task_chunks.values(), ignore_index=True)
    timestamp_col = detect_timestamp_column(full_df)
    
    # Keep only what TSFresh needs, ensure numeric dtypes, and sort per chunk
    full_df = full_df[["id", timestamp_col] + value_cols].copy()
    full_df[value_cols] = full_df[value_cols].apply(pd.to_numeric, errors="coerce")
    full_df = full_df.sort_values(["id", timestamp_col])
    
    # If there are duplicate timestamps within an id, collapse them
    full_df = (full_df
            .groupby(["id", timestamp_col], as_index=False)[value_cols]
            .mean())
    
    # Fill remaining NaNs per chunk (id): ffill/bfill then linear interpolate (Chunks with only missing values have been dropped before)
    def _fill_group(g):
        g[value_cols] = (g[value_cols]
                        .ffill()
                        .bfill()
                        .interpolate(method="linear", limit_direction="both"))
        return g

    full_df = full_df.groupby("id", group_keys=False).apply(_fill_group)
    
    # If anything is still NaN, fill with per-id medians, then global fallback
    medians = full_df.groupby("id")[value_cols].transform("median")
    full_df[value_cols] = full_df[value_cols].fillna(medians)
    full_df[value_cols] = full_df[value_cols].fillna(full_df[value_cols].median())
    
    # 4) Assert clean
    leftover = full_df[value_cols].isna().sum().sum()
    assert leftover == 0, f"Still have {leftover} NaNs after cleaning."

    # Step 2: Run TSFresh
    print("Extracting TSFresh features...")
    extracted_features = extract_features(
        full_df,
        column_id="id",
        column_sort=timestamp_col,
        default_fc_parameters=MinimalFCParameters(), #Deactivate if full feature calculation
        n_jobs=n_jobs,
        disable_progressbar=False
    )

    # Step 3: Impute missing features
    impute(extracted_features)

    # Step 4: Relevance filtering
    relevant_features = calculate_relevance_table(extracted_features, task_labels)
    selected_features = relevant_features[relevant_features["p_value"] < pval_threshold]["feature"]
    
    # Step 5: Final filtered feature matrix
    final_features = extracted_features.loc[task_labels.index, selected_features].reset_index(names="id")

    return final_features

def sanitize_column_names(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = (
        df.columns.astype(str)
        .str.replace(r"[<>\[\]]", "_", regex=True)
        .str.replace(r"\s+", "_", regex=True)
    )
    return df

if __name__ == "__main__":
    # ------------------------- 0. PARAMETERS -------------------------
    
    # Directories
    # store_data_dir = str(Path('~/store/aware/training_data_raw_inputs').expanduser())
    store_dir = str(Path('~/store/aware').expanduser())
    store_raw_inputs_dir = os.path.join(store_dir, "training_data_raw_inputs")
    jcafnet_metadata_path = "logs/jcafnet_classifier/silvery-morning-8/jcafnet_metadata.json" # Change to suitable path
    save_model_path = "logs/xgboost_classifier"
    split_names = ["train", "val", "test"]
    store_splits_dir = os.path.join(store_dir, "splits")
    store_split_dirs = [os.path.join(store_splits_dir, split_name) for split_name in split_names]
    
    # Temporary storage for better I/O performance
    temp_data_dir = "/scratch/aware"
    temp_raw_inputs_dir = os.path.join(temp_data_dir, "training_data_raw_inputs")
    temp_splits_dir = os.path.join(temp_data_dir, "splits")
    temp_split_dirs = [os.path.join(temp_splits_dir, split_name) for split_name in split_names]
    
    #Features names
    features = ['Recording timestamp [ms]', 'Gaze point X [DACS px]', 'Gaze point Y [DACS px]', 'Mouse position X', 'Mouse position Y', 'Event']
    interpolate_cols = ['Gaze point X [DACS px]', 'Gaze point Y [DACS px]', 'Mouse position X', 'Mouse position Y', "Blink"]
    fill_columns = ['Gaze point X [DACS px]', 'Gaze point Y [DACS px]', 'Mouse position X', 'Mouse position Y']
    columns_to_extract = ['Gaze point X [DACS px]', 'Gaze point Y [DACS px]', 'Mouse position X', 'Mouse position Y'] # Columns for TSFresh
    cols_to_drop = ["Seconds per raw time unit", "Timestamp column", "Gaze X column", "Gaze Y column", "Mouse X column", "Mouse Y column"] # Columns to trop before XGBoost
    
    # Define Group K-Fold
    n_splits = 5
    
    # Define hyperparameter grid
    param_grid = {
        "n_estimators": [100, 200],          # Number of trees
        "max_depth": [4, 6, 8],              # Tree depth
        "learning_rate": [0.01, 0.1, 0.3],   # Learning rate
        "subsample": [0.8, 1.0],             # Subsampling ratio
        "colsample_bytree": [0.8, 1.0],      # Feature sampling ratio per tree
        "gamma": [0, 0.1, 0.3],              # Minimum loss reduction
    }
    
    # ------------------------- 1. LOADING DATASET -------------------------
    
    # Loading data from scratch without fixed timestep resampling
    # Keeping the id split for train/test/val made during the training of the JCAFNET
    
    # Loading data from scratch
    chunks_xgboost, blinks, atco_task_map = load_and_process(root_dir=temp_raw_inputs_dir, 
                                                             columns=features, 
                                                             interpolate_cols=interpolate_cols, 
                                                             fill_cols=fill_columns, 
                                                             time_resampling=False,
                                                             fixed_window_ms=10000, # Should be identical to jcafnet script
                                                             window_step_ms=2000, # Should be identical to jcafnet script
                                                             min_task_presence=0.5 # Should be identical to jcafnet script
                                                             )
    
    # Manually filling because not handled by load_and_process if time_resampling = False
    for task_id, chunk in chunks_xgboost.items():
        for col in fill_columns:
            chunks_xgboost[task_id][col] = chunks_xgboost[task_id][col].ffill().bfill()
    
    if Path(jcafnet_metadata_path).exists():
        print("Loading JCAFNet metadata for consistent train test split... ")
        with open(jcafnet_metadata_path) as f:
            metadata = json.load(f)
    else:
        raise FileNotFoundError(f"Metadata for train/test split do not exist in the specified path {jcafnet_metadata_path}.")
    
    train_ids = metadata["train_ids"]
    val_ids = metadata["val_ids"]
    test_ids = metadata["test_ids"]
    
    from collections import defaultdict, Counter
    print("Number of total occurences per task for all data: ")
    task_window_counts = Counter(int(df["Task_id"].iloc[0]) for df in chunks_xgboost.values())
    for task_id in sorted(task_window_counts):
        print(f"Task {task_id}: {task_window_counts[task_id]} windows")
    
    # Optional: finding tasks that are overlapping
    # overlaps = find_overlapping_tasks(chunks_xgboost)
    # for participant, overlapping_pairs in overlaps.items():
    #     print(f"Participant {participant} has overlapping tasks:")
    #     for t1, t2 in overlapping_pairs:
    #         print(f"  - {t1} overlaps with {t2}")
    
    # Dropping id that have one of the features to all zeros: additional safety barrier
    cleaned_chunks = drop_chunks_with_all_zero_features(chunks_xgboost,
                                                             feature_cols=columns_to_extract,
                                                             threshold=0.5)
    
    print("Number of total occurences per task after drop: ")
    task_window_counts = Counter(int(df["Task_id"].iloc[0]) for df in cleaned_chunks.values())
    for task_id in sorted(task_window_counts):
        print(f"Task {task_id}: {task_window_counts[task_id]} windows")

    # ------------------------- 2. MANUAL FEATURE EXTRACTION -------------------------

    gaze_metrics = []
    mouse_metrics = []
    
    for id, chunk in cleaned_chunks.items():
        
        # Extracting gaze metrics
        gaze_processor = GazeMetricsProcessor(chunk, timestamp_unit="ms")
        gaze_compute = gaze_processor.compute_all_metrics()
        gaze_compute.update({"id": id})
        gaze_compute.update({"participant_id": chunk["Participant name"].iloc[0]})
        gaze_compute.update({"Task_id": chunk["Task_id"].iloc[0]})
        gaze_metrics.append(gaze_compute)
        
        # Extracting mouse metrics
        mouse_processor = MouseMetricsProcessor(chunk, timestamp_unit="ms")
        mouse_compute = mouse_processor.compute_all_metrics()
        mouse_compute.update({"id": id})
        mouse_compute.update({"participant_id": chunk["Participant name"].iloc[0]})
        mouse_compute.update({"Task_id": chunk["Task_id"].iloc[0]})
        mouse_metrics.append(mouse_compute)
        
    gaze_metrics_df = pd.DataFrame(gaze_metrics)
    mouse_metrics_df = pd.DataFrame(mouse_metrics)

    # ------------------------- 3. AUTOMATIC FEATURE EXTRACTION -------------------------
    
    ### START DEBUG ###
    # def summarize_zero_nan_features_by_id(df, id_col="id"):
    #     """
    #     Returns a DataFrame summarizing the number of all-zero and all-NaN columns
    #     per sample group (e.g., per participant/task `id`).
    #     """
    #     results = []

    #     feature_cols = df.columns.difference([id_col])

    #     for sample_id, group in df.groupby(id_col):
    #         subset = group[feature_cols]

    #         # Check for all-zero columns
    #         all_zero_mask = (subset == 0).all(axis=0)

    #         # Check for all-NaN columns
    #         all_nan_mask = subset.isna().all(axis=0)

    #         results.append({
    #             "id": sample_id,
    #             "num_all_zeros": all_zero_mask.sum(),
    #             "num_all_nans": all_nan_mask.sum(),
    #             "num_rows": len(group),
    #             "total_features": len(feature_cols)
    #         })
    #     return pd.DataFrame(results)

    # summary_df = summarize_zero_nan_features_by_id(data_extraction)
    # print(summary_df.sort_values("num_all_zeros", ascending=False))
    ### END DEBUG ###
    
    # Run tsfresh feature extraction
    print("Extracting TSFresh features from full dataset...")
    tsfresh_data = extract_tsfresh_features_from_chunks(
        cleaned_chunks, 
        columns_to_extract, 
        pval_threshold=0.05, 
        n_jobs=100)

    # ------------------------- 4. BUILD DATASET -------------------------
    gaze = gaze_metrics_df.drop(columns=cols_to_drop, errors="ignore")
    mouse = mouse_metrics_df.drop(columns=cols_to_drop, errors="ignore")
    merge_keys = ["id", "participant_id", "Task_id"]
    xgboost_data = gaze.merge(mouse, on=merge_keys, how="inner", suffixes=("_gaze", "_mouse"))
    xgboost_data = xgboost_data.merge(tsfresh_data, on="id", how = "inner")

    
    #Train/test split based on the one for JCAFNet
    train_df = xgboost_data[xgboost_data["id"].isin(train_ids)].copy()
    val_df = xgboost_data[xgboost_data["id"].isin(val_ids)].copy()
    test_df = xgboost_data[xgboost_data["id"].isin(test_ids)].copy()
    
    # print("Number of total occurences per task for train: ")
    # task_window_counts = Counter(int(train_df["Task_id"].iloc[0]))
    # for task_id in sorted(task_window_counts):
    #     print(f"Task {task_id}: {task_window_counts[task_id]} windows")
    
    # print("Number of total occurences per task for val: ")
    # task_window_counts = Counter(int(val_df["Task_id"].iloc[0]))
    # for task_id in sorted(task_window_counts):
    #     print(f"Task {task_id}: {task_window_counts[task_id]} windows")
    
    # print("Number of total occurences per task for test: ")
    # task_window_counts = Counter(int(test_df["Task_id"].iloc[0]))
    # for task_id in sorted(task_window_counts):
    #     print(f"Task {task_id}: {task_window_counts[task_id]} windows")
    
    print("Saving XGboost train/test datasets")
    train_df.to_parquet(os.path.join(store_split_dirs[0], "train_xgboost.parquet"))
    test_df.to_parquet(os.path.join(store_split_dirs[2], "test_xgboost.parquet"))
    val_df.to_parquet(os.path.join(store_split_dirs[1], "val_xgboost.parquet"))
    
    # CROSS VALIDATION SPLIT BASED ON PARTICIPANT ID
    # columns that must not go into X
    leaky = [
        "id",               # chunk identifier
        "participant_id",   # used as groups
        "Task_id",          # the label
        "Scenario_id",      # metadata
        "Task_execution",   # metadata
    ]
    X_train = train_df.drop(columns=[c for c in leaky if c in train_df.columns], errors="ignore")
    X_train = X_train.apply(pd.to_numeric, errors="coerce")
    X_train = sanitize_column_names(X_train)
    
    y_train = train_df["Task_id"].astype(int)  
    groups_train = train_df["participant_id"].astype(str)

    # Define Group K-Fold
    gkf = GroupKFold(n_splits=n_splits)

    # Initialize XGBoost model
    num_class = int(pd.Series(y_train).nunique())
    xgb_model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=num_class,
        eval_metric="mlogloss",
        random_state=42
    )

    # Initialize GridSearchCV with Group K-Fold
    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="accuracy",
        cv=gkf,  # Ensures group-based splitting
        n_jobs=-1,  # Use all available CPUs
        verbose=2
    )

    # Perform grid search
    grid_search.fit(X_train, y_train, groups=groups_train)

    # Retrieve best model and accuracy
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_score = grid_search.best_score_

    # Save the best model
    os.makedirs(save_model_path, exist_ok=True)
    joblib.dump(best_model, os.path.join(save_model_path, "best_model.pkl"))
    joblib.dump(grid_search, os.path.join(save_model_path, "full_grid_search.pkl"))
    joblib.dump(X_train.columns.tolist(), os.path.join(save_model_path, "features.pkl"))

    # Display results
    print(f"\nBest Parameters: {best_params}")
    print(f"Best Cross-Validation Accuracy: {best_score:.4f}")