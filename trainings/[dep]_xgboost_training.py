import os
import sys
from typing import Union, Dict, Tuple
import json
from pathlib import Path
# sys.path.append(str(Path('~/git_folder/eye_tracking/').expanduser()))
sys.path.append(str(Path('~/git/eye_tracking/').expanduser()))

import numpy as np
import pandas as pd
from utils.data_processing_gaze_data import GazeMetricsProcessor, MouseMetricsProcessor
from utils.helper import load_and_process, find_overlapping_tasks, drop_chunks_with_all_zero_features

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table

import xgboost as xgb
import joblib 
from sklearn.model_selection import GroupKFold, StratifiedGroupKFold, GridSearchCV, GroupShuffleSplit
from sklearn.metrics import accuracy_score
from sklearn import metrics
from sklearn.base import clone

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

def split_by_participant(df: pd.DataFrame,
                         group_col="participant_id",
                         test_size=0.2,
                         val_size=0.1,
                         random_state=42):
    # Split off test by participant
    gss = GroupShuffleSplit(n_splits=1, test_size=test_size, random_state=random_state)
    trainval_idx, test_idx = next(gss.split(df, groups=df[group_col]))
    df_trainval = df.iloc[trainval_idx].copy()
    df_test     = df.iloc[test_idx].copy()

    # Split train vs val from the remaining participants
    val_prop = val_size / (1.0 - test_size)
    gss2 = GroupShuffleSplit(n_splits=1, test_size=val_prop, random_state=random_state)
    train_idx, val_idx = next(gss2.split(df_trainval, groups=df_trainval[group_col]))
    df_train = df_trainval.iloc[train_idx].copy()
    df_val   = df_trainval.iloc[val_idx].copy()

    # Return the participant IDs per split
    parts = {
        "train_parts": sorted(df_train[group_col].unique().tolist()),
        "val_parts":   sorted(df_val[group_col].unique().tolist()),
        "test_parts":  sorted(df_test[group_col].unique().tolist()),
    }
    return df_train, df_val, df_test, parts

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
    # TODO: Modify task label as idling = -1 -> max(Task_id)+1 for instance
    
    gaze = gaze_metrics_df.drop(columns=cols_to_drop, errors="ignore")
    mouse = mouse_metrics_df.drop(columns=cols_to_drop, errors="ignore")
    merge_keys = ["id", "participant_id", "Task_id"]
    xgboost_data = gaze.merge(mouse, on=merge_keys, how="inner", suffixes=("_gaze", "_mouse"))
    xgboost_data = xgboost_data.merge(tsfresh_data, on="id", how = "inner")

    
    # #Train/test split based on the one for JCAFNet
    # train_df = xgboost_data[xgboost_data["id"].isin(train_ids)].copy()
    # val_df = xgboost_data[xgboost_data["id"].isin(val_ids)].copy()
    # test_df = xgboost_data[xgboost_data["id"].isin(test_ids)].copy()
    
    # Redo Train/test/val split as JCAFNET not implemented yet:
    train_df, val_df, test_df, parts = split_by_participant(xgboost_data, group_col="participant_id",
                                                        test_size=0.2, val_size=0.1, random_state=42)
    
    
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
    
    y_train = train_df["Task_id"].astype(int).values + 1 # +1 to account for idle = 0  
    groups_train = train_df["participant_id"].astype(str)

    # Define Group K-Fold
    cv = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)

    # Initialize XGBoost model
    # NB: IF this doesn't work, we can do a 2-stage detection: 1) detect active VS idling, 2) detect task if active
    num_class = int(pd.Series(y_train).nunique())
    xgb_model = xgb.XGBClassifier(
        objective="multi:softmax",
        num_class=num_class,
        eval_metric="mlogloss",
        random_state=42
    )
    
    # Sampling weights to deal with class imbalance
    counts = pd.Series(y_train).value_counts()
    class_w = (counts.median() / counts).to_dict()
    sample_weight = y_train.map(class_w).values if isinstance(y_train, pd.Series) else np.array([class_w[v] for v in y_train])

    # Initialize GridSearchCV with Group K-Fold

    grid_search = GridSearchCV(
        estimator=xgb_model,
        param_grid=param_grid,
        scoring="f1_macro", # instead of accuracy so all class matter
        cv=cv.split(X_train, y_train, groups_train),  # Ensures group-based splitting
        n_jobs=-1,  # Use all available CPUs
        verbose=2
    )

    # Perform grid search
    grid_search.fit(X_train, y_train, sample_weight=sample_weight)

    # Retrieve best model and accuracy
    best_model = grid_search.best_estimator_
    best_params = grid_search.best_params_
    best_f1_mean = grid_search.best_score_
    best_row = (pd.DataFrame(grid_search.cv_results_)
            .loc[grid_search.best_index_])
    best_f1_std = float(best_row["std_test_score"])

    # Save the best model
    os.makedirs(save_model_path, exist_ok=True)
    joblib.dump(best_model, os.path.join(save_model_path, "best_model.pkl"))
    joblib.dump(grid_search, os.path.join(save_model_path, "full_grid_search.pkl"))
    joblib.dump(X_train.columns.tolist(), os.path.join(save_model_path, "features.pkl"))
    try:
        fi = getattr(best_model, "feature_importances_", None)
        if fi is not None:
            pd.DataFrame({"feature": X_train.columns, "importance": fi}) \
            .sort_values("importance", ascending=False) \
            .to_csv(os.path.join(save_model_path, "feature_importances.csv"), index=False)
    except Exception:
        pass

    # We do a manual CV loop so we can pass per-fold sample_weight and groups.
    accs, f1s = [], []
    y_true_all, y_pred_all = [], []

    for train_idx, val_idx in cv.split(X_train, y_train, groups_train):
        X_tr, X_va = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_va = y_train.iloc[train_idx], y_train.iloc[val_idx]
        w_tr = sample_weight[train_idx]

        m = clone(best_model)
        m.fit(X_tr, y_tr, sample_weight=w_tr)
        y_hat = m.predict(X_va)

        accs.append(metrics.accuracy_score(y_va, y_hat))
        f1s.append(metrics.f1_score(y_va, y_hat, average="macro"))
        y_true_all.append(y_va)
        y_pred_all.append(pd.Series(y_hat, index=y_va.index))

    y_true_all = pd.concat(y_true_all).sort_index()
    y_pred_all = pd.concat(y_pred_all).loc[y_true_all.index]

    acc_mean, acc_std = float(np.mean(accs)), float(np.std(accs))
    f1_cv_mean, f1_cv_std = float(np.mean(f1s)), float(np.std(f1s))
    
    # Reports
    cls_report_str = metrics.classification_report(y_true_all, y_pred_all, digits=3)
    cm = metrics.confusion_matrix(y_true_all, y_pred_all)
    labels_sorted = np.sort(y_train.unique())
    
    # Print nicely
    print("\n=== Cross-Validation (best model) ===")
    print(f"F1-macro (from GridSearchCV): {best_f1_mean:.3f} ± {best_f1_std:.3f}")
    print(f"F1-macro (re-evaluated):      {f1_cv_mean:.3f} ± {f1_cv_std:.3f}")
    print(f"Accuracy:                     {acc_mean:.3f} ± {acc_std:.3f}")
    print("\nPer-class report:\n", cls_report_str)
    
    # Save Artifacts
    # 1) Grid search table
    pd.DataFrame(grid_search.cv_results_).to_csv(os.path.join(save_model_path, "cv_results.csv"), index=False)
    
    # 2) params + metrics
    with open(os.path.join(save_model_path, "best_params.json"), "w") as f:
        json.dump(best_params, f, indent=2)

    metrics_payload = {
        "cv_f1_macro_mean_gs": best_f1_mean,
        "cv_f1_macro_std_gs": best_f1_std,
        "cv_f1_macro_mean_eval": f1_cv_mean,
        "cv_f1_macro_std_eval": f1_cv_std,
        "cv_accuracy_mean": acc_mean,
        "cv_accuracy_std": acc_std,
        "n_classes": int(pd.Series(y_train).nunique()),
        "class_counts": pd.Series(y_train).value_counts().to_dict(),
    }
    with open(os.path.join(save_model_path, "metrics.json"), "w") as f:
        json.dump(metrics_payload, f, indent=2)
    
    # 3) classification report & confusion matrix
    with open(os.path.join(save_model_path, "classification_report.txt"), "w") as f:
        f.write(cls_report_str)

    pd.DataFrame(cm, index=labels_sorted, columns=labels_sorted).to_csv(
        os.path.join(save_model_path, "confusion_matrix.csv")
    )
    
    # Confusion matrix
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 6))
    cm_norm = cm / cm.sum(axis=1, keepdims=True)
    im = ax.imshow(cm_norm, interpolation="nearest")
    ax.set_title("Normalized Confusion Matrix")
    ax.set_xlabel("Predicted"); ax.set_ylabel("True")
    ax.set_xticks(range(len(labels_sorted))); ax.set_xticklabels(labels_sorted, rotation=45, ha="right")
    ax.set_yticks(range(len(labels_sorted))); ax.set_yticklabels(labels_sorted)
    plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    plt.tight_layout()
    fig.savefig(os.path.join(save_model_path, "confusion_matrix.png"), dpi=150)
    plt.close(fig)                                    
                                                       