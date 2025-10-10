from typing import Union, Dict, Tuple
import os
import sys
import json
from pathlib import Path
sys.path.append(str(Path('~/git_folder/eye_tracking/').expanduser()))
# sys.path.append(str(Path('~/git/eye_tracking/').expanduser()))

import pandas as pd
import numpy as np
from utils.data_processing import EyeTrackingProcessor, GazeMetricsProcessor, MouseMetricsProcessor

from tsfresh import extract_features
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.feature_selection.relevance import calculate_relevance_table

import xgboost as xgb
import joblib  # For saving/loading models
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter(action="ignore", category=UserWarning)

# WARNING: 
# - The current verions of the XGboost training uses datasets built for the JCAFNET (i.e. resamples to fixed time steps + interpolated + bfiled)
# - The original version was not using resampled + interpolated datasets, and appeared to be slightly better
# - If we want to modify this, one can redo a dataset processing within this script, but only using the ids that are in train/test sets saved earlier.


if __name__ == "__main__":
    # ------------------------- 0. PARAMETERS -------------------------
    
    # data_path = "/scratch/eye_tracking"
    data_path = "/store/kruu/eye_tracking/"
    meta_path = "logs/jcafnet_classifier/hardy-water-3/model_metadata.json"
    save_model_path = "logs/xgboost_classifier"
    # store_data_path = str(Path('~/store/eye_tracking/splits').expanduser())
    store_data_path = str(Path('/store/kruu/eye_tracking/splits').expanduser())
    
    # TSFresh processing columns
    columns_to_extract = ["Gaze point X", "Gaze point Y", "Mouse position X", "Mouse position Y"]
    
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
    
    # Loading enriched dataset built during JCAFNET training (WARNING, has been resampled !)
    # if Path(data_path).exists():
    #     print("Loading full enriched dataset...")
    #     full_df = pd.read_parquet(data_path)
    # else:
    #     raise FileNotFoundError(f"The full enriched dataset do not exist in the specified path {data_path}.")
    
    # Loading data from scratch
    tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6']
    columns = ['Recording timestamp', 'Gaze point X', 'Gaze point Y', 'Mouse position X', 'Mouse position Y', 'Event', 'Participant name']
    fill_columns = ["Mouse position X", "Mouse position Y", "Gaze point X", "Gaze point Y"]
    files_list = os.listdir(data_path)
    files_list = [os.path.join(data_path, file) for file in files_list if file.endswith(".tsv")]
    
    processor = EyeTrackingProcessor()
    all_data = processor.load_data(files_list)
    dataset = processor.get_features(all_data, tasks, columns)
    full_df, blinks = processor.detect_blinks(dataset)
    for col in fill_columns:
        full_df[col] = full_df[col].ffill().bfill()
    
    if Path(meta_path).exists():
        print("Loading JCAFNet metadata for consistent train test split... ")
        with open(meta_path) as f:
            metadata = json.load(f)
    else:
        raise FileNotFoundError(f"Metadata for train/test split do not exist in the specified path {meta_path}.")
    
    train_ids = metadata["train_ids"]
    val_ids = metadata["val_ids"]
    test_ids = metadata["test_ids"]
    
    # Assign 'id' column if not present
    if "id" not in full_df.columns:
        full_df["id"] = (
            full_df["Participant name"].astype(str)
            + "_" + full_df["Task_id"].astype(str)
            + "_" + full_df["Task_execution"].astype(str)
        )

    # ------------------------- 2. MANUAL FEATURE EXTRACTION -------------------------

    # Group data by Task Execution
    task_groups = full_df.groupby(["Participant name", "Task_id", "Task_execution"])

    # Process all task executions
    gaze_metrics = []
    mouse_metrics = []

    for (participant, task, execution), group in task_groups:
        gaze_processor = GazeMetricsProcessor(group)
        mouse_processor = MouseMetricsProcessor(group)
        gaze_compute = gaze_processor.compute_all_metrics()
        mouse_compute = mouse_processor.compute_all_metrics()
        gaze_compute.update({"Participant": participant, "Task_id": task, "Task_execution": execution})
        mouse_compute.update({"Participant": participant, "Task_id": task, "Task_execution": execution})
        gaze_metrics.append(gaze_compute)
        mouse_metrics.append(mouse_compute)

    # Convert to DataFrame
    gaze_metrics_df = pd.DataFrame(gaze_metrics)
    gaze_metrics_df["id"] = gaze_metrics_df["Participant"].astype(str) + "_" + gaze_metrics_df["Task_id"].astype(str) + "_" + gaze_metrics_df["Task_execution"].astype(str)

    mouse_metrics_df = pd.DataFrame(mouse_metrics)
    mouse_metrics_df["id"] = mouse_metrics_df["Participant"].astype(str) + "_" + mouse_metrics_df["Task_id"].astype(str) + "_" + mouse_metrics_df["Task_execution"].astype(str)


    # ------------------------- 3. AUTOMATIC FEATURE EXTRACTION -------------------------

    data_extraction = full_df.sort_values(by=["Participant name", "Task_id", "Task_execution", "Recording timestamp"])
    
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
    
    # Dropping id that have one of the features to all zeros
    def drop_ids_with_all_zero_feature(df, id_col="id"):
        """
        Drops all rows associated with any ID group that contains at least one feature
        column with all zeros.
        """
        feature_cols = ["Gaze point X", "Gaze point Y", "Mouse position X", "Mouse position Y"]
        bad_ids = []

        for sample_id, group in df.groupby(id_col):
            if (group[feature_cols] == 0).all().any():  # If any feature is all zero in this group
                bad_ids.append(sample_id)

        if bad_ids:
            print("Dropped IDs:", bad_ids)
        else:
            print("No IDs dropped")
        return df[~df[id_col].isin(bad_ids)].copy()
    data_extraction = drop_ids_with_all_zero_feature(data_extraction, id_col="id")

    # Run tsfresh feature extraction
    print("Extracting TSFresh features from full dataset...")
    extracted_features = extract_features(data_extraction[["id", "Recording timestamp"] + columns_to_extract], 
                                        column_id="id", 
                                        column_sort="Recording timestamp", 
                                        # default_fc_parameters=MinimalFCParameters(), # Use minimal features
                                        n_jobs=100,
                                        disable_progressbar=False)

    # Impute missing values (some features may result in NaN)
    impute(extracted_features)

    # Define target variable (Task ID)
    target_variable = data_extraction.groupby("id")["Task_id"].first()

    # Select only relevant features
    relevant_features = calculate_relevance_table(extracted_features, target_variable)
    filtered_features = relevant_features[relevant_features["p_value"] < 0.05]["feature"]

    # Final dataset with selected features
    final_features = extracted_features[filtered_features].reset_index(names="id")

    # ------------------------- 4. BUILD DATASET -------------------------

    agg_feat = gaze_metrics_df.merge(mouse_metrics_df.drop(columns=["Participant", "Task_execution", "Task_id"]), on="id")
    agg_feat = agg_feat.merge(final_features, on="id")
    
    #Train/test split based on the one for JCAFNet
    train_df = agg_feat[agg_feat["id"].isin(train_ids)].copy()
    val_df = agg_feat[agg_feat["id"].isin(val_ids)].copy()
    test_df = agg_feat[agg_feat["id"].isin(test_ids)].copy()
    
    print("Saving XGboost train/test datasets")
    train_df.to_parquet(os.path.join(store_data_path, "train_xgboost.parquet"))
    test_df.to_parquet(os.path.join(store_data_path, "test_xgboost.parquet"))
    val_df.to_parquet(os.path.join(store_data_path, "val_xgboost.parquet"))

    # CROSS VALIDATION SPLIT BASED ON PARTICIPANT ID
    # Define features and target
    drop_cols = ["Task_id", "Participant", "Task_execution", "id"]
    X_train = train_df.drop(columns=drop_cols)
    y_train = train_df["Task_id"] - 1 # Target variable (-1 for xgboost)
    groups_train = train_df["Participant"]  # Grouping variable for CV
    
    # X_test = test_df.drop(columns=drop_cols)
    # y_test = test_df["Task_id"] - 1 # Target variable (-1 for xgboost)
    # groups_test = test_df["Participant"]  # Grouping variable for CV
    
    # X_val = val_df.drop(columns=drop_cols)
    # y_val = val_df["Task_id"] - 1 # Target variable (-1 for xgboost)
    # groups_val = val_df["Participant"]  # Grouping variable for CV

    # Define Group K-Fold
    gkf = GroupKFold(n_splits=n_splits)

    # Initialize XGBoost model
    xgb_model = xgb.XGBClassifier(
        objective="multi:softmax",
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
    joblib.dump(filtered_features.tolist(), os.path.join(save_model_path, "selected_features.pkl"))

    # Display results
    print(f"\nBest Parameters: {best_params}")
    print(f"Best Cross-Validation Accuracy: {best_score:.4f}")