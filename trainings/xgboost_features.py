import os
import sys
sys.path.append('/home/kruu/git_folder/eye_tracking/')

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


if __name__ == "__main__":
  # ------------------------- 1. DATA LOADING & PROCESSING -------------------------

  data_path = "/store/kruu/eye_tracking"
  files_list = os.listdir(data_path)
  files_list = [os.path.join(data_path, file) for file in files_list]

  tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6']
  features = ['Recording timestamp', 'Gaze point X', 'Gaze point Y', 'Mouse position X', 'Mouse position Y', 'Event', 'Participant name']
  interpolate_col = ['Recording timestamp', 'Gaze point X', 'Gaze point Y', 'Mouse position X', 'Mouse position Y', 'Blink']

  processor = EyeTrackingProcessor()
  all_data = processor.load_data(files_list)
  dataset = processor.get_features(all_data, tasks, features)
  dataset, blinks = processor.detect_blinks(dataset)


  # ------------------------- 2. MANUAL FEATURE EXTRACTION -------------------------

  # Group data by Task Execution
  task_groups = dataset.groupby(["Participant name", "Task_id", "Task_execution"])

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

  data_extraction = dataset.sort_values(by=["Participant name", "Task_id", "Task_execution", "Recording timestamp"])

  # Fill missing values (important for tsfresh)
  data_extraction["Mouse position X"] = data_extraction["Mouse position X"].ffill().bfill()
  data_extraction["Mouse position Y"] = data_extraction["Mouse position Y"].ffill().bfill()
  data_extraction["Gaze point X"] = data_extraction["Gaze point X"].ffill().bfill()
  data_extraction["Gaze point Y"] = data_extraction["Gaze point Y"].ffill().bfill()

  # Define ID column (tsfresh needs a unique identifier for each time-series)
  data_extraction["id"] = data_extraction["Participant name"].astype(str) + "_" + data_extraction["Task_id"].astype(str) + "_" + data_extraction["Task_execution"].astype(str)


  columns_to_extract = ["Gaze point X", "Gaze point Y", "Mouse position X", "Mouse position Y"]

  # Run tsfresh feature extraction
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

  # CROSS VALIDATION SPLIT BASED ON PARTICIPANT ID
  # Define features and target
  X = agg_feat.drop(columns=["Task_id", "Participant", "Task_execution", "id"])  # Features
  y = agg_feat["Task_id"] - 1  # Target variable (adjusted indexing for XGBoost)
  groups = agg_feat["Participant"]  # Grouping variable for CV

  # Define Group K-Fold
  n_splits = 5
  gkf = GroupKFold(n_splits=n_splits)

  # Define hyperparameter grid
  param_grid = {
      "n_estimators": [100, 200],          # Number of trees
      "max_depth": [4, 6, 8],              # Tree depth
      "learning_rate": [0.01, 0.1, 0.3],   # Learning rate
      "subsample": [0.8, 1.0],             # Subsampling ratio
      "colsample_bytree": [0.8, 1.0],      # Feature sampling ratio per tree
      "gamma": [0, 0.1, 0.3],              # Minimum loss reduction
  }

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
  grid_search.fit(X, y, groups=groups)

  # Retrieve best model and accuracy
  best_model = grid_search.best_estimator_
  best_params = grid_search.best_params_
  best_score = grid_search.best_score_

  # Save the best model
  joblib.dump(best_model, "logs/xgboost_features/best_model.pkl")
  joblib.dump(grid_search, "logs/xgboost_features/full_grid_search.pkl")

  # Display results
  print(f"\nBest Parameters: {best_params}")
  print(f"Best Cross-Validation Accuracy: {best_score:.4f}")