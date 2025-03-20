import os
import sys
sys.path.append('/home/kruu/git_folder/eye_tracking/')

import pandas as pd
import numpy as np
from utils.data_processing import EyeTrackingProcessor

import xgboost as xgb
import joblib  # For saving/loading models
from sklearn.model_selection import GroupKFold, GridSearchCV
from sklearn.metrics import accuracy_score

import warnings
warnings.simplefilter(action="ignore", category=UserWarning)

def flatten_group(group):
    return np.concatenate([
        group['Gaze point X'].values,
        group['Gaze point Y'].values,
        group['Mouse position X'].values,
        group['Mouse position Y'].values,
        group['Blink'].values,
    ])

def stack_group(group):
    """
    Converts each time series group into a (n_features, seq_len) format
    instead of flattening it into a 1D vector.
    """
    return np.stack([
        group['Gaze point X'].values,
        group['Gaze point Y'].values,
        group['Mouse position X'].values,
        group['Mouse position Y'].values,
        group['Blink'].values,
    ], axis=0)

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
    
    # Fixed Time step resampling
    dataset_time_resampled = processor.resample_tasks_fixed_time(dataset, interpolate_col, timestep = 0.01)
    dataset_time_resampled.Blink = (dataset_time_resampled.Blink > 0.5) #Transform interpolated data
    dataset_time_resampled = processor.pad_tasks(dataset_time_resampled) # Padding with nans to mach longest task
    dataset_time_resampled["id"] = dataset_time_resampled["Participant name"].astype(str) + "_" + dataset_time_resampled["Task_id"].astype(str) + "_" + dataset_time_resampled["Task_execution"].astype(str)
    
    #TODO: Scale with MinMax scaler
    #TODO: Carefully handling missing values/outside screen (see word)
    #TODO: GridSearchCV based on participant with GroupKFold
    
    X = np.vstack(dataset_time_resampled.groupby(["id"]).apply(flatten_group))
    y = dataset_time_resampled.groupby(["id"])["Task_id"].min().values
    groups = dataset_time_resampled.groupby(["id"])["Participant name"].min().values