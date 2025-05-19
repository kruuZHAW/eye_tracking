from typing import Union, Dict, Tuple

import os
import sys
from pathlib import Path
import json
# sys.path.append('/cluster/home/kruu/git/eye_tracking')
sys.path.append(str(Path('~/git/eye_tracking/').expanduser()))

import pandas as pd
import numpy as np

import torch
from torch.utils.data import DataLoader
from models.jcafnet import JCAFNet
from utils.train import train_classifier, split_by_participant
from utils.dataset import GazeMouseDatasetJCAFNet

from utils.data_processing import EyeTrackingProcessor, GazeMetricsProcessor, MouseMetricsProcessor

from tqdm import tqdm

import warnings
warnings.simplefilter(action="ignore", category=UserWarning)

def load_and_process(data_path: Union[str, Path],
                     tasks: list[str],
                     columns: list[str],
                     interpolate_cols: list[str],
                     fill_cols: list[str]
                     ) -> pd.DataFrame:
    
    files_list = os.listdir(data_path)
    files_list = [os.path.join(data_path, file) for file in files_list if file.endswith(".tsv")]
    
    processor = EyeTrackingProcessor()
    all_data = processor.load_data(files_list)
    dataset = processor.get_features(all_data, tasks, columns)
    dataset, blinks = processor.detect_blinks(dataset)
    
    # Fixed Time step resampling
    dataset_time_resampled = processor.resample_tasks_fixed_time(dataset, interpolate_cols, timestep = 0.01)
    dataset_time_resampled.Blink = (dataset_time_resampled.Blink > 0.5) #Transform interpolated data
    dataset_time_resampled["id"] = dataset_time_resampled["Participant name"].astype(str) + "_" + dataset_time_resampled["Task_id"].astype(str) + "_" + dataset_time_resampled["Task_execution"].astype(str)

    for col in fill_cols:
        dataset_time_resampled[col] = dataset_time_resampled[col].ffill().bfill()
        
    return dataset_time_resampled

def compute_joint_features(df:pd.DataFrame):
    gx, gy = df["Gaze point X"].fillna(0), df["Gaze point Y"].fillna(0)
    mx, my = df["Mouse position X"].fillna(0), df["Mouse position Y"].fillna(0)

    distance = np.sqrt((gx - mx)**2 + (gy - my)**2)
    angle = np.arctan2(gy - my, gx - mx)  # radians
    return distance, angle

#Enrich dataframe with features for JCAFNet
def enrich_with_gaze_mouse_metrics(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["Gaze Velocity"] = np.nan
    df["Gaze Acceleration"] = np.nan
    df["Mouse Velocity"] = np.nan
    df["Mouse Acceleration"] = np.nan
    df["Gaze-Mouse Distance"] = np.nan
    df["Angle Between Gaze and Mouse"] = np.nan

    task_group = df.groupby("id")

    for id, group in tqdm(task_group, desc="Enriching features"):
        mask = df["id"] == id

        # Gaze metrics
        gaze_proc = GazeMetricsProcessor(group)
        gaze_vel, gaze_acc = gaze_proc.compute_velocity_acceleration()
        df.loc[mask, "Gaze Velocity"] = gaze_vel.fillna(0).values
        df.loc[mask, "Gaze Acceleration"] = gaze_acc.fillna(0).values

        # Mouse metrics
        mouse_proc = MouseMetricsProcessor(group)
        mouse_vel, mouse_acc = mouse_proc.compute_velocity_acceleration()
        df.loc[mask, "Mouse Velocity"] = mouse_vel.fillna(0).values
        df.loc[mask, "Mouse Acceleration"] = mouse_acc.fillna(0).values

        # Joint features
        dist, angle = compute_joint_features(group)
        df.loc[mask, "Gaze-Mouse Distance"] = dist.fillna(0).values
        df.loc[mask, "Angle Between Gaze and Mouse"] = angle.fillna(0).values

    return df

if __name__ == "__main__":
    
    # ------------------------- 0. PARAMETERS -------------------------
    store_path = str(Path('~/store/eye_tracking').expanduser())
    data_path = "/scratch/eye_tracking"
    processed_path = "/scratch/eye_tracking/processed_enriched.parquet"
    splits_dir = "/scratch/eye_tracking/splits"

    tasks = ['Task 1', 'Task 2', 'Task 3', 'Task 4', 'Task 5', 'Task 6']
    cols = ['Recording timestamp', 'Gaze point X', 'Gaze point Y', 'Mouse position X', 'Mouse position Y', 'Event', 'Participant name']
    interpolate_cols = ['Recording timestamp', 'Gaze point X', 'Gaze point Y', 'Mouse position X', 'Mouse position Y', 'Blink']
    fill_cols = ["Mouse position X", "Mouse position Y", "Gaze point X", "Gaze point Y"]
    
    features = {
        "gaze": ["Gaze point X", "Gaze point Y", "Gaze Velocity", "Gaze Acceleration"],
        "mouse": ["Mouse position X", "Mouse position Y", "Mouse Velocity", "Mouse Acceleration"],
        "joint": ["Gaze-Mouse Distance", "Angle Between Gaze and Mouse"]
    }
    
    gaze_dim = len(features["gaze"])
    mouse_dim = len(features["mouse"])
    joint_dim = len(features["joint"])
    num_classes = 6
    batch_size = 32
    lr = 0.001
    num_epochs = 100
    data_augment = True
    
    # ------------------------- 1. LOAD OR PROCESSING -------------------------
    
    if Path(processed_path).exists():
        print(f"Loading cached enriched dataset from {processed_path}")
        dataset_enriched = pd.read_parquet(processed_path)
    else:
        print("No cached dataset found — processing from raw files...")
        dataset_time_resampled = load_and_process(
            data_path=data_path,
            tasks=tasks,
            columns=cols,
            interpolate_cols=interpolate_cols,
            fill_cols=fill_cols
        )
        dataset_enriched = enrich_with_gaze_mouse_metrics(dataset_time_resampled)
        dataset_enriched.to_parquet(os.path.join(store_path, "processed_enriched.parquet"))
        dataset_enriched.to_parquet(processed_path)
        print(f"Processed dataset saved to {os.path.join(store_path, "processed_enriched.parquet")} and {processed_path}")
        
    # ------------------------- 2. SPLIT (or load) -------------------------
    
    split_files = [f"{splits_dir}/{s}.parquet" for s in ["train", "val", "test"]]
    
    if all(Path(f).exists() for f in split_files):
        print("Loading dataset splits...")
        train_df = pd.read_parquet(split_files[0])
        val_df = pd.read_parquet(split_files[1])
        test_df = pd.read_parquet(split_files[2])
    else:
        print("Creating and saving new splits...")
        
        store_splits_dir = os.path.join(store_path, "splits") # create directory in home/store
        store_split_files = [f"{store_splits_dir}/{s}.parquet" for s in ["train", "val", "test"]]
        
        train_df, val_df, test_df = split_by_participant(dataset_enriched, val_split=0.2, test_split=0.1)
        os.makedirs(store_splits_dir, exist_ok=True)
        os.makedirs(splits_dir, exist_ok=True)
        train_df.to_parquet(split_files[0])
        val_df.to_parquet(split_files[1])
        test_df.to_parquet(split_files[2])
        train_df.to_parquet(store_split_files[0])
        val_df.to_parquet(store_split_files[1])
        test_df.to_parquet(store_split_files[2])
        print(f"✅ Splits saved to {splits_dir} and {store_splits_dir}")
    
    # ------------------------- 3. TRAINING JCAFNET -------------------------

    model = JCAFNet(num_classes, gaze_dim, mouse_dim, joint_dim, lr)

    model_trained, mean, std, best_ckpt_path = train_classifier(model,
                                        train_df,
                                        val_df,
                                        features,
                                        checkpoint_base_dir = "logs/jcafnet_classifier",
                                        num_epochs=num_epochs,
                                        data_augment=data_augment,
                                        use_wandb=True)
    
    export_dir = Path(best_ckpt_path).parent
    metadata = {
        "features": features,
        "mean": {k: float(v) for k, v in mean.to_dict().items()},
        "std": {k: float(v) for k, v in std.to_dict().items()},
        "train_ids": train_df["id"].unique().tolist(),
        "val_ids": val_df["id"].unique().tolist(),
        "test_ids": test_df["id"].unique().tolist(),
        "num_classes": num_classes,
        "batch_size": batch_size,
        "data_augmentation": data_augment,
    }
    
    # Save to JSON
    metadata_path = os.path.join(export_dir,"model_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Metadata saved to {metadata_path}")
    
    # Access best and last validation accuracy
    best_val_acc = model_trained.trainer.checkpoint_callback.best_model_score

    print(f"Best Val Accuracy: {best_val_acc:.4f}")