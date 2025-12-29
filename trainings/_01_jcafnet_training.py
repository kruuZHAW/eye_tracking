import os
import sys
from pathlib import Path
import json
# sys.path.append('/cluster/home/kruu/git/eye_tracking')
sys.path.append(str(Path('~/git/eye_tracking/').expanduser()))

import pandas as pd
import numpy as np

from utils.data_processing_gaze_data import GazeMetricsProcessor, MouseMetricsProcessor
from utils.helper import load_and_process, load_processed_data, save_processed_data
from utils.train import train_classifier, split_chunks_by_participant
from models.jcafnet import JCAFNet


from tqdm import tqdm

import warnings
warnings.simplefilter(action="ignore", category=UserWarning)

 # ------------------------- HELPER FUNCTIONS -------------------------

def compute_joint_features(df:pd.DataFrame):
    gaze_x = next((col for col in df.columns if "Gaze point X" in col), None)
    gaze_y = next((col for col in df.columns if "Gaze point Y" in col), None)
    mouse_x = next((col for col in df.columns if "Mouse position X" in col), None)
    mouse_y = next((col for col in df.columns if "Mouse position Y" in col), None)
    
    gx, gy = df[gaze_x].fillna(0), df[gaze_y].fillna(0)
    mx, my = df[mouse_x].fillna(0), df[mouse_y].fillna(0)

    distance = np.sqrt((gx - mx)**2 + (gy - my)**2)
    angle = np.arctan2(gy - my, gx - mx)  # radians
    return distance, angle

def enrich_with_gaze_mouse_metrics(task_chunks: dict[str, pd.DataFrame]) -> dict[str, pd.DataFrame]:
    enriched_chunks = {}

    for task_id, df in tqdm(task_chunks.items(), desc="Enriching features"):
        df = df.copy()

        # Initialize metrics columns
        df["Gaze Velocity"] = np.nan
        df["Gaze Acceleration"] = np.nan
        df["Mouse Velocity"] = np.nan
        df["Mouse Acceleration"] = np.nan
        df["Gaze-Mouse Distance"] = np.nan
        df["Angle Between Gaze and Mouse"] = np.nan

        # Gaze metrics
        gaze_proc = GazeMetricsProcessor(df)
        gaze_vel, gaze_acc = gaze_proc.compute_velocity_acceleration()
        df["Gaze Velocity"] = gaze_vel.fillna(0).values
        df["Gaze Acceleration"] = gaze_acc.fillna(0).values

        # Mouse metrics
        mouse_proc = MouseMetricsProcessor(df)
        mouse_vel, mouse_acc = mouse_proc.compute_velocity_acceleration()
        df["Mouse Velocity"] = mouse_vel.fillna(0).values
        df["Mouse Acceleration"] = mouse_acc.fillna(0).values

        # Joint gaze-mouse features
        dist, angle = compute_joint_features(df)
        df["Gaze-Mouse Distance"] = dist.fillna(0).values
        df["Angle Between Gaze and Mouse"] = angle.fillna(0).values

        enriched_chunks[task_id] = df

    return enriched_chunks

if __name__ == "__main__":
    
    # ------------------------- 0. PARAMETERS -------------------------
    # Long term storage
    store_dir = str(Path('~/store/aware').expanduser())
    store_raw_inputs_dir = os.path.join(store_dir, "training_data_raw_inputs")
    store_processed_dir = os.path.join(store_dir, "processed_inputs")
    store_splits_dir = os.path.join(store_dir, "splits")
    split_names = ["train", "val", "test"]
    store_split_dirs = [os.path.join(store_splits_dir, split_name) for split_name in split_names]
    
    # Temporary storage for better I/O performance
    temp_data_dir = "/scratch/aware"
    temp_raw_inputs_dir = os.path.join(temp_data_dir, "training_data_raw_inputs")
    temp_processed_dir = os.path.join(temp_data_dir, "processed_enriched_inputs")
    temp_splits_dir = os.path.join(temp_data_dir, "splits")
    temp_split_dirs = [os.path.join(temp_splits_dir, split_name) for split_name in split_names]
    
    
    features = ['Recording timestamp [ms]', 'Gaze point X [DACS px]', 'Gaze point Y [DACS px]', 'Mouse position X', 'Mouse position Y', 'Event']
    interpolate_cols = ['Gaze point X [DACS px]', 'Gaze point Y [DACS px]', 'Mouse position X', 'Mouse position Y', "Blink"]
    fill_columns = ['Gaze point X [DACS px]', 'Gaze point Y [DACS px]', 'Mouse position X', 'Mouse position Y']

    features_group = {
        "gaze": ['Gaze point X [DACS px]', 'Gaze point Y [DACS px]', "Gaze Velocity", "Gaze Acceleration"],
        "mouse": ['Mouse position X', 'Mouse position Y', "Mouse Velocity", "Mouse Acceleration"],
        "joint": ["Gaze-Mouse Distance", "Angle Between Gaze and Mouse"]
    }
    
    gaze_dim = len(features_group["gaze"])
    mouse_dim = len(features_group["mouse"])
    joint_dim = len(features_group["joint"])
    batch_size = 32
    lr = 0.001
    num_epochs = 100
    data_augment = True
    
    # ------------------------- 1. LOAD OR PROCESSING -------------------------
    
    if Path(temp_processed_dir).exists():
        print(f"Loading cached enriched dataset from {temp_processed_dir}")
        enriched_chunks, blinks, atco_task_map = load_processed_data(temp_processed_dir)
    else:
        print("No cached dataset found — processing from raw files...")
        chunks_jcafnet, blinks, atco_task_map  = load_and_process(root_dir=temp_raw_inputs_dir, 
                                                                  columns=features, 
                                                                  interpolate_cols=interpolate_cols, 
                                                                  fill_cols=fill_columns, 
                                                                  time_resampling=True, 
                                                                  fixed_window_ms=10000, # Size of the chunk window. None if chunk per task
                                                                  window_step_ms=2000, # Time step from one window to another. None is no overlap
                                                                  min_task_presence=0.5 # Min proportion of task presence for assigning a label
                                                                  )
        
        # chunks_jcafnet, blinks, atco_task_map  = load_and_process(root_dir=temp_raw_inputs_dir, 
        #                                                           columns=features, 
        #                                                           interpolate_cols=interpolate_cols, 
        #                                                           fill_cols=fill_columns, 
        #                                                           time_resampling=False, 
        #                                                           fixed_window_ms=None, # Size of the chunk window. None if chunk per task
        #                                                           )
        
        
        enriched_chunks = enrich_with_gaze_mouse_metrics(chunks_jcafnet)
        save_processed_data(temp_processed_dir, enriched_chunks, blinks, atco_task_map)
        save_processed_data(store_processed_dir, enriched_chunks, blinks, atco_task_map)
        print(f"Processed input datasets saved to {temp_processed_dir} and {store_processed_dir}")
    
    num_classes = len(atco_task_map.values())
        
    # ------------------------- 2. SPLIT (or load) -------------------------
    
    if all(Path(f).exists() for f in temp_split_dirs):
        print(f"Loading dataset splits from {temp_splits_dir}...")
        train_chunks, _, _ = load_processed_data(temp_split_dirs[0])
        val_chunks, _, _   = load_processed_data(temp_split_dirs[1])
        test_chunks, _, _  = load_processed_data(temp_split_dirs[2])

    else:
        print(f"Creating and saving new train/val/test splits in {temp_splits_dir} and {store_splits_dir}...")
        
        train_chunks, val_chunks, test_chunks = split_chunks_by_participant(enriched_chunks)
        
        # Save each split chunk in the long term storage
        save_processed_data(store_split_dirs[0], train_chunks) 
        save_processed_data(store_split_dirs[1], val_chunks)
        save_processed_data(store_split_dirs[2], test_chunks)
        
        # Save each split chunk in the temp storage
        save_processed_data(temp_split_dirs[0], train_chunks) 
        save_processed_data(temp_split_dirs[1], val_chunks)
        save_processed_data(temp_split_dirs[2], test_chunks)
    
        print(f"Splits saved to {temp_splits_dir} and {store_splits_dir}")
    
    print(f"Train participants: {set(key.split('_')[0] for key in train_chunks.keys())}")
    print(f"Val participants: {set(key.split('_')[0] for key in val_chunks.keys())}")
    print(f"Test participants: {set(key.split('_')[0] for key in test_chunks.keys())}")
    
    # ------------------------- 3. TRAINING JCAFNET -------------------------
    # TODO: Modify task label as idling = -1 -> max(Task_id)+1 for instance

    model = JCAFNet(num_classes, gaze_dim, mouse_dim, joint_dim, lr)

    model_trained, mean, std, best_ckpt_path = train_classifier(model,
                                        train_chunks,
                                        val_chunks,
                                        features_group,
                                        checkpoint_base_dir = "logs/jcafnet_classifier",
                                        num_epochs=num_epochs,
                                        data_augment=data_augment,
                                        use_wandb=True)
    
    export_dir = Path(best_ckpt_path).parent
    metadata = {
        "features": features,
        "mean": {k: float(v) for k, v in mean.to_dict().items()},
        "std": {k: float(v) for k, v in std.to_dict().items()},
        "train_ids": [key for key, _ in train_chunks.items()],
        "val_ids": [key for key, _ in val_chunks.items()],
        "test_ids": [key for key, _ in test_chunks.items()],
        "num_classes": num_classes,
        "batch_size": batch_size,
        "data_augmentation": data_augment,
    }
    
    # Save to JSON
    metadata_path = os.path.join(export_dir,"jcafnet_metadata.json")
    with open(metadata_path, "w") as f:
        json.dump(metadata, f, indent=4)

    print(f"✅ Metadata saved to {metadata_path}")
    
    # Access best and last validation accuracy
    best_val_acc = model_trained.trainer.checkpoint_callback.best_model_score

    print(f"Best Val Accuracy: {best_val_acc:.4f}")