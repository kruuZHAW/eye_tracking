# Ensemble inference
# - Loads XGBoost and JCAFNet trained models
# - Processes input data for both models
# - Runs inference with each
# - Combines class probabilities via soft voting
# - Ouputs predictions and confidence scores

import os
import json
import joblib
import torch
import pandas as pd
import numpy as np
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from models.jcafnet import JCAFNet
from utils.dataset import GazeMouseDatasetJCAFNet
from utils.train import collate_jcafnet
from torch.utils.data import DataLoader
from utils.data_processing import EyeTrackingProcessor, GazeMetricsProcessor, MouseMetricsProcessor

from tqdm import tqdm

# ------------------------- LOADERS -------------------------

def load_xgboost_model(path):
    return joblib.load(path)

def load_jcafnet_model(ckpt_path, metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)

    model = JCAFNet(
        num_classes=metadata["num_classes"],
        gaze_dim=len(metadata["features"]["gaze"]),
        mouse_dim=len(metadata["features"]["mouse"]),
        joint_dim=len(metadata["features"]["joint"]),
        learning_rate=0.001
    )
    state_dict = torch.load(ckpt_path)["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata

# ------------------------- INFERENCE -------------------------

def run_ensemble_inference(xgb_model, jcafnet_model, metadata, test_df):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    jcafnet_model = jcafnet_model.to(device)

    # Prepare data for JCAFNet
    mean = pd.Series(metadata["mean"])
    std = pd.Series(metadata["std"])
    features = metadata["features"]

    dataset_jcaf = GazeMouseDatasetJCAFNet(
        test_df, 
        gaze_features=features["gaze"], 
        mouse_features=features["mouse"], 
        joint_features=features["joint"], 
        augment=False,
        mean=mean,
        std=std
    )
    loader = DataLoader(dataset_jcaf, batch_size=32, shuffle=False, collate_fn=collate_jcafnet)

    # Prepare data for XGBoost
    xgb_features_df = test_df.drop_duplicates("id")  # one row per sequence
    xgb_ids = xgb_features_df["id"]
    X_xgb = xgb_features_df[xgb_model.get_booster().feature_names]

    # Inference
    all_preds, all_probs = [], []
    xgb_probs = xgb_model.predict_proba(X_xgb)

    with torch.no_grad():
        for batch in loader:
            gaze = batch["gaze"].to(device)
            mouse = batch["mouse"].to(device)
            joint = batch["joint"].to(device)

            logits = jcafnet_model(gaze, mouse, joint)
            probs_jcaf = torch.softmax(logits, dim=1).cpu().numpy()

            batch_probs_xgb = xgb_probs[:len(probs_jcaf)]
            xgb_probs = xgb_probs[len(probs_jcaf):]  # Remove used entries

            ensemble_probs = (probs_jcaf + batch_probs_xgb) / 2
            preds = np.argmax(ensemble_probs, axis=1)

            all_preds.extend(preds)
            all_probs.extend(ensemble_probs)

    return all_preds, np.array(all_probs), xgb_ids.tolist()
