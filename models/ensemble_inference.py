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
from utils.data_processing_gaze_data import EyeTrackingProcessor, GazeMetricsProcessor, MouseMetricsProcessor

from tqdm import tqdm
import joblib

from sklearn.metrics import accuracy_score

# ------------------------- LOADERS -------------------------

def load_xgboost_model(path):
    return joblib.load(path)

def load_jcafnet_model(ckpt_path, metadata_path):
    with open(metadata_path, "r") as f:
        metadata = json.load(f)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    model = JCAFNet(
        num_classes=metadata["num_classes"],
        gaze_dim=len(metadata["features"]["gaze"]),
        mouse_dim=len(metadata["features"]["mouse"]),
        joint_dim=len(metadata["features"]["joint"]),
        learning_rate=0.001
    )
    state_dict = torch.load(ckpt_path, map_location=device)["state_dict"]
    model.load_state_dict(state_dict)
    model.eval()
    return model, metadata

# ------------------------- XGBOOST PREDICTION -------------------------

def predict_xgboost(test_df: pd.DataFrame,
                             model_path: str,
                             selected_features_path: str,
                             label_column: str = "Task_id",
                             label_offset: int = 1) -> pd.DataFrame:
    """
    Suppose that the TSfresh features on the test set were already extracted.
    Returns:
        pd.DataFrame with columns: id, true_label, pred_label, and class probabilities.
    """

    # Load model
    model = joblib.load(model_path)
    
    # Keep only selected features
    ids = test_df["id"].tolist()
    y = test_df[label_column] - label_offset
    drop_cols = ["Task_id", "participant_id", "id"]
    X = test_df.drop(columns=drop_cols)

    # Predict probabilities and class
    probs = model.predict_proba(X)
    preds = model.predict(X)
    
    acc = accuracy_score(y, preds)
    print(f"XGBoost Accuracy: {acc:.4f}")

    # Construct result DataFrame
    result_df = pd.DataFrame({
        "id": ids,
        "true_label": y.values,
        "pred_label": preds
    })
    
    # Add probability columns
    for i in range(probs.shape[1]):
        result_df[f"class_{i}_prob"] = probs[:, i]

    return result_df

# ------------------------- INFERENCE -------------------------

def soft_voting_ensemble(
    probs_model1: np.ndarray,
    probs_model2: np.ndarray,
    true_labels: np.ndarray,
    weight_model1: float = 0.5,
    weight_model2: float = 0.5,
    ids: list = None
) -> pd.DataFrame:
    """
    Perform soft voting ensemble using weighted average of probabilities.

    Args:
        probs_model1 (np.ndarray): Class probabilities from model 1. Shape (n_samples, n_classes).
        probs_model2 (np.ndarray): Class probabilities from model 2. Shape (n_samples, n_classes).
        true_labels (np.ndarray): Ground truth labels. Shape (n_samples,).
        weight_model1 (float): Weight for model 1 predictions.
        weight_model2 (float): Weight for model 2 predictions.
        ids (list, optional): Sample IDs corresponding to predictions.

    Returns:
        pd.DataFrame: Result dataframe with columns: id, true_label, pred_label, and class probabilities.
    """
    assert probs_model1.shape == probs_model2.shape, "Probability shapes must match"
    assert np.isclose(weight_model1 + weight_model2, 1.0), "Weights must sum to 1"

    # Weighted average of probabilities
    ensemble_probs = weight_model1 * probs_model1 + weight_model2 * probs_model2
    ensemble_preds = ensemble_probs.argmax(axis=1)

    # Build results
    result = {
        "true_label": true_labels,
        "pred_label": ensemble_preds
    }
    for i in range(ensemble_probs.shape[1]):
        result[f"class_{i}_prob"] = ensemble_probs[:, i]

    if ids is not None:
        result["id"] = ids

    df_result = pd.DataFrame(result)
    
    # Move 'id' to front if it exists
    if "id" in df_result.columns:
        cols = ["id", "true_label", "pred_label"] + [c for c in df_result.columns if c.startswith("class_")]
        df_result = df_result[cols]

    # Print overall accuracy
    acc = accuracy_score(true_labels, ensemble_preds)
    print(f"Soft Voting Ensemble Accuracy: {acc:.4f}")

    return df_result