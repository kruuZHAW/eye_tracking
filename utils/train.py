import os
from pathlib import Path
from typing import Union
from datetime import datetime

import numpy as np
import pandas as pd

import torch
import torch.nn.functional as F
from torch.nn.utils.rnn import pad_sequence
from torchmetrics.classification import Accuracy

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from models.lstm_classifier import LSTMClassifier
from models.jcafnet import JCAFNet

from sklearn.model_selection import train_test_split

from utils.dataset import GazeMouseDatasetLSTM, GazeMouseDatasetJCAFNet
from torch.utils.data import DataLoader, random_split

import onnxruntime as ort

# ------------------------- SPLIT BY PARTICIPANT -------------------------

def split_by_participant(dataset, val_split=0.2, test_split=0.1, random_state=42):
    participants = dataset["Participant name"].unique()
    
    # Split into train and temp (val + test)
    train_participants, temp_participants = train_test_split(
        participants, test_size=(val_split + test_split), random_state=random_state
    )
    
    # Further split temp into val and test
    relative_val_split = val_split / (val_split + test_split)
    val_participants, test_participants = train_test_split(
        temp_participants, test_size=(1 - relative_val_split), random_state=random_state
    )
    
    # Filter dataset
    train_df = dataset[dataset["Participant name"].isin(train_participants)].copy()
    val_df = dataset[dataset["Participant name"].isin(val_participants)].copy()
    test_df = dataset[dataset["Participant name"].isin(test_participants)].copy()

    return train_df, val_df, test_df

# ------------------------- PADDING FOR JCAFNET -------------------------

def collate_jcafnet(batch):
    
    def pad_and_stack(seqs):
        # seqs: list of [C, T] tensors
        # Goal: pad each along time dim to max T, then stack
        max_len = max(seq.shape[1] for seq in seqs)
        padded = [F.pad(seq, (0, max_len - seq.shape[1])) for seq in seqs]  # pad last dim
        return torch.stack(padded)

    gaze = pad_and_stack([b['gaze'] for b in batch]).unsqueeze(2) # [B, C, 1, T] (2D input for ResNet)
    mouse = pad_and_stack([b['mouse'] for b in batch]).unsqueeze(2) 
    joint = pad_and_stack([b['joint'] for b in batch]) # [B, C, T] (Conv1D for joint input)
    task_id = torch.stack([b['task_id'] for b in batch])

    return {
        "gaze": gaze,     # [B, C, 1, T_max]
        "mouse": mouse,
        "joint": joint,
        "task_id": task_id,
    }

# ------------------------- TRAINING FUNCTION -------------------------
def train_classifier(model,
                     train_dict,
                     val_dict, 
                     features, 
                     checkpoint_base_dir: Union[str, Path],
                     batch_size=32, 
                     num_epochs=20,
                     data_augment = True,
                     use_wandb = True):
    """
    Trains the deep learning classifier (LSTMClassifier or JCAFNet).

    Args:
        model: Model to train
        train_dict (dict[str, pd.DataFrame]): Training set.
        val_dict (dict[str, pd.DataFrame]): Validation set.
        features (list): List of input feature column names (for LSTM) or Dict of input features (for JCAFNet)
        batch_size (int): Batch size for training.
        hidden_dim (int): Number of hidden units in LSTM.
        num_layers (int): Number of LSTM layers.
        learning_rate (float): Learning rate for optimizer.
        num_epochs (int): Number of epochs for training.
        val_split (float): Percentage of data for validation.
        test_split (float): Percentage of data for testing.
        use_wandb (bool): Whether to log training in Weights & Biases.

    Returns:
        model (LSTMClassifier or JCAFNet): Trained PyTorch Lightning model.
    """
    
    # DEPRECIATED
    # Determine dataset mode from model class
    if isinstance(model, LSTMClassifier):
        if not isinstance(features, list):
            raise ValueError("For LSTMClassifier, 'features' must be a flat list.")
        feature_list = features
        train_set = GazeMouseDatasetLSTM(train_dict, feature_list, augment=data_augment)
        mean, std = train_set.mean, train_set.std
        val_set = GazeMouseDatasetLSTM(val_dict, features, augment=False, mean = mean, std = std)
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
        
    elif isinstance(model, JCAFNet):
        if not isinstance(features, dict) or not all(k in features for k in ["gaze", "mouse", "joint"]):
            raise ValueError("For JCAFNet, 'features' must be a dict with keys: 'gaze', 'mouse', 'joint'")
        feature_list = features["gaze"] + features["mouse"] + features["joint"]
        train_set = GazeMouseDatasetJCAFNet(
            dataset_dict=train_dict,
            gaze_features=features["gaze"],
            mouse_features=features["mouse"],
            joint_features=features["joint"],
            augment=False, 
            mean = None, 
            std = None,
        )
        mean, std = train_set.mean, train_set.std
        val_set = GazeMouseDatasetJCAFNet(
            dataset_dict=val_dict,
            gaze_features=features["gaze"],
            mouse_features=features["mouse"],
            joint_features=features["joint"],
            augment=False, 
            mean = mean, 
            std = std,
        )
        train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True, collate_fn=collate_jcafnet)
        val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False, collate_fn=collate_jcafnet)
        
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")

    # Debug DataLoader
    # batch = next(iter(train_loader))
    # print("gaze:", batch["gaze"].shape, batch["gaze"].min(), batch["gaze"].max())
    # print("mouse:", batch["mouse"].shape)
    # print("joint:", batch["joint"].shape)
    # print("task_id:", batch["task_id"])
    # print(f"Gaze nan and inf: {torch.isnan(batch["gaze"]).any()}, {torch.isinf(batch["gaze"]).any()}")
    # print(f"Mouse nan and inf: {torch.isnan(batch["mouse"]).any()}, {torch.isinf(batch["mouse"]).any()}")
    # print(f"Joint nan and inf: {torch.isnan(batch["joint"]).any()}, {torch.isinf(batch["joint"]).any()}")
    # return
    
    checkpoint_dir = Path(checkpoint_base_dir)
    
    # Initialize WandB logger
    if use_wandb:
        wandb_logger = WandbLogger(project="GazeMouse_Classification", log_model="checkpoint")
        wandb_run = wandb_logger.experiment  # this is the actual wandb.Run object
        run_name = wandb_run.name or wandb_run.id  # fallback to ID if name not set
        checkpoint_dir = checkpoint_dir / run_name
    else:
        wandb_logger = None
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        checkpoint_dir = checkpoint_dir / f"run_{timestamp}"
        
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
    #Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        dirpath=checkpoint_dir,
        filename="epoch{epoch:02d}-val_acc{val_acc:.2f}",
        auto_insert_metric_name=False
    )
    
    checkpoint_callback_last = ModelCheckpoint(
        dirpath=checkpoint_dir,
        save_top_k=1,
        monitor=None,
        filename="last-epoch{epoch:02d}",
        every_n_epochs=1,
        save_last=True
    )

    # Define trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[
            # EarlyStopping(monitor="val_loss", patience=5, mode="min"),  # Stop if val_loss doesn't improve for 5 epochs
            checkpoint_callback,
            checkpoint_callback_last,
        ],
        gradient_clip_val=1.0,
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)
    return model, mean, std, checkpoint_callback.best_model_path

# ------------------------- EVALUATE PYTORCH CPTK -------------------------

def evaluate_pytorch_model(
    model,
    dict_df,
    features, # For LSTM: list; For JCAFNet: dict with keys ["gaze", "mouse", "joint"]
    num_classes,
    mean,
    std,
    batch_size=32,
):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # --- Select dataset & input unpacking ---
    # DEPRECIATED
    if isinstance(model, LSTMClassifier):
        dataset = GazeMouseDatasetLSTM(dict_df, features, augment=False, mean=mean, std=std)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)
        
    elif isinstance(model, JCAFNet):
        dataset = GazeMouseDatasetJCAFNet(
            dict_df,
            gaze_features=features["gaze"],
            mouse_features=features["mouse"],
            joint_features=features["joint"],
            augment=False,
            mean=mean,
            std=std
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_jcafnet)
    else:
        raise ValueError(f"Unsupported model type: {type(model)}")
        
    # --- Evaluation loop ---
    accuracy_metric = Accuracy(task="multiclass", num_classes=num_classes).to(device)

    all_ids, all_preds, all_labels, all_probs = [], [], [], []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch_idx, batch in enumerate(loader):
            if isinstance(model, LSTMClassifier):
                x = batch["sequence"].to(device)
                lengths = batch["seq_length"].to(device)
                labels = batch["task_id"].to(device)
                logits = model(x, lengths)
            else:  # JCAFNet
                gaze = batch["gaze"].to(device)
                mouse = batch["mouse"].to(device)
                joint = batch["joint"].to(device)
                labels = batch["task_id"].to(device)
                logits = model(gaze, mouse, joint)
            
            if isinstance(model, LSTMClassifier):
                batch_size_actual = batch["sequence"].size(0)
            else:
                batch_size_actual = batch["gaze"].size(0)

            ids = [dataset.get_sample_id(i) for i in range(batch_idx * batch_size, batch_idx * batch_size + batch_size_actual)]
            probs = torch.softmax(logits, dim=1)
            preds = torch.argmax(probs, dim=1)

            loss = torch.nn.functional.cross_entropy(logits, labels)

            total_loss += loss.item()
            num_batches += 1
            accuracy_metric.update(preds, labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_probs.append(probs.cpu())
            all_ids.extend(ids)

    avg_loss = total_loss / num_batches
    accuracy = accuracy_metric.compute().item()

    print(f"🧠 PyTorch Evaluation: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

    # --- Construct final DataFrame ---
    probs_tensor = torch.cat(all_probs, dim=0).numpy()
    result_df = pd.DataFrame({
        "id": all_ids,
        "true_label": all_labels,
        "pred_label": all_preds,
    })

    for i in range(probs_tensor.shape[1]):
        result_df[f"class_{i}_prob"] = probs_tensor[:, i]

    return result_df

###### DEPRECIATED ######

# ------------------------- EXPORT TO ONNX -------------------------

# def export_to_onnx(ckpt_path, export_path, input_dim, hidden_dim, num_classes, num_layers, sequence_len = 1000):
def export_to_onnx(model, input_dim, export_path, sequence_len = 1000):
    """
    WARNING: CANNOT EXPORT PACK_PADDED_SEQUENCES IN ONNIX 
    """
    model.eval()
    
    # Trigger ONNX export mode
    model.exporting_to_onnx = True

    # Create dummy input: [batch_size, sequence_len, input_dim]
    dummy_input = torch.randn(1, sequence_len, input_dim)
    dummy_lengths = torch.tensor([sequence_len], dtype=torch.int64)

    # Export the model
    torch.onnx.export(
        model,
        (dummy_input, dummy_lengths),             
        export_path,                              
        input_names=["sequence"],
        output_names=["logits"],
        dynamic_axes={
            "sequence": {0: "batch_size", 1: "sequence_len"},
            "logits": {0: "batch_size"}
        },
        opset_version=16,
    )
    print(f"✅ Model exported to {export_path}")
    
# ------------------------- EVALUATION ONNX FUNCTION -------------------------

def evaluate_onnx_model(onnx_path, test_df, features, mean, std, batch_size=32):
    # Prepare the dataset
    test_dataset = GazeMouseDataset(test_df, features, augment=False, mean=mean, std=std)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Load ONNX model
    ort_session = ort.InferenceSession(onnx_path)

    # Accuracy tracker
    accuracy_metric = Accuracy(task="multiclass", num_classes=len(torch.unique(test_dataset.task_ids)))

    total_loss = 0
    num_batches = 0
    all_preds = []
    all_labels = []
    all_confidences = []
    all_correct_flags = []

    for batch in test_loader:
        x = batch["sequence"].numpy().astype(np.float32)  # [B, T, F]
        labels = batch["task_id"]  # [B]

        # Run ONNX inference
        ort_inputs = {
            "sequence": x,
        }
        ort_outs = ort_session.run(["logits"], ort_inputs)
        logits = torch.tensor(ort_outs[0])  # Convert to torch for metric compatibility
        probs = torch.softmax(logits, dim=1)
        top_probs, preds = torch.max(probs, dim=1)

        # Compute loss manually if needed (optional)
        loss = torch.nn.functional.cross_entropy(logits, labels)

        total_loss += loss.item()
        num_batches += 1
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_confidences.extend(top_probs.cpu().numpy())
        all_correct_flags.extend((preds == labels).cpu().numpy())
        accuracy_metric.update(preds, labels)

    avg_loss = total_loss / num_batches
    accuracy = accuracy_metric.compute().item()

    print(f"🧠 ONNX Evaluation: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")
    return {
        "labels": all_labels, 
        "predictions": all_preds, 
        "probs" : all_confidences, 
        "correct_flags": all_correct_flags,
        "loss": avg_loss, 
        "accuracy": accuracy
    }