import pandas as pd
import numpy as np
import wandb

import onnxruntime as ort

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import CyclicLR
from torchmetrics.classification import Accuracy

from sklearn.model_selection import train_test_split

import pytorch_lightning as pl
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils.dataset import GazeMouseDataset

# ------------------------- LSTM CLASSIFIER MODEL -------------------------
class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Focal Loss for classification.

        Args:
            alpha (float): Weighting factor for the class imbalance.
            gamma (float): Focusing parameter to reduce relative loss for well-classified examples.
            reduction (str): 'mean' or 'sum'
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, logits, targets):
        ce_loss = F.cross_entropy(logits, targets, reduction='none')  # [B]
        pt = torch.exp(-ce_loss)  # [B]
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss  # [B]
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


# ------------------------- LSTM CLASSIFIER MODEL -------------------------
class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, learning_rate=0.001):
        super(LSTMClassifier, self).__init__()
        self.save_hyperparameters()
        self.train_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        self.val_accuracy = Accuracy(task="multiclass", num_classes=num_classes)
        
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate
        
        # CNN: 1D conv to extract local patterns from feature sequences
        self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        self.dropout_cnn = nn.Dropout(0.3)

        # LSTM for sequence processing
        self.lstm = nn.LSTM(128, hidden_dim, num_layers, batch_first=True, dropout=0.3, bidirectional=True)
        self.dropout = nn.Dropout(0.2)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim*2, num_classes)

        # Loss function (CrossEntropy for classification)
        # Automatically appliey Softmax internally
        # self.criterion = nn.CrossEntropyLoss(label_smoothing=0.1)
        self.criterion = FocalLoss(alpha=1.0, gamma=2.0)

    def forward(self, x, seq_lengths):
        
        # x: [batch, seq_len, input_dim] → permute for CNN
        x = x.permute(0, 2, 1)  # [batch, input_dim, seq_len]
        x = self.relu(self.conv1(x))        # [batch, 64, seq_len]
        x = self.relu(self.conv2(x))        # [batch, 128, seq_len]
        x = self.dropout_cnn(x)
        x = x.permute(0, 2, 1)  # [batch, seq_len, 128]: that way we can use pack_padded_sequence for different lengths handling
        
        if getattr(self, "exporting_to_onnx", False):
            # ONNX export path — no packing
            lstm_out, (hidden, _) = self.lstm(x)
        else:
            # Normal path — pack for efficiency
            packed_input = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
            _, (hidden, _) = self.lstm(packed_input)

        # Use the last hidden state
        forward_last = hidden[-2]  # Forward hidden from last LSTM layer
        backward_last = hidden[-1]  # Backward hidden from last LSTM layer
        last_hidden = torch.cat([forward_last, backward_last], dim=1)

        # Pass through classification layer
        logits = self.fc(last_hidden)

        return logits

    def training_step(self, batch, batch_idx):
        x, seq_lengths, task_ids = batch["sequence"], batch["seq_length"], batch["task_id"]
        logits = self.forward(x, seq_lengths)

        # Compute loss
        loss = self.criterion(logits, task_ids)
        preds = torch.argmax(logits, dim=1)
        self.train_accuracy.update(preds, task_ids)

        # Logging
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("train_acc", self.train_accuracy, prog_bar=True, on_epoch=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, seq_lengths, task_ids = batch["sequence"], batch["seq_length"], batch["task_id"]
        logits = self.forward(x, seq_lengths)

        # Compute loss
        loss = self.criterion(logits, task_ids)
        preds = torch.argmax(logits, dim=1)
        self.val_accuracy.update(preds, task_ids)

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)
        self.log("val_acc", self.val_accuracy, prog_bar=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate, weight_decay=1e-4)
        
        # Define Cyclical Learning Rate scheduler
        # scheduler = CyclicLR(
        #     optimizer, 
        #     base_lr=1e-5,   # Minimum learning rate
        #     max_lr=self.learning_rate,    # Maximum learning rate
        #     step_size_up=200,  # Increase LR over 200 iterations
        #     mode="triangular2",  # More aggressive CLR strategy
        #     cycle_momentum=False  # Recommended for Adam optimizer
        # )
        
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5, patience=5)
        
        return {
            "optimizer": optimizer, 
            # "lr_scheduler": {"scheduler": scheduler, "interval": "step"},  # Adjusts learning rate at each step
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_acc"},
            "gradient_clip_val": 1.0  # Clip gradients to prevent explosion
        }
        
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

# ------------------------- TRAINING FUNCTION -------------------------
def train_classifier(train_df,
                     val_df, 
                     features, 
                     batch_size=32, 
                     hidden_dim=64, 
                     num_layers=2, 
                     learning_rate=0.001, 
                     num_epochs=20,
                     use_wandb = True):
    """
    Trains the LSTM classifier on the gaze/mouse movement dataset.

    Args:
        train_df (pd.DataFrame): Training set.
        val_df (pd.DataFrame): Validation set.
        features (list): List of input feature column names.
        batch_size (int): Batch size for training.
        hidden_dim (int): Number of hidden units in LSTM.
        num_layers (int): Number of LSTM layers.
        learning_rate (float): Learning rate for optimizer.
        num_epochs (int): Number of epochs for training.
        val_split (float): Percentage of data for validation.
        test_split (float): Percentage of data for testing.
        use_wandb (bool): Whether to log training in Weights & Biases.

    Returns:
        model (LSTMClassifier): Trained PyTorch Lightning model.
    """

    # Create dataset & dataloader
    train_set = GazeMouseDataset(train_df, features)
    mean, std = train_set.mean, train_set.std
    val_set = GazeMouseDataset(val_df, features, mean, std)

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    
    # Define model
    num_classes = len(torch.unique(train_set.task_ids))
    input_dim = len(features)
    model = LSTMClassifier(input_dim, hidden_dim, num_classes, num_layers, learning_rate)
    
    # Initialize WandB logger
    if use_wandb:
        wandb_logger = WandbLogger(project="GazeMouse_Classification", log_model="all")
    else:
        wandb_logger = None
        
    #Checkpoint callback
    checkpoint_callback = ModelCheckpoint(
        monitor="val_acc",
        mode="max",
        save_top_k=1,
        filename="epoch{epoch:02d}-val_acc{val_acc:.2f}",
        auto_insert_metric_name=False
    )
    
    checkpoint_callback_last = ModelCheckpoint(
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

# ------------------------- EXPORT TO ONNX -------------------------

def export_to_onnx(ckpt_path, export_path, input_dim, hidden_dim, num_classes, num_layers, sequence_len = 1000):
    """
    WARNING: CANNOT EXPORT PACK_PADDED_SEQUENCES IN ONNIX 
    """
    
    model = LSTMClassifier(input_dim, hidden_dim, num_classes, num_layers)
    model.load_state_dict(torch.load(ckpt_path)["state_dict"])
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

# ------------------------- EVALUATION FUNCTION -------------------------

def evaluate_onnx_model(onnx_path, test_df, features, mean, std, batch_size=32):
    # Prepare the dataset
    test_dataset = GazeMouseDataset(test_df, features, mean=mean, std=std)
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
    
def evaluate_pytorch_model(
    model,
    df,
    features,
    mean,
    std,
    batch_size=32,
):
    model.eval()
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    
    # Untrigger ONNX export mode
    model.exporting_to_onnx = False

    dataset = GazeMouseDataset(df, features, mean=mean, std=std)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

    accuracy_metric = Accuracy(task="multiclass", num_classes=len(torch.unique(dataset.task_ids))).to(device)

    all_preds, all_labels, all_confidences, all_correct_flags = [], [], [], []
    total_loss = 0
    num_batches = 0

    with torch.no_grad():
        for batch in loader:
            x = batch["sequence"].to(model.device)
            lengths = batch["seq_length"].to(model.device)
            labels = batch["task_id"].to(model.device)

            logits = model(x, lengths)
            probs = torch.softmax(logits, dim=1)
            top_probs, preds = torch.max(probs, dim=1)

            loss = torch.nn.functional.cross_entropy(logits, labels)

            total_loss += loss.item()
            num_batches += 1
            accuracy_metric.update(preds, labels)

            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            all_confidences.extend(top_probs.cpu().numpy())
            all_correct_flags.extend((preds == labels).cpu().numpy())

    avg_loss = total_loss / num_batches
    accuracy = accuracy_metric.compute().item()

    print(f"🧠 PyTorch Evaluation: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

    return {
        "labels": all_labels, 
        "predictions": all_preds, 
        "probs" : all_confidences, 
        "correct_flags": all_correct_flags,
        "loss": avg_loss, 
        "accuracy": accuracy
    }


