import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import CyclicLR
import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from torch.nn.utils.rnn import pack_padded_sequence
from torchmetrics.classification import Accuracy
import pandas as pd
import wandb
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint

from utils.dataset import GazeMouseDataset


# ------------------------- LSTM CLASSIFIER MODEL -------------------------
class LSTMClassifier(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, num_classes, num_layers=1, learning_rate=0.001):
        super(LSTMClassifier, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        # LSTM for sequence processing
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=False)

        # Fully connected layer for classification
        self.fc = nn.Linear(hidden_dim, num_classes)

        # Loss function (CrossEntropy for classification)
        # Automatically appliey Softmax internally
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, x, seq_lengths):
        # Pack padded sequence for LSTM efficiency
        packed_input = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Forward pass through LSTM
        _, (hidden, _) = self.lstm(packed_input)

        # Use the last hidden state
        last_hidden = hidden[-1]

        # Pass through classification layer
        logits = self.fc(last_hidden)

        return logits

    def training_step(self, batch, batch_idx):
        x, seq_lengths, task_ids = batch["sequence"], batch["seq_length"], batch["task_id"]
        logits = self.forward(x, seq_lengths)

        # Compute loss
        loss = self.criterion(logits, task_ids)

        # Logging
        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)

        return loss
    
    def validation_step(self, batch, batch_idx):
        x, seq_lengths, task_ids = batch["sequence"], batch["seq_length"], batch["task_id"]
        logits = self.forward(x, seq_lengths)

        # Compute loss
        loss = self.criterion(logits, task_ids)

        # Log validation loss
        self.log("val_loss", loss, prog_bar=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        optimizer = optim.Adam(self.parameters(), lr=self.learning_rate)
        
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
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_loss"},
            "gradient_clip_val": 1.0  # Clip gradients to prevent explosion
        }

# ------------------------- TRAINING FUNCTION -------------------------
def train_classifier(dataset, 
                     features, 
                     batch_size=32, 
                     hidden_dim=64, 
                     num_layers=2, 
                     learning_rate=0.001, 
                     num_epochs=20,
                     val_split = 0.2,
                     use_wandb = True):
    """
    Trains the LSTM classifier on the gaze/mouse movement dataset.

    Args:
        dataset(pd.DataFrame): Processed gaze dataset.
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
    gazeMouseDataset = GazeMouseDataset(dataset, features)
    
    total_size = len(gazeMouseDataset)
    val_size = int(total_size * val_split)
    train_size = total_size - val_size
    train_set, val_set = random_split(gazeMouseDataset, [train_size, val_size])

    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_set, batch_size=batch_size, shuffle=False)
    # Determine number of classes (unique task IDs)
    num_classes = len(torch.unique(gazeMouseDataset.task_ids))

    # Define model
    input_dim = len(features)
    model = LSTMClassifier(input_dim, hidden_dim, num_classes, num_layers, learning_rate)
    
    # Initialize WandB logger
    if use_wandb:
        wandb_logger = WandbLogger(project="GazeMouse_Classification", log_model="all")
    else:
        wandb_logger = None

    # Define trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        log_every_n_steps=10,
        logger=wandb_logger,
        callbacks=[
            # EarlyStopping(monitor="val_loss", patience=5, mode="min"),  # Stop if val_loss doesn't improve for 5 epochs
            ModelCheckpoint(monitor="val_loss", mode="min", save_top_k=1, filename="best_lstm_classifier")
        ]
    )

    # Train model
    trainer.fit(model, train_loader, val_loader)

    return model

def evaluate_model(model, dataset, features, batch_size=32):
    """
    Loads a trained model from a checkpoint and evaluates it on a test dataset.

    Args:
        model (LSTMClassifier): Loaded trained model.
        dataset (pd.DataFrame): Processed gaze dataset
        features (List[str]): List of features
        batch_size (int): Batch size for evaluation.

    Returns:
        dict: Dictionary containing evaluation metrics (accuracy, loss) and outputs
    """
    
    test_dataset = GazeMouseDataset(dataset, features)

    # Create DataLoader
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # Define accuracy metric
    accuracy_metric = Accuracy(task="multiclass", num_classes=model.fc.out_features).to(model.device)

    total_loss = 0
    num_batches = 0
    all_preds = []
    all_labels = []

    with torch.no_grad():  # No gradients needed for evaluation
        for batch in test_loader:
            x, seq_lengths, task_ids = batch["sequence"].to(model.device), batch["seq_length"].to(model.device), batch["task_id"].to(model.device)

            # Forward pass
            logits = model.forward(x, seq_lengths)

            # Compute loss
            loss = model.criterion(logits, task_ids)
            total_loss += loss.item()
            num_batches += 1

            # Get predicted labels
            preds = torch.argmax(logits, dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(task_ids.cpu().numpy())

            # Update accuracy metric
            accuracy_metric.update(preds, task_ids)

    # Compute final metrics
    avg_loss = total_loss / num_batches
    accuracy = accuracy_metric.compute().item()

    print(f"✅ Evaluation Complete: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}")

    return {"labels": all_labels, "predictions": all_preds, "loss": avg_loss, "accuracy": accuracy}
