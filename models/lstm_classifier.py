import torch
import torch.nn as nn
import torch.optim as optim
from torch.nn.utils.rnn import pack_padded_sequence
from torch.optim.lr_scheduler import CyclicLR
from torchmetrics.classification import Accuracy

import pytorch_lightning as pl

from models.tnc_block import TCN

from utils.losses import FocalLoss

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
        # self.conv1 = nn.Conv1d(in_channels=input_dim, out_channels=64, kernel_size=3, padding=1)
        # self.relu = nn.ReLU()
        # self.conv2 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=3, padding=1)
        # self.dropout_cnn = nn.Dropout(0.3)
        
        # TCN feature extractor
        self.tcn = TCN(input_dim, [64, 128], kernel_size=3, dropout=0.3)

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
        # x = self.relu(self.conv1(x))        # [batch, 64, seq_len]
        # x = self.relu(self.conv2(x))        # [batch, 128, seq_len]
        # x = self.dropout_cnn(x)
        x = self.tcn(x)
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


