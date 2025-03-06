import torch
import torch.nn as nn
import torch.optim as optim
import pytorch_lightning as pl
from torch.utils.data import DataLoader
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import pandas as pd

from utils.dataset import GazeMouseDataset
    
# ------------------------- LSTM AUTOENCODER MODEL -------------------------
class LSTMAutoencoder(pl.LightningModule):
    def __init__(self, input_dim, hidden_dim, latent_dim, num_layers=1, learning_rate=0.001):
        super(LSTMAutoencoder, self).__init__()

        self.hidden_dim = hidden_dim
        self.latent_dim = latent_dim
        self.num_layers = num_layers
        self.learning_rate = learning_rate

        # Encoder LSTM (Bidirectional)
        self.encoder_lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Bottleneck (Latent Representation)
        self.fc = nn.Linear(hidden_dim, latent_dim)

        # Decoder LSTM (Bidirectional)
        self.decoder_lstm = nn.LSTM(latent_dim, hidden_dim, num_layers, batch_first=True, bidirectional=True)

        # Output Layer (Reconstruct Input)
        self.output_layer = nn.Linear(2*hidden_dim, input_dim)

        # Loss Function
        self.criterion = nn.MSELoss()

    def forward(self, x, seq_lengths):
        # Pack padded input sequence
        packed_input = pack_padded_sequence(x, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Encoder LSTM
        packed_encoded_output, (hidden, _) = self.encoder_lstm(packed_input)

        # Get last hidden state (bottleneck)
        last_hidden = hidden[-1]

        # Pass through bottleneck layer
        latent = self.fc(last_hidden).unsqueeze(1).expand(-1, x.size(1), -1)

        # Pack the latent representation for the decoder
        packed_latent = pack_padded_sequence(latent, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)

        # Decoder LSTM
        packed_decoded_output, _ = self.decoder_lstm(packed_latent)

        # Unpack sequence (Ensure output has same length as input)
        unpacked_output, _ = pad_packed_sequence(packed_decoded_output, batch_first=True, padding_value=0.0, total_length=x.size(1))

        # Output reconstruction
        output = self.output_layer(unpacked_output)
        return output

    def training_step(self, batch, batch_idx):
        x, seq_lengths = batch["sequence"], batch["seq_length"]
        reconstructed = self.forward(x, seq_lengths)

        # Masking for padded values
        mask = (x != 0).float()
        loss = self.criterion(reconstructed * mask, x * mask)

        self.log("train_loss", loss, prog_bar=True, on_epoch=True, logger=True)

        return loss

    def configure_optimizers(self):
        return optim.Adam(self.parameters(), lr=self.learning_rate)

# ------------------------- TRAINING FUNCTION -------------------------
def train_autoencoder(dataset, features, batch_size=32, hidden_dim=64, latent_dim=32, num_layers=2, learning_rate=0.001, num_epochs=20):
    """
    Train the LSTM Autoencoder and save the final model.

    Args:
    - dataset(pd.DataFramce): Processed gaze dataset.
    - features (list): List of feature column names.
    - batch_size (int): Batch size for training.
    - hidden_dim (int): LSTM hidden state size.
    - latent_dim (int): Bottleneck size.
    - num_layers (int): Number of LSTM layers.
    - learning_rate (float): Learning rate.
    - num_epochs (int): Number of training epochs.

    Returns:
    - model: The trained LSTM Autoencoder model.
    """
    
    # Create dataset & dataloader
    gazeMouseDataset = GazeMouseDataset(dataset, features)
    dataloader = DataLoader(gazeMouseDataset, batch_size=batch_size, shuffle=True, num_workers = 10)

    # Define model
    input_dim = len(features)
    model = LSTMAutoencoder(input_dim, hidden_dim, latent_dim, num_layers, learning_rate)

    # Train using PyTorch Lightning Trainer
    trainer = pl.Trainer(
        max_epochs=num_epochs,
        accelerator="auto",
        log_every_n_steps=10,
        callbacks=[pl.callbacks.ModelCheckpoint(monitor="train_loss", mode="min", save_top_k=1)]
    )
    trainer.fit(model, dataloader)

    return model

# ------------------------- Latent Space Function -------------------------
def get_latent_representation(model, dataset, features):
    """
    Extracts the latent space representation from the trained LSTM Autoencoder.

    Args:
        - model (LSTMAutoencoder): Trained PyTorch Lightning model.
        - dataset(pd.DataFramce): Processed gaze dataset.
        - features (list): List of feature column names.

    Returns:
        torch.Tensor: Latent space representations of shape (batch_size, latent_dim).
    """
    
    model.eval() 
    gazeMouseDataset = GazeMouseDataset(dataset, features)

    with torch.no_grad():  # Disable gradient computation for efficiency
        # Pack the padded sequence
        packed_input = pack_padded_sequence(gazeMouseDataset.padded_sequences, gazeMouseDataset.seq_lengths, batch_first=True, enforce_sorted=False)

        # Forward pass through the encoder LSTM
        packed_encoded_output, (hidden, _) = model.encoder_lstm(packed_input)

        # Extract the last hidden state (bottleneck/latent space)
        latent_representation = hidden[-1]  # Shape: (batch_size, hidden_dim)

        # Pass through bottleneck layer (fully connected layer)
        latent_representation = model.fc(latent_representation)  # Shape: (batch_size, latent_dim)

    return latent_representation, gazeMouseDataset.task_ids

# ------------------------- Latent Space Function -------------------------
def get_latent_representation(model, dataset, features):
    """
    Extracts the latent space representation from the trained LSTM Autoencoder.

    Args:
        - model (LSTMAutoencoder): Trained PyTorch Lightning model.
        - dataset(pd.DataFramce): Processed gaze dataset.
        - features (list): List of feature column names.

    Returns:
        torch.Tensor: Latent space representations of shape (batch_size, latent_dim).
    """
    
    model.eval() 
    gazeMouseDataset = GazeMouseDataset(dataset, features)

    with torch.no_grad():  # Disable gradient computation for efficiency
        # Pack the padded sequence
        packed_input = pack_padded_sequence(gazeMouseDataset.padded_sequences, gazeMouseDataset.seq_lengths, batch_first=True, enforce_sorted=False)

        # Forward pass through the encoder LSTM
        packed_encoded_output, (hidden, _) = model.encoder_lstm(packed_input)

        # Extract the last hidden state (bottleneck/latent space)
        latent_representation = hidden[-1]  # Shape: (batch_size, hidden_dim)

        # Pass through bottleneck layer (fully connected layer)
        latent_representation = model.fc(latent_representation)  # Shape: (batch_size, latent_dim)

    return latent_representation, gazeMouseDataset.task_ids
