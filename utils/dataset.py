# TODO: Modify to accomodate JCAFNet inputs

import numpy as np
import random

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence

# -------------------- LSTM DATASET --------------------
class GazeMouseDatasetLSTM(Dataset):
    def __init__(self, dataset, features, augment = False, mean=None, std=None):
        
        # Batch of different shape depending on the network
        self.augment = augment
        self.features = features
        
        #Build sample id
        if "id" not in dataset.columns:
            dataset["id"] = (
                dataset["Participant name"].astype(str) 
                + "_" 
                + dataset["Task_id"].astype(str) 
                + "_" 
                + dataset["Task_execution"].astype(str)
            )

        # Compute relative timestamps per sequence
        dataset['Relative timestamp'] = dataset.groupby('id')['Recording timestamp'].transform(lambda x: x - x.min())

        # Normalize features using provided mean/std or compute from dataset
        if mean is None or std is None:
            self.mean = dataset[features].mean()
            self.std = dataset[features].std()
        else:
            self.mean = mean
            self.std = std
        
        dataset[features] = (dataset[features] - self.mean) / self.std

        # Group data by sample_id
        grouped = dataset.groupby('id')

        # Store sequences & lengths
        self.sequences, self.seq_lengths, self.task_ids, self.ids = [], [], [], []
        for sample_id, group in grouped:
            group = group.sort_values('Recording timestamp')  # Ensure time order
            seq_tensor = torch.tensor(group[features].values, dtype=torch.float32)
            self.sequences.append(seq_tensor)
            self.seq_lengths.append(len(seq_tensor))
            self.task_ids.append(group["Task_id"].iloc[0].item() - 1)  # Store associated task_id / make task_id begins at 0
            self.ids.append(sample_id)

        # Pad sequences for batching
        # self.padded_sequences = pad_sequence(self.sequences, batch_first=True, padding_value=0)
        # self.seq_lengths = torch.tensor(self.seq_lengths, dtype=torch.int64)
        # self.task_ids = torch.tensor(self.task_ids, dtype = torch.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq, task_id = self.samples[idx]
        seq_len = len(seq)

        if self.augment:
            seq = self._augment_sequence(seq, self.seq_lengths[idx])
        
        return {
            "sequence": pad_sequence([seq], batch_first=True)[0],
            "seq_length": torch.tensor(seq_len, dtype=torch.int64),
            "task_id": torch.tensor(task_id, dtype=torch.int64),
        }
        
    def get_task_id(self, idx):
        """Returns the task_id corresponding to a given sequence index."""
        return self.task_ids[idx]
    
    def _augment_sequence(self, x, true_len):
        """Apply augmentations only to the unpadded part of the sequence."""
        x = x.clone()  # Avoid modifying original data
        x_real = x[:true_len]  # Only augment the valid part

        # Gaussian noise
        noise = torch.randn_like(x_real) * 0.01

        # Random temporal jitter (±5 timesteps)
        shift = random.randint(-5, 5)
        jittered = torch.roll(x_real, shifts=shift, dims=0)

        # Optional random zero masking
        if random.random() < 0.1:
            mask_len = random.randint(5, min(20, len(jittered)))
            start = random.randint(0, max(0, len(jittered) - mask_len))
            jittered[start:start + mask_len] = 0

        # Replace unpadded part with augmented version
        x[:true_len] = jittered + noise
        return x

# -------------------- JCAFNET DATASET --------------------
class GazeMouseDatasetJCAFNet(Dataset):
    def __init__(self, dataset, gaze_features, mouse_features, joint_features, augment=False, mean=None, std=None):
        self.augment = augment
        self.gaze_features = gaze_features
        self.mouse_features = mouse_features
        self.joint_features = joint_features
        self.features = gaze_features + mouse_features + joint_features 

        # Build sample ID
        if "id" not in dataset.columns:
            dataset["id"] = (
                dataset["Participant name"].astype(str) 
                + "_" 
                + dataset["Task_id"].astype(str) 
                + "_" 
                + dataset["Task_execution"].astype(str)
            )

        # Normalize
        if mean is None or std is None:
            self.mean = dataset[self.features].mean()
            self.std = dataset[self.features].std()
        else:
            self.mean = mean
            self.std = std

        dataset[self.features] = (dataset[self.features] - self.mean) / self.std

        # Group data by sample_id
        grouped = dataset.groupby("id")
        
        self.samples = [] # Tuple grouping id, gaze features, mouse features and joint features
        for sample_id, group in grouped:
            group = group.sort_values("Recording timestamp")
            gaze_sequence = torch.tensor(group[self.gaze_features].values, dtype=torch.float32)
            mouse_sequence = torch.tensor(group[self.mouse_features].values, dtype=torch.float32)
            joint_sequence = torch.tensor(group[self.joint_features].values, dtype=torch.float32)
            task_id = group["Task_id"].iloc[0].item() - 1
            self.samples.append((gaze_sequence, mouse_sequence, joint_sequence, task_id))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        gaze_sequence, mouse_sequence, joint_sequence, task_id = self.samples[idx]

        if self.augment:
            gaze_sequence = self._augment_sequence(gaze_sequence)
            mouse_sequence = self._augment_sequence(mouse_sequence)
            joint_sequence = self._augment_sequence(joint_sequence)

        return {
            "gaze": gaze_sequence.T, # [channels, time]
            "mouse": mouse_sequence.T,
            "joint": joint_sequence.T,
            "label": torch.tensor(task_id, dtype=torch.int64),
        }

    def _augment_sequence(self, x):
        """Apply augmentations only to the full sequence (no masking here)."""
        x = x.clone()
        noise = torch.randn_like(x) * 0.01
        shift = random.randint(-5, 5)
        jittered = torch.roll(x, shifts=shift, dims=0)

        if random.random() < 0.1:
            mask_len = random.randint(5, min(20, len(jittered)))
            start = random.randint(0, max(0, len(jittered) - mask_len))
            jittered[start:start + mask_len] = 0

        return jittered + noise