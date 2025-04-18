import numpy as np

import torch
from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence


class GazeMouseDataset(Dataset):
    def __init__(self, dataset, features, mean=None, std=None):
        
        #Build sample id
        if "id" not in dataset.columns:
            dataset["id"] = dataset["Participant name"].astype(str) + "_" + dataset["Task_id"].astype(str) + "_" + dataset["Task_execution"].astype(str)

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
        self.sequences, self.seq_lengths, self.task_ids = [], [], []
        for _, group in grouped:
            group = group.sort_values('Recording timestamp')  # Ensure time order
            seq_tensor = torch.tensor(group[features].values, dtype=torch.float32)
            self.sequences.append(seq_tensor)
            self.seq_lengths.append(len(seq_tensor))
            self.task_ids.append(group["Task_id"].iloc[0].item() - 1)  # Store associated task_id / make task_id begins at 0

        # Pad sequences for batching
        self.padded_sequences = pad_sequence(self.sequences, batch_first=True, padding_value=0)
        self.seq_lengths = torch.tensor(self.seq_lengths, dtype=torch.int64)
        self.task_ids = torch.tensor(self.task_ids, dtype = torch.int64)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return {
            "sequence": self.padded_sequences[idx],
            "seq_length": self.seq_lengths[idx],
            "task_id": self.task_ids[idx]
        }
        
    def get_task_id(self, idx):
        """Returns the task_id corresponding to a given sequence index."""
        return self.task_ids[idx]