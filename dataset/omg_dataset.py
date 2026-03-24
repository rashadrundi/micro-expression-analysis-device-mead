import torch
from torch.utils.data import Dataset
import json
from torch.nn.utils.rnn import pad_sequence
import numpy as np

class OMGDataset(Dataset):
    def __init__(self, metadata_json):
        self.items = json.load(open(metadata_json))
    
    def __len__(self):
        return len(self.items)
    
    def __getitem__(self, idx):
        current = self.items[idx]

        X = torch.tensor(np.load(current["feature_path"]), dtype=torch.float32)
        y = torch.tensor([float(current["v"]), float(current["a"])], dtype=torch.float32)

        return X, y
    
def collate(batch):
    sequence, labels = zip(*batch)

    padded = pad_sequence(sequence, batch_first=True)
    labels = torch.stack(labels)
    
    return padded, labels