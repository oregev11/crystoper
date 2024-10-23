"""
dataset.py
Customize torch Dataset object
"""

from torch.utils.data import Dataset

class ProteinSequencesDataset(Dataset):
    def __init__(self, sequences, tokenizer, device=None):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def collate(self, raw_batch):
        sequences = self.tokenizer(raw_batch, return_tensors='pt', padding=True, return_length=True)
        return sequences.to(self.device)
    

class SentencesDataset(Dataset):
    def __init__(self, sequences, tokenizer, device=None):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]

    def collate(self, raw_batch):
        sequences = self.tokenizer(raw_batch, return_tensors='pt', padding=True)
        return sequences.to(self.device)
    
