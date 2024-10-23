"""
dataset.py
Customize torch Dataset object
"""

from torch.utils.data import Dataset

class ProteinSequences(Dataset):
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
    
class PdbxDetails(Dataset):
    def __init__(self, sentences, tokenizer, device=None, max_length=512):
        self.sentences = sentences
        self.tokenizer = tokenizer
        self.device = device
        self.max_length = max_length  # Add a max_length parameter to control sequence length

    def __len__(self):
        return len(self.sentences)

    def __getitem__(self, idx):
        return self.sentences[idx]

    def collate(self, raw_batch):
        tokenized_batch = self.tokenizer(
            raw_batch, 
            return_tensors='pt', 
            padding=True,  # Pad to the longest sequence in the batch
            truncation=True,  # Truncate if sequences are too long
            max_length=self.max_length,  # Set the max length for tokenization
            return_length=True  # Return the length of each sequence
        )
        if self.device:  # Move tensors to the specified device
            tokenized_batch = {key: value.to(self.device) for key, value in tokenized_batch.items()}
        
        return tokenized_batch