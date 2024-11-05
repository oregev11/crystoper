"""
dataset.py
Customize torch Dataset object
"""
import torch
from torch.utils.data import Dataset
from . import config

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
    

#Dataset for bart vectors inference

class BartDetailsDataset(Dataset):
    def __init__(self, pdb_ids, sequences, details, tokenizer, device=None):
        self.pdb_ids = pdb_ids
        self.sequences = sequences
        self.details = details

        self.tokenizer = tokenizer
        self.device = device

    def __len__(self):
        return len(self.details)

    def __getitem__(self, idx):
        return {'pdb_ids': self.pdb_ids[idx],
                'details': self.details[idx],
                'sequences': self.sequences[idx]}

    def collate(self, raw_batch):

        pdb_ids = [item['pdb_ids'] for item in raw_batch]
        sequences = [item['sequences'] for item in raw_batch]
        details = [item['details'] for item in raw_batch]

        #we pad using max len 250 which is our assumed max length of the pdbx_details
        input_ids = self.tokenizer(details, padding='max_length', max_length=config.N_WORDS_IN_DETAILS, return_tensors='pt')

        return {'pdb_ids' : pdb_ids,
                'sequences' : sequences,
                'details': details,
                'input_ids': input_ids.to(self.device)}



#dataset for custom esm training on bart coded vectors
class Sequence2BartDataset(Dataset):
    def __init__(self, sequences, target_matrices, tokenizer, device=None):
        self.sequences = sequences
        self.tokenizer = tokenizer
        self.target_matrices = target_matrices
        self.device = device

    def __len__(self):
        return len(self.sequences)  # Length of the dataset

    def __getitem__(self, idx):
        return {'sequence': self.sequences[idx],
                'det_vec': self.target_matrices[idx]}

    def collate(self, raw_batch):

        sequences = [item['sequence'] for item in raw_batch]
        det_vec = torch.stack([item['det_vec'] for item in raw_batch])

        encoding = self.tokenizer(sequences, return_tensors="pt",
                                padding=True, truncation=True)


        input_ids = encoding['input_ids']
        attention_mask = encoding['attention_mask']

        return {'input_ids': input_ids.to(self.device),
                'attention_mask': attention_mask.to(self.device),
                'target_matrices': det_vec.to(self.device)}

