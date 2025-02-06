import torch
import torch.nn as nn
from torch.nn import functional as F

class DataLoader:
    def __init__(self, text_path):
        with open(text_path, "r", encoding='utf-8') as f:
            text = f.open()
        self.chars = sorted(list(set(text)))
        self.vocab_size = len(self.chars)
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.data = torch.tensor(self.encode(text), dtype=torch.long)
        self.n = int(0.9 * len(self.data))
        self.train_data = self.data[:self.n]
        self.val_data = self.data[self.n:]

    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def get_batch(self, split, batch_size, block_size, device="cuda"):
        data = self.train_data if split == 'train' else self.val_data
        ix = torch.randint(len(data) - block_size, (batch_size,))
        x = torch.stack([data[i:i+block_size] for i in ix])
        y = torch.stack([data[i+1:i+block_size+1] for i in ix])
        x, y = x.to(device), y.to(device)
        return x, y

