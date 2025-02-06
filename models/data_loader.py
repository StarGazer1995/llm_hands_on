import torch
import torch.nn as nn
import tiktoken
from torch.utils.data import Dataset

class DataLoader(Dataset):
    def __init__(self, train_dataset, eval_dataset, mode, manual_seed=137, batch_size=4, block_size=8):
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        self.manual_seed = manual_seed
        self.batch_size = batch_size
        self.block_size = block_size
        self.data = None
        self.mode = mode
        assert self.mode in ("train", "eval", "test")
        torch.manual_seed(self.manual_seed)
        if self.mode == "train":
            self._load_file(self.train_dataset)
        else:
            self._load_file(self.eval_dataset)

        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}

    def __len__(self):
        return len(self.data)
    
    def encode(self, s):
        return [self.stoi[c] for c in s]

    def decode(self, l):
        return ''.join([self.itos[i] for i in l])

    def _load_file(self, file_path):
        with open(file_path, "r", encoding="utf-8") as f:
            self.data = f.read()
        
        self.chars = sorted(list(set(self.data)))
        self.stoi = {ch: i for i, ch in enumerate(self.chars)}
        self.itos = {i: ch for i, ch in enumerate(self.chars)}
        self.data = self.encode(self.data)
        self.data = torch.tensor(self.data, dtype=torch.long)
        
    def get_batch(self, index):
        idx = torch.randint(len(self.data) - self.block_size, (self.batch_size, ))
        inputs = torch.stack([self.data[i:i + self.block_size] for i in idx])
        targets = torch.stack([self.data[i+1:i+1 + self.block_size] for i in idx])
        return inputs, targets
    
    def __getitem__(self, index):
        return self.get_batch(index)
        
if __name__ == "__main__":
    data_path = "./dataset/input.txt"
    dataset = DataLoader(train_dataset=data_path, eval_dataset=data_path, mode="train")
    for batch_idx, (inputs, targets) in enumerate(dataset):
        print(f"Batch {batch_idx}:")
        print(f"Data: {inputs}")
        print(f"Labels: {targets}")