import torch
import torch.nn as nn
from torch.nn import functional as F

# Hyperparameters
batch_size = 32
block_size = 8
max_iters = 5000
eval_interval = 300
learning_rate = 3e-4
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
n_embd = 384
n_head = 6
n_layer = 6
dropout = 0.2

torch.manual_seed(1337)

# class DataLoader:
#     def __init__(self, text_path):
#         with open(text_path, "r", encoding='utf-8') as f:
#             text = f.open()
#         self.chars = sorted(list(set(text)))
#         self.vocab_size = len(self.chars)
#         self.stoi = {ch: i for i, ch in enumerate(self.chars)}
#         self.itos = {i: ch for i, ch in enumerate(self.chars)}
#         self.data = torch.tensor(self.encode(text), dtype=torch.long)
#         self.n = int(0.9 * len(self.data))
#         self.train_data = self.data[:self.n]
#         self.val_data = self.data[self.n:]

#     def encode(self, s):
#         return [self.stoi[c] for c in s]

#     def decode(self, l):
#         return ''.join([self.itos[i] for i in l])

#     def get_batch(self, split):
#         data = self.train_data if split == 'train' else self.val_data
#         ix = torch.randint(len(data) - block_size, (batch_size,))
#         x = torch.stack([data[i:i+block_size] for i in ix])
#         y = torch.stack([data[i+1:i+block_size+1] for i in ix])
#         x, y = x.to(device), y.to(device)
#         return x, y

class SelfAttention(nn.Module):
    def __init__(self, input_dim, head_size, block_size):
        super().__init__()
        self.head_size = head_size
        self.input_dim = input_dim
        self.mlp_q = nn.Linear(self.input_dim, self.head_size, bias=False)
        self.mlp_k = nn.Linear(self.input_dim, self.head_size, bias=False)
        self.mlp_v = nn.Linear(self.input_dim, self.head_size, bias=False)
        self.register_buffer("tril", torch.tril(torch.ones(block_size, block_size)))

    def forward(self, token):
        batch, time, channel = token.shape
        q = self.mlp_q(token)
        k = self.mlp_k(token)
        v = self.mlp_v(token)
        wei = q @ k.transpose(-2, -1) * self.head_size**-0.5
        wei = wei.masked_fill(self.tril[:time, :time] == 0, float('-inf'))
        wei = F.softmax(wei, dim=-1)
        out = wei @ v
        return out

class MultiHeadAttention(nn.Module):
    def __init__(self, head_num, head_size, block_size):
        super().__init__()
        input_dim = head_num * head_size
        self.heads = nn.ModuleList([SelfAttention(input_dim, head_size, block_size) for _ in range(head_num)])
        self.proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        x = torch.cat([head(x) for head in self.heads], dim=-1)
        x = self.proj(x)
        return x

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.ReLU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)

class Block(nn.Module):
    def __init__(self, head_num, input_dim, block_size):
        super().__init__()
        head_size = input_dim // head_num
        self.sa = MultiHeadAttention(head_num=head_num, head_size=head_size, block_size=block_size)
        self.ffn = FeedForward(input_dim)
        self.layer_norm1 = nn.LayerNorm(input_dim)
        self.layer_norm2 = nn.LayerNorm(input_dim)

    def forward(self, x):
        x = self.sa(self.layer_norm1(x)) + x
        x = self.ffn(self.layer_norm2(x)) + x
        return x

class BigramLanguageModel(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        self.token_embedding_table = nn.Embedding(self.cfg.vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_head, n_embd, self.cfg.block_size) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, self.cfg.vocab_size)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding_table(idx)
        pos_emb = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_emb + pos_emb
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        for _ in range(max_new_tokens):
            idx_con = idx[:, -block_size:]
            logits, loss = self(idx_con)
            logits = logits[:, -1, :]
            probs = F.softmax(logits, dim=-1)
            idx_next = torch.multinomial(probs, num_samples=1)
            idx = torch.cat((idx, idx_next), dim=1)
        return idx

# class Trainer:
#     def __init__(self, model, optimizer, dataloader):
#         self.model = model
#         self.optimizer = optimizer
#         self.dataloader = data_loader

#     def estimate_loss(self):
#         out = {}
#         self.model.eval()
#         for split in ['train', 'val']:
#             losses = torch.zeros(eval_iters)
#             for k in range(eval_iters):
#                 X, Y = self.data_loader.get_batch(split)
#                 logits, loss = self.model(X, Y)
#                 losses[k] = loss.item()
#             out[split] = losses.mean()
#         self.model.train()
#         return out

#     def train(self):
#         for iter in range(max_iters):
#             if iter % eval_interval == 0 or iter == max_iters - 1:
#                 losses = self.estimate_loss()
#                 print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

#             xb, yb = self.data_loader.get_batch('train')
#             logits, loss = self.model(xb, yb)
#             self.optimizer.zero_grad(set_to_none=True)
#             loss.backward()
#             self.optimizer.step()

# if __name__ == "__main__":

#     data_loader = DataLoader("./dataset/input.txt")
#     model = BigramLanguageModel().to(device)
#     optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
#     trainer = Trainer(model, optimizer)

#     # Train the model
#     trainer.train()

#     # Generate text
#     context = torch.zeros((1, 1), dtype=torch.long, device=device)
#     print(data_loader.decode(model.generate(context, max_new_tokens=500)[0].tolist()))