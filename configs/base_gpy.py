model=dict(
# Hyperparameters
batch_size = 32,
block_size = 8,
max_iters = 5000,
eval_interval = 300,
learning_rate = 3e-4,
device = 'cuda',
eval_iters = 200,
n_embd = 384,
n_head = 6,
n_layer = 6,
dropout = 0.2,
datapath="dataset/input.txt",
vocab_size=65
)