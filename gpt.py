import os
import torch
import torch.nn as nn
from torch.nn import functional as F

from regex_tokenizer import RegexTokenizer

from datasets import load_dataset

# hyperparameters
batch_size = 64
block_size = 256
max_iters = 10000
eval_interval = 500
learning_rate = 3e-4
device = 'mps' if torch.backends.mps.is_available() else 'cpu'
eval_iters = 100
n_embd = 384
n_layer = 6
n_head = 6
dropout = 0.2
# ------------

train = load_dataset("roneneldan/TinyStories", split="train")
val = load_dataset("roneneldan/TinyStories", split="validation")

tokenizer = RegexTokenizer()
if os.path.exists('tokenizer.pkl'):
    tokenizer.load('tokenizer.pkl')
else:
    with open('tinystories_train.md', 'r') as f:
        tokenizer.train(f.read(), vocab_size=4096)
    tokenizer.save('tokenizer.pkl')

vocab_size = tokenizer.vocab_size
train_text = '\n'.join(train['text'][:100_000])
val_text = '\n'.join(val['text'][:10_000])

train = torch.tensor(tokenizer.encode(train_text), dtype=torch.long)
val = torch.tensor(tokenizer.encode(val_text), dtype=torch.long)

def get_batch(split):
    data = train if split == 'train' else val
    ix = torch.randint(len(data) - block_size, (batch_size,))
    x = torch.stack([data[i:i+block_size] for i in ix])
    y = torch.stack([data[i+1:i+block_size+1] for i in ix])
    x, y = x.to(device), y.to(device)
    return x, y

@torch.no_grad()
def estimate_loss():
    out = {}
    model.eval()
    for split in ['train', 'val']:
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            X, Y = get_batch(split)
            logits, loss = model(X, Y)
            losses[k] = loss.item()
        out[split] = losses.mean()
    model.train()
    return out

class Head(nn.Module):

    def __init__(self, head_size):
        super().__init__()

        self.key = nn.Linear(n_embd, head_size, bias = False )
        self.query = nn.Linear(n_embd, head_size, bias = False)
        self.value = nn.Linear(n_embd, head_size, bias = False)

        self.register_buffer('tril', torch.tril(torch.ones(block_size, block_size)))
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape

        key = self.key(x)
        query = self.query(x)

        head_size = key.shape[-1]

        wei = query @ key.transpose(-2, -1) * head_size ** -0.5
        wei = wei.masked_fill(self.tril[:T, :T] == 0, float('-inf'))
        wei = F.softmax(wei, dim = -1)
        wei = self.dropout(wei)

        value = self.value(x)
        output = wei @ value

        return output
    
class MultiHeadAttention(nn.Module):

    def __init__(self, num_heads, head_size):
        super().__init__()
        self.heads = nn.ModuleList([Head(head_size) for _ in range(num_heads)])
        self.proj = nn.Linear(n_embd, n_embd)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = torch.cat([h(x) for h in self.heads], dim = -1)
        out = self.dropout(self.proj(out))

        return out

class FeedForward(nn.Module):
    def __init__(self, n_embd):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout)
        )
    
    def forward(self, x):
        return self.net(x)
    
class Block(nn.Module):
    def __init__(self, n_embd, n_heads):
        super().__init__()
        
        head_size = n_embd // n_heads
        self.sa = MultiHeadAttention(n_heads, head_size)
        self.ffwd = FeedForward(n_embd)
        self.ln1 = nn.LayerNorm(n_embd)
        self.ln2 = nn.LayerNorm(n_embd)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x

class GPT(nn.Module):

    def __init__(self, vocab_size):
        super().__init__()
        # each token directly reads off the logits for the next token from a lookup table
        self.token_embedding_table = nn.Embedding(vocab_size, n_embd)
        self.position_embedding_table = nn.Embedding(block_size, n_embd)
        self.blocks = nn.Sequential(*[Block(n_embd, n_heads=n_head) for _ in range(n_layer)])
        self.ln_f = nn.LayerNorm(n_embd)
        self.lm_head = nn.Linear(n_embd, vocab_size, bias=False)
        self.lm_head.weight = self.token_embedding_table.weight
        

    def forward(self, idx, targets=None):
        B, T = idx.shape
        ''' we are gettin gthe shape and the time stamps or position of idx. we then
            create a token embedding and a position embedding for this imput idx
            and add them togehter in to x, embedding both the token embedding and position, and we
            have a lookup table to then revert it back'''
         
        tok_embd = self.token_embedding_table(idx) # (B,T,C)
        pos_embd = self.position_embedding_table(torch.arange(T, device=device))
        x = tok_embd + pos_embd
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.lm_head(x) #(B,T,vocabsize)

        if targets is None:
            loss = None
        else:
            B, T, C = logits.shape
            logits = logits.view(B*T, C)
            targets = targets.view(B*T)
            loss = F.cross_entropy(logits, targets)

        return logits, loss

    def generate(self, idx, max_new_tokens):
        # idx is (B, T) array of indices in the current context
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -block_size:]
            # get the predictions
            logits, loss = self(idx_cond)
            # focus only on the last time step
            logits = logits[:, -1, :] # becomes (B, C)
            # apply softmax to get probabilities
            probs = F.softmax(logits, dim=-1) # (B, C)
            # sample from the distribution
            idx_next = torch.multinomial(probs, num_samples=1) # (B, 1)
            # append sampled index to the running sequence
            idx = torch.cat((idx, idx_next), dim=1) # (B, T+1)
        return idx

model = GPT(vocab_size).to(device)
print(f"using device: {device}")

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for step in range(max_iters):

    if step % eval_interval == 0:
        losses = estimate_loss()
        print(f"step {step}: train loss {losses['train']:.4f}, val loss {losses['val']:.4f}")

    # sample a batch of data
    xb, yb = get_batch('train')

    # evaluate the loss
    logits, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()


torch.save(model.state_dict(), 'gpt.pt')

# generate from the model
context = torch.zeros((1, 1), dtype=torch.long, device=device)
generated = tokenizer.decode(model.generate(context, max_new_tokens=500)[0].tolist())
print(generated)
with open('output.md', 'w') as f:
    f.write(generated)
