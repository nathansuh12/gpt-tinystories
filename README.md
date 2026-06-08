# gpt-tinystories

A decoder-only GPT and a **from-scratch byte-pair-encoding tokenizer**, both implemented in PyTorch and trained on the [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) dataset. The goal was to build the full pipeline — tokenizer, transformer, training loop, and sampling — from the ground up rather than reaching for library abstractions.

Nothing here uses `tiktoken` or Hugging Face model classes: the BPE tokenizer (with the GPT-4 split regex) and the transformer (multi-head causal self-attention, pre-norm blocks, weight-tied head) are written by hand.

## Components

### 1. Regex BPE tokenizer (`regex_tokenizer.py`)

A byte-level BPE tokenizer in the style of the GPT-4 tokenizer:

- Text is first split into chunks using the **GPT-4 split pattern** (a regex that keeps contractions, words, numbers, and punctuation from merging across boundaries), then BPE merges are learned *within* chunks.
- Training starts from the 256 raw byte tokens and greedily merges the most frequent adjacent pair, repeating until it reaches the target vocab size (4096 here, i.e. 3,840 learned merges).
- `encode` re-splits text by the regex and applies the learned merges in priority order; `decode` maps token ids back to bytes and UTF-8 decodes them.
- Merges are pickled to disk (`tokenizer.pkl`) so training only happens once.

Running the tokenizer as a script dumps the full learned vocabulary to `test.md` for inspection.

### 2. GPT model + training (`gpt.py`)

A decoder-only transformer following the nanoGPT structure:

- **Attention** — single-head attention (`Head`) computes scaled dot-product attention with a causal mask (lower-triangular `tril` buffer), and `MultiHeadAttention` runs several heads in parallel, concatenates them, and applies an output projection.
- **Blocks** — each `Block` is pre-norm: `x = x + attn(ln1(x))` then `x = x + ffwd(ln2(x))`, with a position-wise feed-forward network (`Linear → GELU → Linear`) at 4× expansion.
- **Embeddings & head** — separate token and positional embedding tables are summed; the final LM head **shares weights with the token embedding** (weight tying), which removes a large parameter block and ties input/output representations.
- **Training** — standard cross-entropy next-token loss, AdamW, with `estimate_loss()` averaging over multiple batches to report train/val loss during training.
- **Generation** — autoregressive sampling that crops context to `block_size`, softmaxes the final-position logits, and samples with `torch.multinomial`.

## Model configuration

| | |
|---|---|
| Parameters | **~12.3M** (token + position embeddings, 6 transformer blocks; LM head weight-tied) |
| Vocab size | 4,096 (learned BPE) |
| Context length (`block_size`) | 256 |
| Embedding dim (`n_embd`) | 384 |
| Layers (`n_layer`) | 6 |
| Heads (`n_head`) | 6 (head size 64) |
| Dropout | 0.2 |
| Optimizer | AdamW, lr 3e-4 |
| Training iterations | 10,000 |
| Device | auto-selects `mps`, falls back to `cpu` |

## Results

After training, the model generates coherent TinyStories-style prose — simple sentences, recurring child characters, and basic narrative structure — with the local incoherence you'd expect from a ~12M-parameter model at this scale. A full 500-token sample is saved in [`output.md`](output.md).

> **Train/val loss:** _add your final reported values here_ — the training loop prints them every 500 steps; capturing them gives the repo a concrete quantitative result alongside the qualitative sample.

## Setup

```bash
pip install torch datasets regex
```

A GPU (CUDA or Apple Silicon MPS) is recommended for the 10k-iteration training run; it will fall back to CPU but slowly.

## Usage

Train the tokenizer (if not already cached), train the model, and generate a sample:

```bash
python gpt.py
```

This will:
1. Load the TinyStories dataset.
2. Train the BPE tokenizer on first run and cache it to `tokenizer.pkl` (reused afterward).
3. Train the GPT for 10,000 iterations, printing train/val loss periodically.
4. Save weights to `gpt.pt` and write a 500-token generation to `output.md`.

To inspect the learned vocabulary separately:

```bash
python regex_tokenizer.py   # writes the full vocab to test.md
```

## Files

| File | Purpose |
|---|---|
| `gpt.py` | Transformer model, training loop, and sampling |
| `regex_tokenizer.py` | From-scratch regex BPE tokenizer |
| `tinystories_train.md` | Text sample used to train the tokenizer |
| `test.md` | Dump of the learned BPE vocabulary |
| `output.md` | Sample generation from the trained model |

## Notes

- The tokenizer is trained on the text in `tinystories_train.md` (a sample), while the model trains on the first 100,000 stories streamed from the Hugging Face dataset — so the vocabulary is derived from a subset of the full training corpus.
- This is a learning-focused implementation prioritizing clarity of mechanism over throughput (e.g. attention heads run as a Python list rather than a single batched matmul).
