from datasets import load_dataset, concatenate_datasets
import regex as re

#moved dataset to txt file
'''train = load_dataset("roneneldan/TinyStories", split="train")
val = load_dataset("roneneldan/TinyStories", split="validation")
ds = concatenate_datasets([train, val])'''


def get_stats(ids, counts = None):
    if counts is None:
        counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1

    return counts

def merge(ids, pair, idx):
    new_ids = []
    i = 0

    while i < len(ids):
        if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
            new_ids.append(idx)
            i += 2
        else:
            new_ids.append(ids[i])
            i += 1

    return new_ids

GPT4_SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,3}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

class RegexTokenizer():
    def __init__(self):
        super().__init__()

        self.merges = {}
        self.vocab = {idx: bytes([idx]) for idx in range(256)}

        self.regex = GPT4_SPLIT_PATTERN

    def train(self, text, vocab_size, verbose = False):
        assert vocab_size > 256
        num_merges = vocab_size - 256

        text_chunk = re.findall(self.regex, text)
        ids = [list(ch.encode('utf-8')) for ch in text_chunk]

        for i in range(num_merges):
            stats = {}

            for chunk_ids in ids:
                get_stats(chunk_ids, stats)
            
            if not stats:
                break
            top_pair = max(stats, key=stats.get)
            idx = 256 + i
            print(f" merging pair {top_pair} into a new token {idx}")

            ids = [merge(chunk_ids, top_pair, idx) for chunk_ids in ids]
            self.merges[top_pair] = idx
            self.vocab[idx] = self.vocab[top_pair[0]] + self.vocab[top_pair[1]]
        
            if verbose:
                print(f"merge {i+1}/{num_merges}: {top_pair} -> {idx} ({self.vocab[idx]}) had {stats[top_pair]} occurrences")
    
    def decode(self, ids):
        cat =[]

        for chunk_ids in ids:
            cat.append(self.vocab[chunk_ids])

        tokens = b"".join(cat)
        text = tokens.decode('utf-8', errors='replace')

        return text
    
    def encode(self, text):
        text_chunk = re.findall(self.regex, text)
        ids  = [list(ch.encode('utf-8')) for ch in text_chunk]

        encoded_ids = []
        for chunk_ids in ids:
            while len(chunk_ids) >= 2:
                stats = get_stats(chunk_ids)
                pair = min(stats, key = lambda p: self.merges.get(p, float('inf')))
                if pair not in self.merges:
                    break
                idx = self.merges[pair]
                chunk_ids = merge(chunk_ids, pair, idx)
            encoded_ids.extend(chunk_ids)

        return encoded_ids
    
if __name__ == "__main__":
    with open("tinystories_train.md", "r", encoding = "utf-8") as f:
        text = f.read()
        
    tokenizer = RegexTokenizer()
    tokenizer.train(text, vocab_size=4096, verbose=False)

    lines = []

    lines.append("\n# Full Vocab\n")
    for idx, token_bytes in tokenizer.vocab.items():
        token_str = token_bytes.decode('utf-8', errors='replace')
        if idx in tokenizer.merges.values():
            pair = next(p for p, i in tokenizer.merges.items() if i == idx)
            lines.append(f"{idx}: `{token_str}` (merged from {pair[0]} + {pair[1]})\n")
        else:
            lines.append(f"{idx}: `{token_str}`\n")

    with open("test.md", "w", encoding="utf-8") as f:
        f.writelines(lines)



        