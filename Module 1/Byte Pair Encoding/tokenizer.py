from collections import Counter
from collections import defaultdict

class BytePairEncoding:
    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.vocab = {}
        self.merges = []
        self.token_to_id = {"<unk>": 0}  
        self.id_to_token = {0: "<unk>"}

    def get_vocab(self, corpus):
        vocab = Counter()
        for word in corpus:
            # Add the end-of-word marker
            word = list(word) + ['</w>']
            vocab[tuple(word)] += 1
        return vocab

    def get_stats(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            for i in range(len(word) - 1):
                pairs[(word[i], word[i + 1])] += freq
        return pairs

    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        bigram = ''.join(pair)
        for word, freq in vocab.items():
            new_word = []
            i = 0
            while i < len(word):
                if i < len(word) - 1 and word[i] == pair[0] and word[i + 1] == pair[1]:
                    new_word.append(bigram)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            new_vocab[tuple(new_word)] = freq
        return new_vocab

    def fit(self, corpus):
        self.vocab = self.get_vocab(corpus)
        for _ in range(self.num_merges):
            pairs = self.get_stats(self.vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get) #type: ignore
            self.merges.append(best_pair)
            self.vocab = self.merge_vocab(best_pair, self.vocab)

        # Generate token-to-ID and ID-to-token mappings
        unique_tokens = set()
        for word in self.vocab.keys():
            unique_tokens.update(word)
        for idx, token in enumerate(sorted(unique_tokens), start=1):  # Start at 1 to keep <unk> at 0
            self.token_to_id[token] = idx
            self.id_to_token[idx] = token

    def encode(self, word):
        word = list(word) + ['</w>']
        for merge in self.merges:
            bigram = ''.join(merge)
            i = 0
            while i < len(word) - 1:
                if word[i] == merge[0] and word[i + 1] == merge[1]:
                    word[i:i + 2] = [bigram]
                else:
                    i += 1
        # Convert tokens to integers, falling back to <unk> for unknown characters
        return [self.token_to_id.get(token, self.token_to_id["<unk>"]) for token in word]

    def decode(self, tokens):
        # Convert integers back to tokens
        word = [self.id_to_token[token] for token in tokens]
        return ''.join(word).replace('</w>', '')
    




def get_stats(ids):
    counts = {}
    for pair in zip(ids, ids[1:]):
        counts[pair] = counts.get(pair, 0) + 1
    return counts

def merge(ids, pair, idx):
  newids = []
  i = 0
  while i < len(ids):
    if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
      newids.append(idx)
      i += 2
    else:
      newids.append(ids[i])
      i += 1
  return newids

# ---
vocab_size = 276 # the desired final vocabulary size
num_merges = vocab_size - 256
ids = list(tokens) # copy so we don't destroy the original list

merges = {} # (int, int) -> int
for i in range(num_merges):
  stats = get_stats(ids)
  pair = max(stats, key=stats.get)
  idx = 256 + i
  print(f"merging {pair} into a new token {idx}")
  ids = merge(ids, pair, idx)
  merges[pair] = idx


vocab = {idx: bytes([idx]) for idx in range(256)}
for (p0, p1), idx in merges.items():
    vocab[idx] = vocab[p0] + vocab[p1]

def decode(ids):
  # given ids (list of integers), return Python string
  tokens = b"".join(vocab[idx] for idx in ids)
  text = tokens.decode("utf-8", errors="replace")
  return text

def encode(text):
  # given a string, return list of integers (the tokens)
  tokens = list(text.encode("utf-8"))
  while len(tokens) >= 2:
    stats = get_stats(tokens)
    pair = min(stats, key=lambda p: merges.get(p, float("inf")))
    if pair not in merges:
      break # nothing else can be merged
    idx = merges[pair]
    tokens = merge(tokens, pair, idx)
  return tokens