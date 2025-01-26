from collections import Counter
from collections import defaultdict

class BytePairEncoding:
    def __init__(self, num_merges):
        self.num_merges = num_merges
        self.vocab = {}
        self.merges = []
        self.token_to_id = {"<unk>": 0}  # Initialize with <unk> token
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
            best_pair = max(pairs, key=pairs.get)
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