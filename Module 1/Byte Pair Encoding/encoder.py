from typing import List,Dict, Tuple
from collections import Counter

class BytePairEncoding:
    def __init__(self, vocab_size: int, num_merges : int =20) -> None:
        self.merges = {}
        self.vocab_size = vocab_size
        self.num_merges = num_merges
        self.vocab = {idx: bytes([idx]) for idx in range(vocab_size-num_merges)}

    def get_stats(self, ids: List[int]) -> Dict[Tuple[int, int], int]:
        count = {}
        
        for pair in zip(ids, ids[1:]):
            count[pair] = count.get(pair,0)+1
        
        return count

    def merge(self, ids, pair, idx) -> List[int]:
        
        new_ids = []

        i = 0
        while i < len(ids):
            if i < len(ids) -1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                new_ids.append(idx)
                i+=2
            else:
                new_ids.append(ids[i])
                i+=1
        return new_ids
    
    
    def fit(self, corpus : str):
        
        tokens = corpus.encode("utf-8")
        ids = list(map(int, tokens))
        
        for i in range(self.num_merges):
            stats = self.get_stats(ids)
            pair = max(stats, key=stats.get) # type: ignore
            idx = (self.vocab_size-self.num_merges)+i
            ids = self.merge(ids, pair, idx)
            self.merges[pair] = idx

        for (p0, p1), idx in self.merges.items():
            self.vocab[idx] = self.vocab[p0]+self.vocab[p1]
   
    
    def encode(self, text: str) -> List[int]:

        tokens = list(text.encode("utf-8"))

        while len(tokens) >= 2:
            stats = self.get_stats(tokens)

            pair = min(stats, key=lambda p: self.merges.get(p, float("inf")))

            if pair not in self.merges:
                break
            idx = self.merges[pair]
            tokens = self.merge(tokens, pair, idx)
        return tokens 

    def decode(self,ids: List[int]) -> str:
        tokens = b"".join(self.vocab[idx] for idx in ids)
        text = tokens.decode("utf-8", errors="replace")

        return text


if __name__ == "__main__":

    with open("output.txt", "r") as file:
        corpus = file.read()
    
    bpe = BytePairEncoding(276)
    bpe.fit(corpus)
    print(bpe.encode("hello world"))
    print(bpe.decode(bpe.encode("hello world")))
