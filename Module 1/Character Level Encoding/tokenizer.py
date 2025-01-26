from typing import List

class CharacterLevelEncoding:
    def __init__(self) -> None:
        self.chars = None
        self.stoi = None
        self.itoi = None
    
    def train(self, text):
        self.chars = sorted(list(set(text)))
        self.stoi = {ch:i for i, ch in enumerate(self.chars)}
        self.itoi = {i:ch for i, ch in enumerate(self.chars)}

    def encode(self, text) -> List[int]:
        tokens = []

        for ch in text:
            tokens.append(self.stoi[ch])
        return tokens
    
    def decode(self, tokens)-> str:
        
        tokens_back = []

        for token in tokens:
            tokens_back.append(self.itoi[token])
        return "".join(tokens_back)
    
    def n_vocab(self):
        return len(self.chars)
    