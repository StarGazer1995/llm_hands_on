from tqdm import tqdm
from collections import defaultdict
import regex as re
# from types import List


class GPTTokenizer:
    def __init__(self, interations, vocab_size=30, mini_frequency = 2, base_idx=256, byte_fallback = True):
        self.interation = interations
        self.vocab_size = vocab_size
        self.merges = {}
        self.base_idx = base_idx
        self.mini_frequency = mini_frequency
        self.decode_vocab = None
        self.pattern = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+""")
        pass

    def string_encoding(self, string: str) :
        # print(list(string.encode("utf-8")))
        words = re.findall(self.pattern, string)
        encodes = []
        for word in words:
            encodes.extend(list(word.encode('utf-8')))
        return encodes
    
    def string_decoding(self, tokens) -> str:
        return bytes(tokens).decode("utf-8")
    
    def get_stats(self, ids):
        counts = defaultdict(int)
        for pair in zip(ids, ids[1:]):
            counts[pair] += 1
        return counts
    
    def merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i <= (len(ids) - 1):
            if ids[i] == pair[0] and ids[i + 1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids
    
    def add_all(self, base_idx):
        for pair, freq in self.stats.items():
            self.merges[pair] = base_idx
            base_idx += 1

    def encoding_data(self, data):
        token = self.string_encoding(data[0])
        while len(token) <= 2:
            stats = self.get_stats(token)
            mini_frequency_pair = min(stats, key=lambda p:self.merges.get(p, float('inf')))
            if mini_frequency_pair not in self.merges:
                break # nothing to be merged
            
            idx = self.merges[mini_frequency_pair]
            token = self.merge(token, mini_frequency_pair, idx)
        return token
    

    def decoding_data(self, ids):
        if self.decode_vocab is None:
            self.decode_vocab = {idx : bytes([idx]) for idx in range(256)}
            for (p0, p1), idx in self.merges.items():
                self.decode_vocab[idx] = self.decode_vocab[p0] + self.decode_vocab[p1]

        data = []
        for idx in ids:
            if self.decode_vocab.get(idx, None) is not None:
                data.append(self.decode_vocab.get(idx, None))
            else:
                data.append(idx.encode("utf-8"))
        data = b"".join(data)
        text = data.decode("utf-8", errors="replace")
        return text

    def train(self, data):
        tokens = list(data)
        self.vocab = []
        for t in tokens:
            self.vocab.extend(self.string_encoding(t))

        for i in range(self.interation):
            self.stats = self.get_stats(self.vocab)
            most_freq_pair = max(self.stats, key=self.stats.get)

            idx = i + self.base_idx #token index

            if self.stats[most_freq_pair] < self.mini_frequency: # fast end
                self.add_all(idx)
                break

            self.vocab = self.merge(self.vocab, most_freq_pair, idx)
            self.merges[most_freq_pair] = idx
            
if __name__ == "__main__":
    tokenizer = GPTTokenizer(30)
    tokenizer.train(["hello hello world"])
    print(tokenizer.encoding_data(["hello world"]))
    print(tokenizer.decoding_data(tokenizer.encoding_data(["hello world 你好"])))
    