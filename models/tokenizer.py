from collections import defaultdict
import re

class BPE:
    def __init__(self, num_merges=30000, min_frequency=2):
        self.num_merges = num_merges
        self.min_frequency = min_frequency
        self.final_vacab = set()

    
    def get_stat(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i + 1]] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        new_voceb = dict()
        bigram = ' '.join(pair)
        replacement = ''.join(pair)
        for word in vocab:
            new_word = word.replace(bigram, replacement)
            new_voceb[new_word] = vocab[word]
        return new_voceb
    
    def build_vocab(self, text):
        '''build vocabulary from the input text'''
        vocab = defaultdict(int)
        tokens = []
        for token in re.findall(r"[\u4e00-\u9fff]+|[a-zA-Z']+|\d+\.\d+|\d+|\S", text):
            if re.match(r"[\u4e00-\u9fff]+", token):
                tokens.extend(list(token)) #chinese
            elif re.match(r"[a-zA-Z']+", token):
                tokens.append(token) #english
            else:
                tokens.append(token) #other
        for token in tokens:
            vocab[token] += 1
        return vocab
    
    def bpe_tokenize(self, text):
        '''perform BPE algorithm on the input text'''
        vocab = defaultdict(int)
        for word in text.split():
            # Add spaces between characters to represent them as tokens
            tokenized_word = ' '.join(list(word))
            vocab[tokenized_word] += 1

        for i in range(self.num_merges):
            pairs = self.get_stat(vocab)
            if not pairs:
                break
            best_pair = max(pairs, key=pairs.get)
            if pairs[best_pair] < self.min_frequency:
                break
            vocab = self.merge_vocab(best_pair, vocab)

        self.final_vacab = set()
        for word in vocab:
            for token in word.split():
                self.final_vacab.add(token)
    
    def post_process(self, text):
        # Tokenize the input text using the final vocabulary
        tokenized_text = []
        for word in text.split():
            tokenized_word = ' '.join(list(word))
            for token in self.final_vacab:
                tokenized_word = tokenized_word.replace(' '.join(list(token)), token)
            tokenized_text.append(tokenized_word)
        return tokenized_text
    
    def train(self, texts):
        self.bpe_tokenize(texts)

    def tokenize(self, text):
        return self.post_process(text)
    
class BPEByteLevel(BPE):

    def build_vocab(self, text):
        vocab = defaultdict(int)
        for sentence in text:
            for word in sentence.strip().split():
                s_word = str.join(' ', list(word))
                vocab[s_word] += 1
        return vocab
    
    def get_stat(self, vocab):
        pairs = defaultdict(int)
        for word, freq in vocab.items():
            symbols = word.split()
            for i in range(len(symbols) - 1):
                pairs[symbols[i], symbols[i+1]] += freq
        return pairs
    
    def merge_vocab(self, pair, vocab):
        new_vocab = {}
        first, second = pair
        replace_ment = first + second # concatenate two bytes
        for word in vocab:
            new_word = word.replace(first + " " + second, replace_ment)
            new_vocab[new_word] = vocab[word]
        return new_vocab

    
    def train(self, byte_data):
        self.vocab = self.build_vocab(byte_data)
        self.pair = None
        for i in range(self.num_merges):
            pair = self.get_stat(self.vocab)
            if not pair:
                break
            self.pair = pair
            most_freq_pair = max(pair, key=pair.get)
            if pair.get(most_freq_pair) < self.min_frequency:
                break
            self.vocab = self.merge_vocab(most_freq_pair, self.vocab)
        self.final_vacab = set()
        for word in self.vocab:
            self.final_vacab.add(word)

    def tokenize(self, text):
        tokenized_data = []
        for byte_data in text:
            tokenized_seq = byte_data
            for token in self.final_vacab:
                replace_token = token
                replacement = " {} ".format(token)
                tokenized_seq = tokenized_seq.replace(replace_token, replacement)
            tokenized_data.append(tokenized_seq)
        return tokenized_data
    
if __name__ == "__main__":
    import unittest

    class TestBPEByteLevel(unittest.TestCase):
        def test_bpe_byte_level(self):
            # Initialize the BPEByteLevel tokenizer
            tokenizer = BPEByteLevel(100)
            
            # Define the training data
            train_text = [
                        "hello",
                        "world", 
                        "hello world", 
                        "这里是中文测试", 
                        "这里", 
                        "中文", 
                        "123", 
                        "12", 
                        "3", 
                        "low", 
                        "lower", 
                        "lowest",
                        "big",
                        "bigger",
                        "biggest",
                        "estimate",
                        "charger",
                        "charge",
                        "flower",
                        "4",
                        "3.14",
                        "5.12"
                        ]
            
            # Train the tokenizer
            tokenizer.train(train_text)
            
            # Define the test data
            test_text = [
                        "hello", 
                        "world",
                        "hello world", 
                        "hello hello world", 
                        "这里是中文测试", 
                        "123",
                        "lowest"]
            
            # Tokenize the test data
            tokenized_text = tokenizer.tokenize(test_text)
            
            decoded_text = [t.split() for t in tokenized_text]
            
            expected_output = ["hello".split(), 
                                "world".split(),
                                "hello world".split(), 
                                "hello hello world".split(), 
                                "这里 是 中文 测试".split(), 
                                "12 3".split(),
                                "low est".split()]
            self.assertEqual(decoded_text, expected_output)

    unittest.main()
