class Tokenizer:

    def __init__(self, text):
        # Split text into words
        self.tokens = text.split()

        # Get unique words
        vocab = sorted(set(self.tokens))

        # word → ID
        self.stoi = {word: i for i, word in enumerate(vocab)}

        # ID → word
        self.itos = {i: word for i, word in enumerate(vocab)}

    def encode(self, text):
        tokens = text.split()
        return [self.stoi[token] for token in tokens]

    def decode(self, ids):
        return " ".join(self.itos[i] for i in ids)

with open("data/processed/corpus.txt", "r", encoding="utf-8") as f:
    text = f.read()

tokenizer = Tokenizer(text)

print("Vocabulary size:", len(tokenizer.stoi))

sample = "motion is the change"

encoded = tokenizer.encode(sample)

print("Encoded:", encoded)
print("Decoded:", tokenizer.decode(encoded))