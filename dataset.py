import numpy as np
from tokenizer import Tokenizer


def create_dataset(token_ids, context_length):
    X = []
    Y = []

    for i in range(len(token_ids) - context_length):
        X.append(token_ids[i:i + context_length])
        Y.append(token_ids[i + 1:i + context_length + 1])

    return np.array(X), np.array(Y)


if __name__ == "__main__":

    with open("data/processed/corpus.txt", "r", encoding="utf-8") as f:
        text = f.read()

    tokenizer = Tokenizer(text)

    token_ids = tokenizer.encode(text)

    context_length = 8

    X, Y = create_dataset(token_ids, context_length)

    print("Vocabulary size:", len(tokenizer.stoi))
    print("Total tokens:", len(token_ids))
    print("X shape:", X.shape)
    print("Y shape:", Y.shape)

    print("\nFirst input:")
    print(X[0])

    print("\nFirst target:")
    print(Y[0])

    print("\nDecoded input:")
    print(tokenizer.decode(X[0]))

    print("\nDecoded target:")
    print(tokenizer.decode(Y[0]))