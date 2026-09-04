import numpy as np


class Embedding:

    def __init__(self, vocab_size, embedding_dim):

        self.weights = (
            np.random.randn(
                vocab_size,
                embedding_dim
            ) * 0.02
        )

        self.token_ids = None

    def forward(self, token_ids):

        self.token_ids = token_ids

        return self.weights[token_ids]

    def backward(self, d_output):

        d_weights = np.zeros_like(self.weights)

        # Flatten all token positions
        flat_ids = self.token_ids.reshape(-1)
        flat_gradients = d_output.reshape(
            -1,
            d_output.shape[-1]
        )

        for token_id, gradient in zip(
            flat_ids,
            flat_gradients
        ):
            d_weights[token_id] += gradient

        return d_weights


if __name__ == "__main__":

    embedding = Embedding(
        vocab_size=1564,
        embedding_dim=128
    )

    # Batch of 4 sequences, each 8 tokens
    token_ids = np.random.randint(
        0,
        1564,
        size=(4, 8)
    )

    output = embedding.forward(token_ids)

    print("Input:", token_ids.shape)
    print("Output:", output.shape)

    d_output = np.random.randn(
        4, 8, 128
    )

    d_weights = embedding.backward(
        d_output
    )

    print("Gradient:", d_weights.shape)