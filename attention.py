import numpy as np


def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(x)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)


class SelfAttention:

    def __init__(self, embedding_dim, head_dim):
        self.W_Q = np.random.randn(embedding_dim, head_dim) * 0.02
        self.W_K = np.random.randn(embedding_dim, head_dim) * 0.02
        self.W_V = np.random.randn(embedding_dim, head_dim) * 0.02

    def forward(self, X):

        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        scores = (Q @ K.T) / np.sqrt(K.shape[-1])

        # Causal mask: don't look at future tokens
        seq_len = X.shape[0]
        mask = np.triu(
            np.ones((seq_len, seq_len)),
            k=1
        )

        scores = np.where(mask == 1, -1e9, scores)

        attention_weights = softmax(scores)

        output = attention_weights @ V

        return output


if __name__ == "__main__":

    attention = SelfAttention(
        embedding_dim=128,
        head_dim=32
    )

    X = np.random.randn(8, 128)

    output = attention.forward(X)

    print("Input shape:", X.shape)
    print("Output shape:", output.shape)