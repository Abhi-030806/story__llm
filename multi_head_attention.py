import numpy as np


class MultiHeadAttention:

    def __init__(self, embedding_dim, num_heads):

        assert embedding_dim % num_heads == 0

        self.embedding_dim = embedding_dim
        self.num_heads = num_heads
        self.head_dim = embedding_dim // num_heads

        self.W_Q = np.random.randn(
            embedding_dim, embedding_dim
        ) * 0.02

        self.W_K = np.random.randn(
            embedding_dim, embedding_dim
        ) * 0.02

        self.W_V = np.random.randn(
            embedding_dim, embedding_dim
        ) * 0.02

        self.W_O = np.random.randn(
            embedding_dim, embedding_dim
        ) * 0.02

        self.X = None
        self.Q = None
        self.K = None
        self.V = None
        self.weights = None
        self.combined = None

    def forward(self, X):

        self.X = X

        # X: (batch, seq, embedding)
        Q = X @ self.W_Q
        K = X @ self.W_K
        V = X @ self.W_V

        batch_size, seq_len, _ = X.shape

        # (batch, seq, heads, head_dim)
        Q = Q.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        K = K.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        V = V.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        # (batch, heads, seq, head_dim)
        Q = Q.transpose(0, 2, 1, 3)
        K = K.transpose(0, 2, 1, 3)
        V = V.transpose(0, 2, 1, 3)

        self.Q = Q
        self.K = K
        self.V = V

        # Attention scores
        scores = (
            Q @ K.transpose(0, 1, 3, 2)
        ) / np.sqrt(self.head_dim)

        # Causal mask
        mask = np.triu(
            np.ones((seq_len, seq_len)),
            k=1
        )

        scores = np.where(
            mask == 1,
            -1e9,
            scores
        )

        # Softmax
        scores = scores - np.max(
            scores,
            axis=-1,
            keepdims=True
        )

        exp_scores = np.exp(scores)

        attention = exp_scores / np.sum(
            exp_scores,
            axis=-1,
            keepdims=True
        )

        self.weights = attention

        # Attention × V
        output = attention @ V

        # (batch, heads, seq, head_dim)
        # → (batch, seq, heads, head_dim)
        output = output.transpose(0, 2, 1, 3)

        # Combine heads
        self.combined = output.reshape(
            batch_size,
            seq_len,
            self.embedding_dim
        )

        return self.combined @ self.W_O

    def backward(self, d_output):

        batch_size, seq_len, _ = self.X.shape

        # Output projection
        d_combined = d_output @ self.W_O.T

        dW_O = (
            self.combined.reshape(-1, self.embedding_dim).T
            @ d_output.reshape(-1, self.embedding_dim)
        )

        d_combined = d_combined.reshape(
            batch_size,
            seq_len,
            self.num_heads,
            self.head_dim
        )

        d_combined = d_combined.transpose(
            0, 2, 1, 3
        )

        dX = np.zeros_like(self.X)

        dW_Q = np.zeros_like(self.W_Q)
        dW_K = np.zeros_like(self.W_K)
        dW_V = np.zeros_like(self.W_V)

        scale = np.sqrt(self.head_dim)

        for h in range(self.num_heads):

            q = self.Q[:, h]
            k = self.K[:, h]
            v = self.V[:, h]

            attention = self.weights[:, h]

            dO = d_combined[:, h]

            # O = A @ V
            dA = dO @ v.transpose(0, 2, 1)

            dV = attention.transpose(
                0, 2, 1
            ) @ dO

            # Softmax backward
            dS = attention * (
                dA
                - np.sum(
                    dA * attention,
                    axis=-1,
                    keepdims=True
                )
            )

            # S = QK^T / sqrt(d)
            dQ = (
                dS @ k
            ) / scale

            dK = (
                dS.transpose(0, 2, 1) @ q
            ) / scale

            start = h * self.head_dim
            end = (h + 1) * self.head_dim

            # Q = XWQ
            self_X = self.X.reshape(
                -1,
                self.embedding_dim
            )

            dQ_flat = dQ.reshape(
                -1,
                self.head_dim
            )

            dK_flat = dK.reshape(
                -1,
                self.head_dim
            )

            dV_flat = dV.reshape(
                -1,
                self.head_dim
            )

            dW_Q[:, start:end] += (
                self_X.T @ dQ_flat
            )

            dW_K[:, start:end] += (
                self_X.T @ dK_flat
            )

            dW_V[:, start:end] += (
                self_X.T @ dV_flat
            )

            dX += (
                dQ @ self.W_Q[:, start:end].T
                + dK @ self.W_K[:, start:end].T
                + dV @ self.W_V[:, start:end].T
            )

        return (
            dX,
            dW_Q,
            dW_K,
            dW_V,
            dW_O
        )


if __name__ == "__main__":

    np.random.seed(42)

    attention = MultiHeadAttention(
        embedding_dim=128,
        num_heads=4
    )

    X = np.random.randn(
        4, 8, 128
    )

    output = attention.forward(X)

    print("Input:", X.shape)
    print("Output:", output.shape)

    d_output = np.random.randn(
        4, 8, 128
    )

    gradients = attention.backward(d_output)

    print("dX:", gradients[0].shape)
    print("dW_Q:", gradients[1].shape)
    print("dW_O:", gradients[4].shape)