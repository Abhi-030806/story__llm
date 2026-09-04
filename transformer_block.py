import numpy as np
from multi_head_attention import MultiHeadAttention
from feed_forward import FeedForward


class LayerNorm:

    def __init__(self, dim, eps=1e-5):
        self.gamma = np.ones(dim)
        self.beta = np.zeros(dim)
        self.eps = eps

        self.X = None
        self.mean = None
        self.var = None
        self.normalized = None

    def forward(self, X):

        self.X = X

        self.mean = np.mean(
            X,
            axis=-1,
            keepdims=True
        )

        self.var = np.var(
            X,
            axis=-1,
            keepdims=True
        )

        self.normalized = (
            X - self.mean
        ) / np.sqrt(
            self.var + self.eps
        )

        return (
            self.gamma * self.normalized
            + self.beta
        )

    def backward(self, d_output):

        # Sum over batch AND sequence
        dgamma = np.sum(
            d_output * self.normalized,
            axis=(0, 1)
        )

        dbeta = np.sum(
            d_output,
            axis=(0, 1)
        )

        N = self.X.shape[-1]

        std_inv = 1 / np.sqrt(
            self.var + self.eps
        )

        dX = (
            self.gamma
            * std_inv
            / N
            * (
                N * d_output
                - np.sum(
                    d_output,
                    axis=-1,
                    keepdims=True
                )
                - self.normalized
                * np.sum(
                    d_output * self.normalized,
                    axis=-1,
                    keepdims=True
                )
            )
        )

        return dX, dgamma, dbeta

    
class TransformerBlock:

    def __init__(
        self,
        embedding_dim=128,
        num_heads=4,
        hidden_dim=512
    ):

        self.norm1 = LayerNorm(embedding_dim)

        self.attention = MultiHeadAttention(
            embedding_dim,
            num_heads
        )

        self.norm2 = LayerNorm(embedding_dim)

        self.ffn = FeedForward(
            embedding_dim,
            hidden_dim
        )

    def forward(self, X):

        # Pre-norm
        norm1_output = self.norm1.forward(X)

        # Self-attention
        attention_output = self.attention.forward(
            norm1_output
        )

        # First residual connection
        self.after_attention = (
            X + attention_output
        )

        # Second normalization
        norm2_output = self.norm2.forward(
            self.after_attention
        )

        # Feed-forward network
        ffn_output = self.ffn.forward(
            norm2_output
        )

        # Second residual connection
        output = (
            self.after_attention + ffn_output
        )

        return output

    def backward(self, d_output):

        # FFN backward
        d_ffn_input, dW1, db1, dW2, db2 = (
            self.ffn.backward(d_output)
        )

        # Residual path
        d_after_attention = (
            d_output + d_ffn_input
        )

        # LayerNorm 2
        d_norm2, dgamma2, dbeta2 = (
            self.norm2.backward(d_ffn_input)
        )

        d_after_attention += d_norm2

        # Attention backward
        (
            d_norm1,
            dW_Q,
            dW_K,
            dW_V,
            dW_O
        ) = self.attention.backward(
            d_after_attention
        )

        # LayerNorm 1
        dX_norm1, dgamma1, dbeta1 = (
            self.norm1.backward(d_norm1)
        )

        # First residual path
        dX = (
            d_after_attention + dX_norm1
        )

        return {
            "dX": dX,

            "dW_Q": dW_Q,
            "dW_K": dW_K,
            "dW_V": dW_V,
            "dW_O": dW_O,

            "dW1": dW1,
            "db1": db1,
            "dW2": dW2,
            "db2": db2,

            "dgamma1": dgamma1,
            "dbeta1": dbeta1,
            "dgamma2": dgamma2,
            "dbeta2": dbeta2
        }

if __name__ == "__main__":

    np.random.seed(42)

    block = TransformerBlock(
        embedding_dim=128,
        num_heads=4,
        hidden_dim=512
    )

    X = np.random.randn(
        4, 8, 128
    )

    output = block.forward(X)

    print("Input:", X.shape)
    print("Output:", output.shape)

    d_output = np.random.randn(
        4, 8, 128
    )

    gradients = block.backward(d_output)

    print("dX:", gradients["dX"].shape)
    print("dW_Q:", gradients["dW_Q"].shape)
    print("dW1:", gradients["dW1"].shape)
    print("dgamma1:", gradients["dgamma1"].shape)