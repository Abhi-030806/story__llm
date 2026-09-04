import numpy as np


class OutputLayer:

    def __init__(self, embedding_dim, vocab_size):

        self.W = (
            np.random.randn(
                embedding_dim,
                vocab_size
            ) * 0.02
        )

        self.b = np.zeros(vocab_size)
        self.X = None

    def forward(self, X):

        self.X = X

        return X @ self.W + self.b

    def backward(self, d_logits):

        X_flat = self.X.reshape(
            -1,
            self.X.shape[-1]
        )

        d_logits_flat = d_logits.reshape(
            -1,
            d_logits.shape[-1]
        )

        # Weight gradient
        dW = X_flat.T @ d_logits_flat

        # Bias gradient
        db = np.sum(
            d_logits_flat,
            axis=0
        )

        # Gradient back to X
        dX = d_logits @ self.W.T

        return dX, dW, db


if __name__ == "__main__":

    np.random.seed(42)

    layer = OutputLayer(
        embedding_dim=128,
        vocab_size=1564
    )

    X = np.random.randn(
        4, 8, 128
    )

    logits = layer.forward(X)

    print("Input:", X.shape)
    print("Logits:", logits.shape)

    d_logits = np.random.randn(
        4, 8, 1564
    )

    dX, dW, db = layer.backward(
        d_logits
    )

    print("dX:", dX.shape)
    print("dW:", dW.shape)
    print("db:", db.shape)