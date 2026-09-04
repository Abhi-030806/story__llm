import numpy as np


class FeedForward:

    def __init__(self, embedding_dim, hidden_dim):

        self.W1 = (
            np.random.randn(
                embedding_dim,
                hidden_dim
            ) * 0.02
        )

        self.b1 = np.zeros(hidden_dim)

        self.W2 = (
            np.random.randn(
                hidden_dim,
                embedding_dim
            ) * 0.02
        )

        self.b2 = np.zeros(embedding_dim)

        self.X = None
        self.Z = None
        self.H = None

    def forward(self, X):

        self.X = X

        self.Z = X @ self.W1 + self.b1

        self.H = np.maximum(
            0,
            self.Z
        )

        output = (
            self.H @ self.W2
            + self.b2
        )

        return output

    def backward(self, d_output):

        # Flatten batch and sequence dimensions
        X_flat = self.X.reshape(
            -1,
            self.X.shape[-1]
        )

        H_flat = self.H.reshape(
            -1,
            self.H.shape[-1]
        )

        d_output_flat = d_output.reshape(
            -1,
            d_output.shape[-1]
        )

        # W2
        dW2 = H_flat.T @ d_output_flat

        # b2
        db2 = np.sum(
            d_output_flat,
            axis=0
        )

        # Gradient through W2
        dH = d_output @ self.W2.T

        # ReLU
        dZ = dH * (self.Z > 0)

        # W1
        dZ_flat = dZ.reshape(
            -1,
            dZ.shape[-1]
        )

        dW1 = X_flat.T @ dZ_flat

        # b1
        db1 = np.sum(
            dZ_flat,
            axis=0
        )

        # Gradient back to X
        dX = dZ @ self.W1.T

        return dX, dW1, db1, dW2, db2


if __name__ == "__main__":

    np.random.seed(42)

    ffn = FeedForward(
        embedding_dim=128,
        hidden_dim=512
    )

    X = np.random.randn(
        4, 8, 128
    )

    output = ffn.forward(X)

    print("Input:", X.shape)
    print("Output:", output.shape)

    d_output = np.random.randn(
        4, 8, 128
    )

    gradients = ffn.backward(
        d_output
    )

    print("dX:", gradients[0].shape)
    print("dW1:", gradients[1].shape)
    print("db1:", gradients[2].shape)
    print("dW2:", gradients[3].shape)
    print("db2:", gradients[4].shape)