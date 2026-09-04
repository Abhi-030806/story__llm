import numpy as np
from transformer_block import TransformerBlock


class Transformer:

    def __init__(
        self,
        embedding_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_layers=2
    ):

        self.blocks = [
            TransformerBlock(
                embedding_dim,
                num_heads,
                hidden_dim
            )
            for _ in range(num_layers)
        ]

    def forward(self, X):

        for block in self.blocks:
            X = block.forward(X)

        return X


if __name__ == "__main__":

    model = Transformer(
        embedding_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_layers=2
    )

    X = np.random.randn(8, 128)

    output = model.forward(X)

    print("Input shape:", X.shape)
    print("Output shape:", output.shape)