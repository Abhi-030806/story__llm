import numpy as np

from embedding import Embedding
from positional_encoding import positional_encoding
from transformer_block import TransformerBlock
from output_layer import OutputLayer


class PhysicsLLM:

    def __init__(
        self,
        vocab_size,
        embedding_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_layers=2,
        context_length=32
    ):

        self.embedding = Embedding(
            vocab_size,
            embedding_dim
        )

        self.position_encoding = positional_encoding(
            context_length,
            embedding_dim
        )

        self.blocks = [
            TransformerBlock(
                embedding_dim,
                num_heads,
                hidden_dim
            )
            for _ in range(num_layers)
        ]

        self.output_layer = OutputLayer(
            embedding_dim,
            vocab_size
        )
    def load(self, path):

        parameters = np.load(path)

        self.embedding.weights = parameters["embedding"]
        self.output_layer.W = parameters["output_W"]
        self.output_layer.b = parameters["output_b"]

        for i, block in enumerate(self.blocks):

            block.attention.W_Q = parameters[f"block{i}_WQ"]
            block.attention.W_K = parameters[f"block{i}_WK"]
            block.attention.W_V = parameters[f"block{i}_WV"]
            block.attention.W_O = parameters[f"block{i}_WO"]

            block.ffn.W1 = parameters[f"block{i}_W1"]
            block.ffn.b1 = parameters[f"block{i}_b1"]
            block.ffn.W2 = parameters[f"block{i}_W2"]
            block.ffn.b2 = parameters[f"block{i}_b2"]

            block.norm1.gamma = parameters[f"block{i}_gamma1"]
            block.norm1.beta = parameters[f"block{i}_beta1"]

            block.norm2.gamma = parameters[f"block{i}_gamma2"]
            block.norm2.beta = parameters[f"block{i}_beta2"]

        print("Model loaded!")

    def forward(self, token_ids):

        # (batch, seq)
        X = self.embedding.forward(token_ids)

        # (seq, embedding)
        position = self.position_encoding[
            :token_ids.shape[1]
        ]

        # Broadcasting:
        # (batch, seq, embedding)
        X = X + position

        for block in self.blocks:
            X = block.forward(X)

        logits = self.output_layer.forward(X)

        return logits

    def save(self, path):

        parameters = {
            "embedding": self.embedding.weights,
            "output_W": self.output_layer.W,
            "output_b": self.output_layer.b,
        }

        for i, block in enumerate(self.blocks):

            parameters[f"block{i}_WQ"] = block.attention.W_Q
            parameters[f"block{i}_WK"] = block.attention.W_K
            parameters[f"block{i}_WV"] = block.attention.W_V
            parameters[f"block{i}_WO"] = block.attention.W_O

            parameters[f"block{i}_W1"] = block.ffn.W1
            parameters[f"block{i}_b1"] = block.ffn.b1
            parameters[f"block{i}_W2"] = block.ffn.W2
            parameters[f"block{i}_b2"] = block.ffn.b2

            parameters[f"block{i}_gamma1"] = block.norm1.gamma
            parameters[f"block{i}_beta1"] = block.norm1.beta
            parameters[f"block{i}_gamma2"] = block.norm2.gamma
            parameters[f"block{i}_beta2"] = block.norm2.beta

        np.savez(path, **parameters)

    def backward(self, d_logits):

        dX, dW, db = self.output_layer.backward(
            d_logits
        )

        gradients = {
            "output_dW": dW,
            "output_db": db,
            "blocks": []
        }

        for block in reversed(self.blocks):

            block_grads = block.backward(dX)

            gradients["blocks"].append(
                block_grads
            )

            dX = block_grads["dX"]

        gradients["embedding"] = (
            self.embedding.backward(dX)
        )

        return gradients


if __name__ == "__main__":

    np.random.seed(42)

    model = PhysicsLLM(
        vocab_size=1564,
        embedding_dim=128,
        num_heads=4,
        hidden_dim=512,
        num_layers=2,
        context_length=32
    )

    # 4 sequences × 32 tokens
    token_ids = np.random.randint(
        0,
        1564,
        size=(4, 32)
    )

    logits = model.forward(token_ids)

    print("Input:", token_ids.shape)
    print("Logits:", logits.shape)

    d_logits = np.random.randn(
        4, 32, 1564
    )

    gradients = model.backward(d_logits)

    print("Embedding gradient:",
          gradients["embedding"].shape)

    print("Output gradient:",
          gradients["output_dW"].shape)

    print("Number of block gradients:",
          len(gradients["blocks"]))

    print("Block dX:",
          gradients["blocks"][0]["dX"].shape)