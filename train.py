import numpy as np

from tokenizer import Tokenizer
from dataset import create_dataset
from model import PhysicsLLM
from loss import cross_entropy_loss, cross_entropy_backward
from adam import Adam


# --------------------------------------------------
# Update model parameters
# --------------------------------------------------

def update_model(model, gradients, optimizer):

    optimizer.step()

    # Output layer
    optimizer.update(
        "output_W",
        model.output_layer.W,
        gradients["output_dW"]
    )

    optimizer.update(
        "output_b",
        model.output_layer.b,
        gradients["output_db"]
    )

    # Transformer blocks
    for i, (block, grads) in enumerate(
        zip(
            model.blocks,
            reversed(gradients["blocks"])
        )
    ):

        optimizer.update(
            f"block{i}_WQ",
            block.attention.W_Q,
            grads["dW_Q"]
        )

        optimizer.update(
            f"block{i}_WK",
            block.attention.W_K,
            grads["dW_K"]
        )

        optimizer.update(
            f"block{i}_WV",
            block.attention.W_V,
            grads["dW_V"]
        )

        optimizer.update(
            f"block{i}_WO",
            block.attention.W_O,
            grads["dW_O"]
        )

        optimizer.update(
            f"block{i}_W1",
            block.ffn.W1,
            grads["dW1"]
        )

        optimizer.update(
            f"block{i}_b1",
            block.ffn.b1,
            grads["db1"]
        )

        optimizer.update(
            f"block{i}_W2",
            block.ffn.W2,
            grads["dW2"]
        )

        optimizer.update(
            f"block{i}_b2",
            block.ffn.b2,
            grads["db2"]
        )

        optimizer.update(
            f"block{i}_gamma1",
            block.norm1.gamma,
            grads["dgamma1"]
        )

        optimizer.update(
            f"block{i}_beta1",
            block.norm1.beta,
            grads["dbeta1"]
        )

        optimizer.update(
            f"block{i}_gamma2",
            block.norm2.gamma,
            grads["dgamma2"]
        )

        optimizer.update(
            f"block{i}_beta2",
            block.norm2.beta,
            grads["dbeta2"]
        )

    # Embedding
    optimizer.update(
        "embedding",
        model.embedding.weights,
        gradients["embedding"]
    )


# --------------------------------------------------
# Load corpus
# --------------------------------------------------

with open(
    "data/processed/corpus.txt",
    "r",
    encoding="utf-8"
) as f:

    text = f.read()


# --------------------------------------------------
# Tokenization
# --------------------------------------------------

tokenizer = Tokenizer(text)

token_ids = tokenizer.encode(text)

vocab_size = len(tokenizer.stoi)

print("Vocabulary size:", vocab_size)


# --------------------------------------------------
# Dataset
# --------------------------------------------------

context_length = 32

X, Y = create_dataset(
    token_ids,
    context_length
)

print("Dataset shape:", X.shape)


# --------------------------------------------------
# Model
# --------------------------------------------------

model = PhysicsLLM(
    vocab_size=vocab_size,
    embedding_dim=128,
    num_heads=4,
    hidden_dim=512,
    num_layers=2,
    context_length=context_length
)


# --------------------------------------------------
# Optimizer
# --------------------------------------------------

optimizer = Adam(
    learning_rate=0.001
)


# --------------------------------------------------
# Training settings
# --------------------------------------------------

epochs = 10
batch_size = 32


# --------------------------------------------------
# Training loop
# --------------------------------------------------

for epoch in range(epochs):

    # Shuffle dataset
    indices = np.random.permutation(
        len(X)
    )

    X_shuffled = X[indices]
    Y_shuffled = Y[indices]

    total_loss = 0.0
    num_batches = 0

    for start in range(
        0,
        len(X),
        batch_size
    ):

        end = min(
            start + batch_size,
            len(X)
        )

        x_batch = X_shuffled[
            start:end
        ]

        y_batch = Y_shuffled[
            start:end
        ]

        # Forward
        logits = model.forward(
            x_batch
        )

        # Loss
        loss = cross_entropy_loss(
            logits,
            y_batch
        )

        # Backward
        d_logits = cross_entropy_backward(
            logits,
            y_batch
        )

        gradients = model.backward(
            d_logits
        )

        # Update
        update_model(
            model,
            gradients,
            optimizer
        )

        total_loss += loss
        num_batches += 1

    average_loss = (
        total_loss / num_batches
    )

    print(
        f"Epoch {epoch + 1}/{epochs} "
        f"Loss: {average_loss:.4f}"
    )

model.save("checkpoints/physics_llm.npz")

print("Model saved!")