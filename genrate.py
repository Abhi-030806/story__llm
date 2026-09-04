import numpy as np

from tokenizer import Tokenizer
from model import PhysicsLLM


# --------------------------------------------------
# Load corpus and tokenizer
# --------------------------------------------------

with open(
    "data/processed/corpus.txt",
    "r",
    encoding="utf-8"
) as f:
    text = f.read()

tokenizer = Tokenizer(text)

vocab_size = len(tokenizer.stoi)


# --------------------------------------------------
# Create model
# --------------------------------------------------

model = PhysicsLLM(
    vocab_size=vocab_size,
    embedding_dim=128,
    num_heads=4,
    hidden_dim=512,
    num_layers=2,
    context_length=32
)

model.load(
    "checkpoints/physics_llm.npz"
)


# --------------------------------------------------
# Generate text
# --------------------------------------------------

def generate(prompt, max_tokens=30):

    token_ids = tokenizer.encode(prompt)

    for _ in range(max_tokens):

        # Keep only the latest context
        context = token_ids[-32:]

        x = np.array(
            context,
            dtype=np.int64
        )[np.newaxis, :]

        # Shape:
        # (1, sequence)
        logits = model.forward(x)

        # Take prediction for final token
        next_logits = logits[0, -1]

        # Greedy prediction
        next_token = np.argmax(
            next_logits
        )

        token_ids.append(
            int(next_token)
        )

    return tokenizer.decode(token_ids)


# --------------------------------------------------
# Test
# --------------------------------------------------

prompt = "force is"

print()
print("Prompt:", prompt)
print()
print("Generated:")
print(generate(prompt, max_tokens=30))