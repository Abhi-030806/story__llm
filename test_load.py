from model import PhysicsLLM

model = PhysicsLLM(
    vocab_size=1564,
    embedding_dim=128,
    num_heads=4,
    hidden_dim=512,
    num_layers=2,
    context_length=32
)

model.load(
    "checkpoints/physics_llm.npz"
)

print("Embedding shape:", model.embedding.weights.shape)
print("Output shape:", model.output_layer.W.shape)