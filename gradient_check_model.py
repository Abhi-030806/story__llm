import numpy as np

from tokenizer import Tokenizer
from model import PhysicsLLM
from loss import cross_entropy_loss, cross_entropy_backward


np.random.seed(42)

# Small model for gradient checking
vocab_size = 10
context_length = 4

model = PhysicsLLM(
    vocab_size=vocab_size,
    embedding_dim=4,
    num_heads=2,
    hidden_dim=8,
    num_layers=1,
    context_length=context_length
)

token_ids = np.array([1, 2, 3, 4])
targets = np.array([2, 3, 4, 5])


# -------------------------
# Analytical gradient
# -------------------------

logits = model.forward(token_ids)

loss = cross_entropy_loss(
    logits,
    targets
)

d_logits = cross_entropy_backward(
    logits,
    targets
)

gradients = model.backward(d_logits)


# -------------------------
# Numerical gradient
# -------------------------

epsilon = 1e-5

row = 1
col = 2

original = model.output_layer.W[row, col]


model.output_layer.W[row, col] = original + epsilon

loss_plus = cross_entropy_loss(
    model.forward(token_ids),
    targets
)


model.output_layer.W[row, col] = original - epsilon

loss_minus = cross_entropy_loss(
    model.forward(token_ids),
    targets
)


model.output_layer.W[row, col] = original


numerical_gradient = (
    loss_plus - loss_minus
) / (2 * epsilon)

analytical_gradient = gradients[
    "output_dW"
][row, col]

difference = abs(
    numerical_gradient
    - analytical_gradient
)


print("Loss:", loss)
print("Numerical gradient:", numerical_gradient)
print("Analytical gradient:", analytical_gradient)
print("Difference:", difference)