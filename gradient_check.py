import numpy as np

from output_layer import OutputLayer
from loss import cross_entropy_loss, cross_entropy_backward


np.random.seed(42)

vocab_size = 10
embedding_dim = 4

layer = OutputLayer(
    embedding_dim,
    vocab_size
)

X = np.random.randn(3, embedding_dim)
targets = np.array([2, 5, 1])


# -------------------------
# Analytical gradient
# -------------------------

logits = layer.forward(X)

loss = cross_entropy_loss(
    logits,
    targets
)

d_logits = cross_entropy_backward(
    logits,
    targets
)

dX, dW, db = layer.backward(
    d_logits
)


# -------------------------
# Numerical gradient
# -------------------------

epsilon = 1e-5

row = 1
col = 2

original = layer.W[row, col]


layer.W[row, col] = original + epsilon

loss_plus = cross_entropy_loss(
    layer.forward(X),
    targets
)


layer.W[row, col] = original - epsilon

loss_minus = cross_entropy_loss(
    layer.forward(X),
    targets
)


layer.W[row, col] = original


numerical_gradient = (
    loss_plus - loss_minus
) / (2 * epsilon)


analytical_gradient = dW[row, col]


print("Numerical gradient:",
      numerical_gradient)

print("Analytical gradient:",
      analytical_gradient)

print(
    "Difference:",
    abs(
        numerical_gradient
        - analytical_gradient
    )
)