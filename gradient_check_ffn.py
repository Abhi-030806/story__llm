import numpy as np

from feed_forward import FeedForward


np.random.seed(42)

ffn = FeedForward(
    embedding_dim=4,
    hidden_dim=6
)

X = np.random.randn(3, 4)

# Forward
output = ffn.forward(X)

# Use a simple scalar loss:
# L = sum(output)
d_output = np.ones_like(output)

# Analytical gradients
dX, dW1, db1, dW2, db2 = ffn.backward(
    d_output
)


# Numerical gradient for W1[1, 2]
epsilon = 1e-5

row = 1
col = 2

original = ffn.W1[row, col]


# W + epsilon
ffn.W1[row, col] = original + epsilon

loss_plus = np.sum(
    ffn.forward(X)
)


# W - epsilon
ffn.W1[row, col] = original - epsilon

loss_minus = np.sum(
    ffn.forward(X)
)


# Restore
ffn.W1[row, col] = original


numerical_gradient = (
    loss_plus - loss_minus
) / (2 * epsilon)

analytical_gradient = dW1[row, col]

difference = abs(
    numerical_gradient
    - analytical_gradient
)


print("Numerical gradient:", numerical_gradient)
print("Analytical gradient:", analytical_gradient)
print("Difference:", difference)