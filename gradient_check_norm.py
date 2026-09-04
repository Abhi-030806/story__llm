import numpy as np

from transformer_block import LayerNorm


np.random.seed(42)

norm = LayerNorm(4)

X = np.random.randn(3, 4)

# Forward
output = norm.forward(X)

# Simple scalar loss
d_output = np.ones_like(output)

# Analytical gradient
dX, dgamma, dbeta = norm.backward(
    d_output
)


# Numerical gradient for X[1, 2]
epsilon = 1e-5

row = 1
col = 2

original = X[row, col]


# X + epsilon
X[row, col] = original + epsilon

loss_plus = np.sum(
    norm.forward(X)
)


# X - epsilon
X[row, col] = original - epsilon

loss_minus = np.sum(
    norm.forward(X)
)


# Restore
X[row, col] = original

numerical_gradient = (
    loss_plus - loss_minus
) / (2 * epsilon)

analytical_gradient = dX[row, col]

difference = abs(
    numerical_gradient
    - analytical_gradient
)


print("Numerical gradient:", numerical_gradient)
print("Analytical gradient:", analytical_gradient)
print("Difference:", difference)