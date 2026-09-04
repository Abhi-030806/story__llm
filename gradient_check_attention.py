import numpy as np

from multi_head_attention import MultiHeadAttention


np.random.seed(42)

attention = MultiHeadAttention(
    embedding_dim=4,
    num_heads=2
)

X = np.random.randn(3, 4)

# Forward
output = attention.forward(X)

# Simple scalar loss
d_output = np.ones_like(output)

# Analytical gradients
dX, dW_Q, dW_K, dW_V, dW_O = attention.backward(
    d_output
)


# Numerical gradient for W_Q[1, 2]
epsilon = 1e-5

row = 1
col = 2

original = attention.W_Q[row, col]


# W + epsilon
attention.W_Q[row, col] = original + epsilon

loss_plus = np.sum(
    attention.forward(X)
)


# W - epsilon
attention.W_Q[row, col] = original - epsilon

loss_minus = np.sum(
    attention.forward(X)
)


# Restore
attention.W_Q[row, col] = original

numerical_gradient = (
    loss_plus - loss_minus
) / (2 * epsilon)

analytical_gradient = dW_Q[row, col]

difference = abs(
    numerical_gradient
    - analytical_gradient
)


print("Numerical gradient:", numerical_gradient)
print("Analytical gradient:", analytical_gradient)
print("Difference:", difference)