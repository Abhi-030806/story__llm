import numpy as np


# Forward
X = np.array([[2.0, 3.0]])

W = np.array([
    [0.5],
    [0.2]
])

b = np.array([0.1])

target = np.array([[2.0]])

# Prediction
Y = X @ W + b

# Mean squared error
loss = np.mean((Y - target) ** 2)

print("Prediction:", Y)
print("Loss:", loss)


# Backward
dY = 2 * (Y - target)

dW = X.T @ dY
db = np.sum(dY)

# Learning rate
learning_rate = 0.1

# Update parameters
W = W - learning_rate * dW
b = b - learning_rate * db

print("\nUpdated W:")
print(W)

print("Updated b:")
print(b)

# Forward again
Y_new = X @ W + b
loss_new = np.mean((Y_new - target) ** 2)

print("New prediction:", Y_new)
print("New loss:", loss_new)