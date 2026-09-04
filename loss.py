import numpy as np


def softmax(logits):

    logits = logits - np.max(
        logits,
        axis=-1,
        keepdims=True
    )

    exp_logits = np.exp(logits)

    return exp_logits / np.sum(
        exp_logits,
        axis=-1,
        keepdims=True
    )


def cross_entropy_loss(logits, targets):

    batch_size = logits.shape[0]
    sequence_length = logits.shape[1]

    logits_flat = logits.reshape(
        -1,
        logits.shape[-1]
    )

    targets_flat = targets.reshape(-1)

    probabilities = softmax(logits_flat)

    correct_probabilities = probabilities[
        np.arange(len(targets_flat)),
        targets_flat
    ]

    loss = -np.mean(
        np.log(correct_probabilities + 1e-9)
    )

    return loss


def cross_entropy_backward(logits, targets):

    logits_flat = logits.reshape(
        -1,
        logits.shape[-1]
    )

    targets_flat = targets.reshape(-1)

    probabilities = softmax(logits_flat)

    gradients = probabilities.copy()

    gradients[
        np.arange(len(targets_flat)),
        targets_flat
    ] -= 1

    gradients /= len(targets_flat)

    return gradients.reshape(
        logits.shape
    )


if __name__ == "__main__":

    np.random.seed(42)

    logits = np.random.randn(
        4, 8, 1564
    )

    targets = np.random.randint(
        0,
        1564,
        size=(4, 8)
    )

    loss = cross_entropy_loss(
        logits,
        targets
    )

    d_logits = cross_entropy_backward(
        logits,
        targets
    )

    print("Logits:", logits.shape)
    print("Targets:", targets.shape)
    print("Loss:", loss)
    print("Gradient:", d_logits.shape)