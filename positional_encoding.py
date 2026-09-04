import numpy as np


def positional_encoding(sequence_length, embedding_dim):

    position = np.arange(sequence_length)[:, np.newaxis]
    dimension = np.arange(embedding_dim)[np.newaxis, :]

    angle = position / np.power(
        10000,
        (2 * (dimension // 2)) / embedding_dim
    )

    encoding = np.zeros((sequence_length, embedding_dim))

    encoding[:, 0::2] = np.sin(angle[:, 0::2])
    encoding[:, 1::2] = np.cos(angle[:, 1::2])

    return encoding


if __name__ == "__main__":

    encoding = positional_encoding(4, 128)

    print("Shape:", encoding.shape)
    print("Position 0:")
    print(encoding[0])