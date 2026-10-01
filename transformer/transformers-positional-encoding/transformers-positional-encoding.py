import numpy as np

def positional_encoding(seq_length: int, d_model: int) -> np.ndarray:
    """
    Returns the sinusoidal position matrix.
    """

    position = []

    for pos in range(seq_length):

        arr = []

        for i in range(d_model):

            if i % 2 == 0:
                value = np.sin(
                    pos / (10000 ** (i / d_model))
                )
            else:
                value = np.cos(
                    pos / (10000 ** ((i - 1) / d_model))
                )

            arr.append(value)

        position.append(arr)

    return np.array(position)