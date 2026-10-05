import numpy as np

def relu(x):
    return np.maximum(0, x)

def feed_forward(x: np.ndarray, W1: np.ndarray, b1: np.ndarray,
                 W2: np.ndarray, b2: np.ndarray) -> np.ndarray:
    """
    Returns the position-wise feed-forward output.
    """
    h1=np.add((x@W1),b1)
    h1_relu=relu(h1)
    out=np.add((h1_relu@W2),b2)
    return out
