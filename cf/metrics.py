import numpy as np


def rmse(pred, target):
    """Root mean squared error between two arrays of the same shape."""
    pred = np.asarray(pred, dtype=np.float64)
    target = np.asarray(target, dtype=np.float64)
    return float(np.sqrt(np.mean((pred - target) ** 2)))