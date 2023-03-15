import numpy as np

class Input():
    def __init__(self, units : int):
        self.shape = tuple(units)

def normalize(array : np.ndarray, r : tuple[int, int]):
    return np.interp(array, (array.min(), array.max()), r)
