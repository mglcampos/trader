import numpy as np

class Welford(object):
    """Calculate the moving standard deviation using Welford's Algorithm."""

    def __init__(self, window_size):
        """Create object with a given window size.
        Size can be -1, infinite size or > 1 meaning static size."""

        self.window_size = window_size
        if not (self.window_size == -1 or self.window_size > 1):
            raise Exception("size must be -1 or > 1")

        self._window = []
        self.n = 0.0
        self.mean = 0.0
        self._s = 0.0

    @property
    def std(self):
        """Returns the standard deviation."""

        if self.n == 1:
            return 0.0
        return np.sqrt(self._s / (self.n - 1))

    def update(self, value):
        """Updates the standard deviation with a new value and removes the old one if n > window_size."""

        self.n += 1.0
        diff = (value - self.mean)
        self.mean += diff / self.n
        self._s += diff * (value - self.mean)

        if self.n > self.window_size:
            old = self._window.pop(0)
            oldM = (self.n * self.mean - old) / (self.n - 1)
            self._s -= (old - self.mean) * (old - oldM)
            self.mean = oldM
            self.n -= 1
        self._window.append(value)

    def update_and_return(self, vector):
        for value in vector:
            self.update(value)
        return self.std
