import numpy as np
import typing


class Normalizer:
    """
    Normalizes input data to mean of zero and variance of one.
    Unlike scikit, supports N-dimensional arrays, with features on an arbitrary axis.
    """
    _coefs = None
    _axis = None

    def fit(self, data: np.ndarray, axis):
        data = data.copy()
        featureNum = data.shape[axis]
        self._axis = axis
        self._coefs = np.empty((featureNum, 2))

        for i in range(0, featureNum):
            # Extract the feature.
            selector = self._get_selector(i, axis, data.ndim)
            feature = data[selector]
            # Compute the moments.
            mean = np.mean(feature)
            std = np.std(feature)
            # Save the moments for future use
            self._coefs[i, :] = [mean, std]

            # Scale the feature.
            data[selector] = (feature - mean) / std

        return self

    def scale(self, data: np.ndarray):
        data = data.copy()
        for i in range(0, data.shape[self._axis]):
            selector = self._get_selector(i, self._axis, data.ndim)
            data[selector] = (data[selector] - self._coefs[i, 0]) / self._coefs[i, 1]

        return data

    def scale_back(self, data: np.ndarray):
        data = data.copy()
        for i in range(0, data.shape[self._axis]):
            selector = self._get_selector(i, self._axis, data.ndim)
            data[selector] = data[selector] * self._coefs[i, 1] + self._coefs[i, 0]

        return data

    def _get_selector(self, index, axis, dimNum):
        return [(slice(None) if a != axis else index) for a in range(0, dimNum)]