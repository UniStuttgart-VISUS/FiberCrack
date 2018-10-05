import pickle

import numpy as np


class Normalizer:
    """
    Normalizes input data to mean of zero and variance of one.
    Unlike scikit, supports N-dimensional arrays, with features on an arbitrary axis.
    """

    def __init__(self, minStd: float = 1e-3):
        self._axis = 0
        self._coefs = np.array([])
        self._isFit = False
        self._minStd = minStd
        self._dtype = None  # type: np.dtype

    def fit(self, data: np.ndarray, axis):

        if np.issubdtype(data.dtype, np.integer):
            raise ValueError("Usage of Normalizer with integer types can lead to overflow.")

        self._dtype = data.dtype

        if axis < 0:
            axis += data.ndim

        # data = data.copy()
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

        self._isFit = True

        return self

    def reset_to_trivial(self, featureNum, axis):
        self._axis = axis
        self._coefs = np.tile([0, 1], (featureNum, 1))
        return self

    def scale(self, data: np.ndarray, inPlace=False):

        if np.issubdtype(data.dtype, np.integer):
            raise ValueError("Usage of Normalizer with integer types can lead to overflow.")

        if self._dtype != data.dtype:
            raise ValueError("Normalizer has been 'fit()' with dtype {}, but dtype '{}' is given to 'scale()'."
                             .format(self._dtype, data.dtype))

        if not inPlace:
            data = data.copy()

        for i in range(0, data.shape[self._axis]):
            selector = self._get_selector(i, self._axis, data.ndim)
            if not self.is_feature_degenerate(i):  # Guard from division be zero or very small values.
                data[selector] = (data[selector] - self._coefs[i, 0]) / self._coefs[i, 1]
            else:
                data[selector] = data[selector] - self._coefs[i, 0]

        return data

    def scale_column(self, data: np.ndarray, columnIndex):
        assert data.shape[self._axis] == 1

        selector = self._get_selector(0, self._axis, data.ndim)
        return (data[selector] - self._coefs[columnIndex, 0]) / self._coefs[columnIndex, 1]

    def scale_back(self, data: np.ndarray):
        data = data.copy()
        for i in range(0, data.shape[self._axis]):
            selector = self._get_selector(i, self._axis, data.ndim)
            data[selector] = data[selector] * self._coefs[i, 1] + self._coefs[i, 0]

        return data

    def scale_back_column(self, data: np.ndarray, columnIndex):
        assert data.shape[self._axis] == 1

        selector = self._get_selector(0, self._axis, data.ndim)
        return data[selector] * self._coefs[columnIndex, 1] + self._coefs[columnIndex, 0]

    def is_feature_degenerate(self, featureIndex):
        self._throw_if_not_fit()

        return self._coefs[featureIndex, 1] < self._minStd

    def zero_degenerate_features(self, data):
        self._throw_if_not_fit()

        degenerateFeatures = [i for i in range(data.shape[self._axis]) if self.is_feature_degenerate(i)]
        data[self._get_selector(degenerateFeatures, self._axis, data.ndim)] = 0
        return data

    def _get_selector(self, index, axis, dimNum):
        return [(slice(None) if a != axis else index) for a in range(0, dimNum)]

    def _throw_if_not_fit(self):
        if not self._isFit:
            raise RuntimeError("Invalid operation: Normalizer hasn't been fit to any data yet.")

    def save(self, filepath: str):
        with open(filepath, 'wb') as file:
            pickle.dump(self, file)

    @classmethod
    def load(cls, filepath: str) -> 'Normalizer':
        with open(filepath, 'rb') as file:
            return pickle.load(file)