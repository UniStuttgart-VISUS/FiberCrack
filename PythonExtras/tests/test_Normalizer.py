import tempfile
import os
from typing import Union, List, Tuple

import unittest
import h5py
import numpy as np

from PythonExtras.Normalizer import Normalizer


class NormalizerTest(unittest.TestCase):

    def test_basic_normalization(self):

        data = np.empty((30, 10, 10, 5), dtype=np.float32)
        for featureIndex in range(0, 5):
            data[..., featureIndex] = np.arange(0, 10) * (featureIndex + 1) + 5 * featureIndex

        normalizer = Normalizer().fit(data, axis=3)

        dataNorm = normalizer.scale(data)

        self.assertAlmostEqual(np.mean(dataNorm), 0.0)
        self.assertAlmostEqual(np.std(dataNorm), 1.0)

        self.assertTrue(np.all(np.logical_not(np.isclose(data, dataNorm))),
                        msg='Normalized data should differ')
        self.assertTrue(np.all(np.isclose(data, normalizer.scale_back(dataNorm))),
                        msg='Scaling back should restore the original data.')

    def test_scaling_in_place(self):

        data = np.empty((30, 10, 10, 5), dtype=np.float32)
        for featureIndex in range(0, 5):
            data[..., featureIndex] = np.arange(0, 10) * (featureIndex + 1) + 5 * featureIndex

        origData = data.copy()

        normalizer = Normalizer().fit(data, axis=3)

        normalizer.scale(data, inPlace=True)

        self.assertAlmostEqual(np.mean(data), 0.0)
        self.assertAlmostEqual(np.std(data), 1.0)

        self.assertTrue(np.all(np.logical_not(np.isclose(data, origData))),
                        msg='Normalized data should differ')
        self.assertTrue(np.all(np.isclose(origData, normalizer.scale_back(data))),
                        msg='Scaling back should restore the original data.')

    def test_serialization(self):
        data = np.random.randint(0, 100, (30, 10, 10, 5), dtype=np.int32).astype(np.float32)
        normalizer = Normalizer().fit(data, axis=3)
        originalResult = normalizer.scale(data, inPlace=False)

        tempFilepath = os.path.join(tempfile.gettempdir(), 'test-normalizer-serialization.pcl')
        normalizer.save(tempFilepath)

        normalizer = Normalizer.load(tempFilepath)
        resultAfterLoad = normalizer.scale(data, inPlace=False)

        self.assertTrue(np.all(np.equal(originalResult, resultAfterLoad)))


if __name__ == '__main__':
    unittest.main()
