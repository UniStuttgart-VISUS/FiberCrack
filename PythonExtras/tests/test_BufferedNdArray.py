import os
from typing import Union, List, Tuple

import unittest
import numpy as np

from PythonExtras.BufferedNdArray import BufferedNdArray


class BufferedNdArrayTest(unittest.TestCase):
    """
    Some of the tests are implemented in C++.
    """

    def test_construction_destruction(self):
        filepath = 'T:\\tmp\\test-buffered--nd-array-python.raw'

        if os.path.exists(filepath):
            os.unlink(filepath)

        with BufferedNdArray(filepath, (10, 128, 128, 128), np.uint8, int(1e5)) as array:
            pass

        self.assertTrue(os.path.exists(filepath))