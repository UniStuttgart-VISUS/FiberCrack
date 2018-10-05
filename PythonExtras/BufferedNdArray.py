
from typing import Tuple

import numpy as np

import PythonExtras.numpy_extras as npe
from PythonExtras.CppWrapper import CppWrapper


class BufferedNdArray:
    """
    A Python wrapper around a C++ class implementing buffered read/write access
    to an nd-array on disk.
    This wrapper stores a pointer to a C++ object and uses ctypes and static wrapper
    functions to call the class methods.
    """

    def __init__(self, filepath: str, shape: Tuple, dtype: np.dtype, maxBufferSize: int):
        assert (dtype == np.uint8)

        self.filepath = filepath
        self.dtype = dtype
        self.ndim = len(shape)
        self.shape = shape
        self.cPointer = CppWrapper.bna_construct(filepath, shape, maxBufferSize)

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        self.destroy()

    def read_slice(self, index: int) -> np.ndarray:
        output = np.empty(self.shape[1:], dtype=self.dtype)
        CppWrapper.bna_read_slice(self.cPointer, output, index)

        return output

    def destroy(self):
        CppWrapper.bna_destruct(self.cPointer)

    def flush(self, flushOsBuffer: bool = False):
        CppWrapper.bna_flush(self.cPointer, flushOsBuffer)

    def fill_box(self, value, cornerLow: Tuple, cornerHigh: Tuple):
        CppWrapper.bna_fill_box(self.cPointer, value, cornerLow, cornerHigh)