import time
import math
import functools
import operator
import timeit
from typing import Union, List, Tuple

import unittest
import h5py
import numpy as np


import PythonExtras.numpy_extras as numpy_extras
import PythonExtras.volume_tools as volume_tools
from PythonExtras.CppWrapper import CppWrapper
import ctypes

def reference_extract_patches(data: Union[np.ndarray, h5py.Dataset], sourceAxes: Tuple, patchSize: Tuple,
                              patchStride: Tuple = None, verbose=False):
    # By default, extract every patch.
    if patchStride is None:
        patchStride = (1,) * len(sourceAxes)

    result = None

    patchCenters = []  # Store geometrical center of each path. (Can be useful for the caller.)
    patchNumber = reference_compute_patch_number(data.shape, sourceAxes, patchSize, patchStride)
    patchNumberFlat = np.prod(np.asarray(patchNumber))  # type: int

    i = 0
    for patch, patchCenter, patchIndex in reference_extract_patches_gen(data, sourceAxes, patchSize, patchStride, verbose):
        if result is None:
            resultShape = (patchNumberFlat,) + patch.shape
            result = np.empty(resultShape, dtype=patch.dtype)
        result[i, ...] = patch
        patchCenters.append(patchCenter)
        i += 1

    return result, patchCenters, patchNumber


def reference_compute_patch_number(dataShape: Tuple, sourceAxes: Tuple, patchSize: Tuple,
                                   patchStride: Tuple = None):
    patchNumber = []
    for i in sourceAxes:
        totalPatchNumber = dataShape[i] - patchSize[sourceAxes.index(i)] + 1
        stride = patchStride[sourceAxes.index(i)]
        patchNumber.append(int(math.ceil(totalPatchNumber / stride)))

    return patchNumber


def reference_extract_patches_gen(data: Union[np.ndarray, h5py.Dataset], sourceAxes: Tuple, patchSize: Tuple,
                                  patchStride: Tuple = None, verbose=False):
    # By default, extract every patch.
    if patchStride is None:
        patchStride = (1,) * len(sourceAxes)

    patchNumber = reference_compute_patch_number(data.shape, sourceAxes, patchSize, patchStride)
    patchNumberFlat = np.prod(np.asarray(patchNumber))

    lastPrintTime = time.time()

    # Since the number of traversed dimension is dynamically specified, we cannot use 'for' loops.
    # Instead, iterate a flat index and unflatten it inside the loop. (Easier than recursion.)
    for indexFlat in range(patchNumberFlat):
        patchIndexNd = numpy_extras.unflatten_index(patchNumber, indexFlat)
        # Since we skip some of the patches, scale the index accordingly.
        dataIndexNd = tuple(np.asarray(patchIndexNd, dtype=np.int) * np.asarray(patchStride, dtype=np.int))

        dataSelector = tuple()
        patchCenter = tuple()
        for axis in range(data.ndim):
            if axis not in sourceAxes:
                dataSelector += (slice(None),)  # Take everything, if it's not a source axis.
            else:
                patchAxis = sourceAxes.index(axis)
                # Take a slice along the axis.
                dataSelector += (slice(dataIndexNd[patchAxis], dataIndexNd[patchAxis] + patchSize[patchAxis]),)
                patchCenter += (dataIndexNd[patchAxis] + int(patchSize[patchAxis] / 2),)

        yield (data[dataSelector], patchCenter, patchIndexNd)

        if verbose and time.time() - lastPrintTime > 20:
            lastPrintTime = time.time()
            print("Extracting patches: {:02.2f}%".format(indexFlat / patchNumberFlat * 100))


def reference_extract_patched_all_data_without_empty_4d(data: np.ndarray,
                                                        patchSize: Tuple, patchStride: Tuple, emptyValue: int):

    patchedData, *r = reference_extract_patches(data, (0, 1, 2, 3), patchSize, patchStride)

    allDataX = patchedData[:, :-1, ...]
    # The center of the last frame is the prediction target.
    allDataY = patchedData[:, -1, patchSize[1] // 2, patchSize[2] // 2, patchSize[3] // 2]

    nonemptyPatchMaskY = np.not_equal(allDataY, emptyValue)
    nonemptyPatchMaskX = np.any(np.not_equal(allDataX, emptyValue), axis=tuple(range(1, allDataX.ndim)))
    nonemptyPatchMask = np.logical_or(nonemptyPatchMaskX, nonemptyPatchMaskY)

    allDataX = allDataX[nonemptyPatchMask]
    allDataY = allDataY[nonemptyPatchMask, np.newaxis]

    return allDataX, allDataY, None


def time_extract_patches():
    data = np.random.uniform(0, 255, (256, 256, 256)).astype(dtype=np.uint8)

    iterationNumber = 3

    print("Small patch")
    timeStart = time.time()
    for i in range(iterationNumber):
        print("Testing Python {}/{}".format(i, iterationNumber))
        patchesCorrect, *r = reference_extract_patches(data, (0, 1, 2), (3, 3, 3), (3, 3, 3))

    print("Python time: {:.2f} seconds.".format(time.time() - timeStart))

    timeStart = time.time()
    for i in range(iterationNumber):
        print("Testing C++ {}/{}".format(i, iterationNumber))
        patchesCpp, *r = CppWrapper.extract_patches(data, (0, 1, 2), (3, 3, 3), (3, 3, 3))

    print("C++ time: {:.2f} seconds.".format(time.time() - timeStart))

    print("Large patch")
    timeStart = time.time()
    for i in range(iterationNumber):
        print("Testing Python {}/{}".format(i, iterationNumber))
        patchesCorrect, *r = reference_extract_patches(data, (0, 1, 2), (16, 16, 16), (16, 16, 16))

    print("Python time: {:.2f} seconds.".format(time.time() - timeStart))

    timeStart = time.time()
    for i in range(iterationNumber):
        print("Testing C++ {}/{}".format(i, iterationNumber))
        patchesCpp, *r = CppWrapper.extract_patches(data, (0, 1, 2), (16, 16, 16), (16, 16, 16))

    print("C++ time: {:.2f} seconds.".format(time.time() - timeStart))


def time_extract_training_data_without_empty():
    print("-------- Training data extraction ---------")
    data = np.random.uniform(0, 255, (256, 64, 64, 64)).astype(dtype=np.uint8)

    data[128:, ...] = 0

    iterationNumber = 3

    print("Small patch")
    timeStart = time.time()
    for i in range(iterationNumber):
        print("Testing Python {}/{}".format(i, iterationNumber))
        r = reference_extract_patched_all_data_without_empty_4d(data, (3, 3, 3, 3), (3, 3, 3, 3), 0)

    print("Python time: {:.2f} seconds.".format(time.time() - timeStart))

    timeStart = time.time()
    for i in range(iterationNumber):
        print("Testing C++ {}/{}".format(i, iterationNumber))
        r = numpy_extras.extract_patched_all_data_without_empty_4d(data, (3, 3, 3, 3), (3, 3, 3, 3), 0)

    print("C++ time: {:.2f} seconds.".format(time.time() - timeStart))

    print("Large patch")
    timeStart = time.time()
    for i in range(iterationNumber):
        print("Testing Python {}/{}".format(i, iterationNumber))
        r = reference_extract_patched_all_data_without_empty_4d(data, (16, 16, 16, 16), (16, 16, 16, 16), 0)

    print("Python time: {:.2f} seconds.".format(time.time() - timeStart))

    timeStart = time.time()
    for i in range(iterationNumber):
        print("Testing C++ {}/{}".format(i, iterationNumber))
        r = numpy_extras.extract_patched_all_data_without_empty_4d(data, (16, 16, 16, 16), (16, 16, 16, 16), 0)

    print("C++ time: {:.2f} seconds.".format(time.time() - timeStart))


def time_multiply_implementations():

    def multiply_numpy(iterable):
        return np.prod(np.array(iterable))

    def multiply_functools(iterable):
        return functools.reduce(operator.mul, iterable)

    def multiply_manual(iterable):
        prod = 1
        for x in iterable:
            prod *= x

        return prod

    sizesToTest = [5, 10, 100, 1000, 10000, 100000]

    for size in sizesToTest:
        data = [1] * size

        timerNumpy = timeit.Timer(lambda: multiply_numpy(data))
        timerFunctools = timeit.Timer(lambda: multiply_functools(data))
        timerManual = timeit.Timer(lambda: multiply_manual(data))

        repeats = int(1e6 / size)
        resultNumpy = timerNumpy.timeit(repeats)
        resultFunctools = timerFunctools.timeit(repeats)
        resultManual = timerManual.timeit(repeats)

        print("Input size: {: 7d} Repeats: {: 8d}    Numpy: {:.3f}, functools: {:.3f}, manual: {:.3f}".format(
            size, repeats, resultNumpy, resultFunctools, resultManual
        ))

    # Input size:       5 Repeats:   200000    Numpy: 0.815, functools: 0.110, manual: 0.086
    # Input size:      10 Repeats:   100000    Numpy: 0.456, functools: 0.090, manual: 0.081
    # Input size:     100 Repeats:    10000    Numpy: 0.103, functools: 0.042, manual: 0.039
    # Input size:    1000 Repeats:     1000    Numpy: 0.052, functools: 0.045, manual: 0.035
    # Input size:   10000 Repeats:      100    Numpy: 0.060, functools: 0.048, manual: 0.052
    # Input size:  100000 Repeats:       10    Numpy: 0.052, functools: 0.035, manual: 0.047

if __name__ == '__main__':
    time_multiply_implementations()
    # time_extract_training_data_without_empty()
    # time_extract_patches()


























