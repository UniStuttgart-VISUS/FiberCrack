import time
import math
import tempfile
import os
import json
from typing import Union, List, Tuple

import unittest
import h5py
import numpy as np

import PythonExtras.numpy_extras as numpy_extras
import PythonExtras.volume_tools as volume_tools
from PythonExtras.CppWrapper import CppWrapper
from PythonExtras.BufferedNdArray import BufferedNdArray


# ---------------------------------------------------------
# Reference Python implementations of the methods reimplemented in C++.

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
    for patch, patchCenter, patchIndex in reference_extract_patches_gen(data, sourceAxes, patchSize, patchStride,
                                                                        verbose):
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

    patchedData, patchCenters, *r = reference_extract_patches(data, (0, 1, 2, 3), patchSize, patchStride)

    allDataX = patchedData[:, :-1, ...]
    # The center of the last frame is the prediction target.
    allDataY = patchedData[:, -1, patchSize[1] // 2, patchSize[2] // 2, patchSize[3] // 2]

    nonemptyPatchMaskY = np.not_equal(allDataY, emptyValue)
    nonemptyPatchMaskX = np.any(np.not_equal(allDataX, emptyValue), axis=tuple(range(1, allDataX.ndim)))
    nonemptyPatchMask = np.logical_or(nonemptyPatchMaskX, nonemptyPatchMaskY)

    allDataX = allDataX[nonemptyPatchMask]
    allDataY = allDataY[nonemptyPatchMask, np.newaxis]
    patchCenters = np.asarray(patchCenters, dtype=np.uint64)[nonemptyPatchMask]

    patchIndices = patchCenters[:, :] - np.asarray(patchSize) // 2

    return allDataX, allDataY, patchIndices


class NumpyExtrasTest(unittest.TestCase):
    def test_flatten_index_simple(self):
        self.assertEqual(numpy_extras.flatten_index((1, 2, 3), (3, 4, 5)),
                         1 * 4 * 5 + 2 * 5 + 3)

    def test_extract_patches_simple(self):
        size = (3, 6, 5, 4)
        volume = np.random.uniform(0, 255, size).astype(np.uint8)

        patchSize = (2, 5, 3, 4)
        patchStride = (1, 4, 2, 2)

        patches, c, patchNumber = numpy_extras.extract_patches(volume, (0, 1, 2, 3), patchSize, patchStride)

        patchIndexNd = (1, 0, 1, 0)
        patchIndex = numpy_extras.flatten_index(patchIndexNd, patchNumber)

        correctPatch = volume[1:3, 0:5, 2:5, 0:4]
        actualPatch = patches[patchIndex, ...]

        self.assertTrue(np.equal(correctPatch, actualPatch).all())

    def test_c_wrapper_test(self):
        input = np.arange(0, 16).reshape((4, 4))  # type: np.ndarray
        expectedOutput = input * 2

        CppWrapper.test(input)

        self.assertTrue(np.equal(input, expectedOutput).all())

    def test_resize_array_point(self):
        input = np.asarray([[1, 2, 3],
                            [4, 5, 6],
                            [7, 8, 9],
                            [10, 11, 12],
                            [13, 14, 15],
                            [16, 17, 18]], dtype=np.float)
        targetSize = (3, 2)

        expectedOutput = np.asarray([[1, 3],
                                     [10, 12],
                                     [16, 18]], dtype=np.float)
        actualOutput = volume_tools.resize_array_point(input, targetSize)

        self.assertTrue(np.all(np.equal(expectedOutput, actualOutput)),
                        msg='\n {} \n <- expected, got -> \n {}'.format(expectedOutput, actualOutput))

        actualOutput = volume_tools.resize_array_point(input.astype(np.uint8), targetSize)

        self.assertTrue(np.all(np.equal(expectedOutput.astype(np.uint8), actualOutput)),
                        msg='\n {} \n <- expected, got -> \n {}'.format(expectedOutput, actualOutput))

    def test_extract_patches(self):
        # It's kind of stupid – testing against a reference implementation,
        # but manually generating large test cases is fairly annoying.
        # Also, we are testing the C++ implementation against the trusted Python code.

        data = np.arange(0, 5 * 7 * 9, dtype=np.uint8).reshape((5, 7, 9))
        patchesCorrect, centersCorrect, *r = reference_extract_patches(data, (0, 2), (2, 5), (1, 2))
        patchesCpp, centersCpp, *r = numpy_extras.extract_patches(data, (0, 2), (2, 5), (1, 2))

        self.assertEqual(patchesCorrect.shape, patchesCpp.shape)
        self.assertTrue(np.all(np.equal(patchesCorrect, patchesCpp)))
        self.assertTrue(np.all(np.equal(centersCorrect, centersCpp)))

        patchesCorrect, centersCorrect, *r = reference_extract_patches(data, (0, 1, 2), (1, 1, 1), (1, 1, 1))
        patchesCpp, centersCpp, *r = numpy_extras.extract_patches(data, (0, 1, 2), (1, 1, 1), (1, 1, 1))

        self.assertEqual(patchesCorrect.shape, patchesCpp.shape)
        self.assertTrue(np.all(np.equal(patchesCorrect, patchesCpp)))
        self.assertTrue(np.all(np.equal(centersCorrect, centersCpp)))

        patchesCorrect, centersCorrect, *r = reference_extract_patches(data, (1, 2), (1, 3), (5, 1))
        patchesCpp, centersCpp, *r = numpy_extras.extract_patches(data, (1, 2), (1, 3), (5, 1))

        self.assertEqual(patchesCorrect.shape, patchesCpp.shape)
        self.assertTrue(np.all(np.equal(patchesCorrect, patchesCpp)))
        self.assertTrue(np.all(np.equal(centersCorrect, centersCpp)))

        # --------------
        data = np.arange(0, 2 * 3 * 4 * 5 * 6, dtype=np.uint8).reshape((2, 3, 4, 5, 6))

        patchesCorrect, centersCorrect, *r = reference_extract_patches(data, (2, 4), (2, 2), (1, 3))
        patchesCpp, centersCpp, *r = numpy_extras.extract_patches(data, (2, 4), (2, 2), (1, 3))

        self.assertEqual(patchesCorrect.shape, patchesCpp.shape)
        self.assertTrue(np.all(np.equal(patchesCorrect, patchesCpp)))
        self.assertTrue(np.all(np.equal(centersCorrect, centersCpp)))

        # --------------
        data = np.arange(0, 20, dtype=np.uint8).reshape((5, 4))
        patchesCpp, *r = numpy_extras.extract_patches(data, (0, 1), (2, 3), (2, 1))

        manualResult = np.asarray([
            [[0, 1, 2],
             [4, 5, 6]],
            [[1, 2, 3],
             [5, 6, 7]],
            [[8, 9, 10],
             [12, 13, 14]],
            [[9, 10, 11],
             [13, 14, 15]],
        ], dtype=np.uint8)

        self.assertEqual(manualResult.shape, patchesCpp.shape)
        self.assertTrue(np.all(np.equal(manualResult, patchesCpp)))

        # --------------
        data = np.arange(0, 3 * 4 * 5, dtype=np.uint8).reshape((3, 4, 5))
        patchesCpp, *r = numpy_extras.extract_patches(data, (0, 2), (2, 2), (1, 3))

        manualResult = np.asarray([
            [
                [[0, 1], [5, 6], [10, 11], [15, 16]],
                [[20, 21], [25, 26], [30, 31], [35, 36]]
            ],
            [
                [[3, 4], [8, 9], [13, 14], [18, 19]],
                [[23, 24], [28, 29], [33, 34], [38, 39]]
            ],
            [
                [[20, 21], [25, 26], [30, 31], [35, 36]],
                [[40, 41], [45, 46], [50, 51], [55, 56]],
            ],
            [
                [[23, 24], [28, 29], [33, 34], [38, 39]],
                [[43, 44], [48, 49], [53, 54], [58, 59]]
            ],
        ], dtype=np.uint8)

        self.assertEqual(manualResult.shape, patchesCpp.shape)
        self.assertTrue(np.all(np.equal(manualResult, patchesCpp)))

    def test_extract_patches_batched(self):
        # It's kind of stupid – testing against a reference implementation,
        # but manually generating large test cases is fairly annoying.

        data = np.arange(0, 2 * 3 * 4 * 5 * 6, dtype=np.uint8).reshape((2, 3, 4, 5, 6))

        patchesCorrect, centersCorrect, *r = reference_extract_patches(data, (2, 4), (2, 2), (1, 3))
        patchesCpp, centersCpp, *r = numpy_extras.extract_patches_batch(data, (2, 4), (2, 2), (1, 3),
                                                                        batchStart=4, batchSize=2)

        self.assertEqual(patchesCorrect.shape[1:], patchesCpp.shape[1:])
        self.assertTrue(np.all(np.equal(patchesCorrect[4:4 + 2], patchesCpp)))
        self.assertTrue(np.all(np.equal(centersCorrect[4:4 + 2], centersCpp)))

        # --------------
        data = np.arange(0, 3 * 4 * 5, dtype=np.uint8).reshape((3, 4, 5))
        patchesCpp, *r = numpy_extras.extract_patches_batch(data, (0, 2), (2, 2), (1, 3),
                                                            batchStart=2, batchSize=2)

        manualResult = np.asarray([
            [
                [[20, 21], [25, 26], [30, 31], [35, 36]],
                [[40, 41], [45, 46], [50, 51], [55, 56]],
            ],
            [
                [[23, 24], [28, 29], [33, 34], [38, 39]],
                [[43, 44], [48, 49], [53, 54], [58, 59]]
            ]
        ], dtype=np.uint8)

        self.assertEqual(manualResult.shape, patchesCpp.shape)
        self.assertTrue(np.all(np.equal(manualResult, patchesCpp)))

    def test_extract_patched_training_data_without_empty_4d(self):
        data = np.arange(0, 3 * 3 * 3 * 3, dtype=np.uint8).reshape((3, 3, 3, 3))

        dataXActual, dataYActual, *r = \
            numpy_extras.extract_patched_all_data_without_empty_4d(data, (2, 2, 2, 2), (2, 1, 1, 1), 15,
                                                                   batchSize=100)

        manualResult = np.asarray(
            [[[[[0, 1],
                [3, 4]],
               [[9, 10],
                [12, 13]]]],
             [[[[1, 2],
                [4, 5]],
               [[10, 11],
                [13, 14]]]],
             [[[[3, 4],
                [6, 7]],
               [[12, 13],
                [15, 16]]]],
             [[[[4, 5],
                [7, 8]],
               [[13, 14],
                [16, 17]]]],
             [[[[9, 10],
                [12, 13]],
               [[18, 19],
                [21, 22]]]],
             [[[[10, 11],
                [13, 14]],
               [[19, 20],
                [22, 23]]]],
             [[[[12, 13],
                [15, 16]],
               [[21, 22],
                [24, 25]]]],
             [[[[13, 14],
                [16, 17]],
               [[22, 23],
                [25, 26]]]]]
            , dtype=np.uint8)

        self.assertEqual(manualResult.shape, dataXActual.shape)
        self.assertTrue(np.all(np.equal(manualResult, dataXActual)))

        # ------------------------------------------------------

        # It's kind of stupid – testing against a reference implementation,
        # but manually generating large test cases is fairly annoying.
        # Also, we are testing the C++ implementation against the trusted Python code.

        data = np.arange(0, 5 * 7 * 9 * 10, dtype=np.uint8).reshape((5, 7, 9, 10))
        dataXCorrect, dataYCorrect, patchIndicesCorrect = \
            reference_extract_patched_all_data_without_empty_4d(data, (3, 4, 3, 5), (1, 1, 2, 1), 15)

        dataXActual, dataYActual, patchIndicesActual = \
            numpy_extras.extract_patched_all_data_without_empty_4d(data, (3, 4, 3, 5), (1, 1, 2, 1), 15,
                                                                   batchSize=100)

        self.assertEqual(dataXCorrect.shape, dataXActual.shape)
        self.assertEqual(dataYCorrect.shape, dataYActual.shape)
        self.assertEqual(patchIndicesCorrect.shape, patchIndicesActual.shape)

        self.assertTrue(np.all(np.equal(dataXCorrect, dataXActual)))
        self.assertTrue(np.all(np.equal(dataYCorrect, dataYActual)))
        self.assertTrue(np.all(np.equal(patchIndicesCorrect, patchIndicesActual)))

        # ------------------------------------------------------

        # Stride of (1, 1, 1, 1).
        data = np.arange(0, 5 * 7 * 9 * 10, dtype=np.uint8).reshape((5, 7, 9, 10))
        dataXCorrect, dataYCorrect, patchIndicesCorrect = \
            reference_extract_patched_all_data_without_empty_4d(data, (3, 4, 3, 5), (1, 1, 1, 1), 15)

        dataXActual, dataYActual, patchIndicesActual = \
            numpy_extras.extract_patched_all_data_without_empty_4d(data, (3, 4, 3, 5), (1, 1, 1, 1), 15,
                                                                   batchSize=100)
        self.assertEqual(dataXCorrect.shape, dataXActual.shape)
        self.assertEqual(dataYCorrect.shape, dataYActual.shape)
        self.assertEqual(patchIndicesCorrect.shape, patchIndicesActual.shape)

        self.assertTrue(np.all(np.equal(dataXCorrect, dataXActual)))
        self.assertTrue(np.all(np.equal(dataYCorrect, dataYActual)))
        self.assertTrue(np.all(np.equal(patchIndicesCorrect, patchIndicesActual)))

        # ------------------------------------------------------
        # The same thing, but with some empty patches
        data = np.arange(0, 5 * 7 * 9 * 10, dtype=np.uint8).reshape((5, 7, 9, 10))

        data[0:4, 0:6, 0:8, 1:9] = 15

        dataXCorrect, dataYCorrect, patchIndicesCorrect = \
            reference_extract_patched_all_data_without_empty_4d(data, (3, 4, 3, 5), (1, 1, 2, 1), 15)

        dataXActual, dataYActual, patchIndicesActual = \
            numpy_extras.extract_patched_all_data_without_empty_4d(data, (3, 4, 3, 5), (1, 1, 2, 1), 15,
                                                                   batchSize=100)

        self.assertEqual(dataXCorrect.shape, dataXActual.shape)
        self.assertEqual(dataYCorrect.shape, dataYActual.shape)
        self.assertEqual(patchIndicesCorrect.shape, patchIndicesActual.shape)

        self.assertTrue(np.all(np.equal(dataXCorrect, dataXActual)))
        self.assertTrue(np.all(np.equal(dataYCorrect, dataYActual)))
        self.assertTrue(np.all(np.equal(patchIndicesCorrect, patchIndicesActual)))

    def test_shuffle_hdf_arrays_together(self):
        shapeX = (1000, 25, 25)
        shapeY = (1000, 25, 1)

        tempDir = tempfile.mkdtemp()
        file = h5py.File(os.path.join(tempDir, 'temp.h5py'))
        dataX = file.create_dataset('tempX', shapeX, np.float32)
        dataX[...] = np.random.uniform(0, 1000, shapeX)

        dataY = file.create_dataset('tempY', shapeY, np.float32)
        for i in range(0, shapeX[0]):
            for j in range(0, shapeX[1]):
                simpleHash = np.sum(dataX[i, j, :].astype(np.int32)) % 73
                dataY[i, j] = simpleHash

        firstColumnBefore = dataX[:, 0, 0]
        dataYBefore = dataY[...].copy()
        timeBefore = time.time()
        numpy_extras.shuffle_hdf_arrays_together(dataX, dataY, blockSize=13)
        print("Shuffled in {:.2f} s.".format(time.time() - timeBefore))

        # Check that order has changed.
        self.assertFalse(np.all(np.equal(firstColumnBefore, dataX[:, 0, 0])),
                         msg='If we are extremely unlucky, the order might not change')

        # Check that the arrays are still in sync.
        for i in range(0, shapeX[0]):
            for j in range(0, shapeX[1]):
                simpleHash = np.sum(dataX[i, j, :].astype(np.int32)) % 73
                self.assertEqual(dataY[i, j], simpleHash)

        # Check that arrays have the same content.
        self.assertTrue(np.all(np.equal(np.sort(dataYBefore.flatten()),
                                        np.sort(dataY[...].flatten()))))

    def test_abs_diff_hdf_arrays(self):
        shape = (10000, 25, 25)

        tempDir = tempfile.mkdtemp()
        file = h5py.File(os.path.join(tempDir, 'temp.h5py'))
        dataA = file.create_dataset('tempA', shape, np.uint8)
        dataB = file.create_dataset('tempB', shape, np.uint8)
        out = file.create_dataset('out', shape, np.float32)

        dataA[...] = np.random.uniform(0, 255, shape)
        dataB[...] = np.random.uniform(0, 255, shape)
        out[...] = np.random.uniform(0, 255, shape)

        numpy_extras.abs_diff_hdf_arrays(dataA, dataB, out, np.float32, batchSizeFlat=3119)

        trueDiff = np.abs(dataA[...].astype(np.float32) - dataB[...].astype(np.float32))
        self.assertTrue(np.all(np.equal(out, trueDiff)))

    def test_abs_diff_hdf_arrays_masked(self):
        shape = (10000, 25, 25)

        tempDir = tempfile.mkdtemp()
        file = h5py.File(os.path.join(tempDir, 'temp.h5py'))
        dataA = file.create_dataset('tempA', shape, np.uint8)
        dataB = file.create_dataset('tempB', shape, np.uint8)
        mask = file.create_dataset('mask', shape, np.bool)
        out = file.create_dataset('out', shape, np.float32)

        dataA[...] = np.random.uniform(0, 255, shape)
        dataB[...] = np.random.uniform(0, 255, shape)
        mask[...] = False
        mask[:5000] = True
        out[...] = np.random.uniform(0, 255, shape)

        numpy_extras.abs_diff_hdf_arrays_masked(dataA, dataB, mask, out, np.float32, batchSizeFlat=3119)

        trueDiff = np.abs(dataA[...].astype(np.float32) - dataB[...].astype(np.float32))
        trueDiff[np.logical_not(mask)] = 0
        self.assertTrue(np.all(np.equal(out, trueDiff)))

    def test_mse_hdf_arrays(self):
        shape = (10000, 25, 25)

        tempDir = tempfile.mkdtemp()
        file = h5py.File(os.path.join(tempDir, 'temp.h5py'))
        dataA = file.create_dataset('tempA', shape, np.uint8)
        dataB = file.create_dataset('tempB', shape, np.uint8)

        dataA[...] = np.random.uniform(0, 255, shape)
        dataB[...] = np.random.uniform(0, 255, shape)

        # Test typical conditions.
        mse = numpy_extras.mse_hdf_arrays(dataA, dataB, np.float64, batchSizeFlat=7119)

        trueMse = np.mean(np.square(dataA[...].astype(np.float64) - dataB[...].astype(np.float64)), dtype=np.float64)
        self.assertAlmostEqual(mse, trueMse)

        # Test batches with an imbalanced number of elements.
        mse = numpy_extras.mse_hdf_arrays(dataA, dataB, np.float64, batchSizeFlat=8000 * 25 * 25 * 4)

        trueMse = np.mean(np.square(dataA[...].astype(np.float64) - dataB[...].astype(np.float64)), dtype=np.float64)
        self.assertAlmostEqual(mse, trueMse)

    def test_mse_hdf_arrays_masked(self):
        shape = (10000, 25, 25)

        tempDir = tempfile.mkdtemp()
        file = h5py.File(os.path.join(tempDir, 'temp.h5py'))
        dataA = file.create_dataset('tempA', shape, np.uint8)
        dataB = file.create_dataset('tempB', shape, np.uint8)
        mask = file.create_dataset('mask', shape, np.bool)

        dataA[...] = np.random.uniform(0, 255, shape)
        dataB[...] = np.random.uniform(0, 255, shape)
        mask[...] = False
        mask[:5000] = True

        # Test typical conditions.
        mse = numpy_extras.mse_hdf_arrays_masked(dataA, dataB, mask, np.float64, batchSizeFlat=7119)

        trueMse = np.mean(np.square(dataA[:5000].astype(np.float64) - dataB[:5000].astype(np.float64)),
                          dtype=np.float64)
        self.assertAlmostEqual(mse, trueMse)

        # Test batches with an imbalanced number of elements.
        mse = numpy_extras.mse_hdf_arrays_masked(dataA, dataB, mask, np.float64, batchSizeFlat=8000 * 25 * 25 * 4)

        trueMse = np.mean(np.square(dataA[:5000].astype(np.float64) - dataB[:5000].astype(np.float64)),
                          dtype=np.float64)
        self.assertAlmostEqual(mse, trueMse)


    def test_var_hdf_array(self):
        shape = (10000, 25, 25)

        tempDir = tempfile.mkdtemp()
        file = h5py.File(os.path.join(tempDir, 'temp.h5py'))
        data = file.create_dataset('tempA', shape, np.uint8)

        data[...] = np.random.uniform(0, 255, shape)

        # Test typical conditions.
        var = numpy_extras.var_hdf_array(data, np.float64, batchSizeFlat=7119)

        trueVar = np.mean(np.square(data[...].astype(np.float64)), dtype=np.float64) - \
                  np.mean(data[...].astype(np.float64), dtype=np.float64) ** 2
        self.assertAlmostEqual(var, trueVar)

        # Test batches with an imbalanced number of elements.
        var = numpy_extras.var_hdf_array(data, np.float64, batchSizeFlat=8000 * 25 * 25 * 4)

        trueVar = np.mean(np.square(data[...].astype(np.float64)), dtype=np.float64) - \
                  np.mean(data[...].astype(np.float64), dtype=np.float64) ** 2
        self.assertAlmostEqual(var, trueVar)

        data = file.create_dataset('tempB', (10, 1), np.uint8)
        data[...] = np.asarray([0, 0, 0, 0, 0, 10, 10, 10, 10, 10]).reshape(10, 1)

        var = numpy_extras.var_hdf_array(data, np.float64, batchSizeFlat=3119)

        self.assertAlmostEqual(var, 25.0)

    def test_var_hdf_array_masked(self):
        shape = (10000, 25, 15)

        tempDir = tempfile.mkdtemp()
        file = h5py.File(os.path.join(tempDir, 'temp.h5py'))
        data = file.create_dataset('tempA', shape, np.uint8)
        mask = file.create_dataset('maskA', shape, np.bool)

        data[...] = np.random.uniform(0, 255, shape)
        mask[...] = False
        mask[:5000] = True

        # Test typical conditions.
        var = numpy_extras.var_hdf_array_masked(data, mask, np.float64, batchSizeFlat=7119)

        trueVar = np.mean(np.square(data[:5000].astype(np.float64)), dtype=np.float64) - \
                  np.mean(data[:5000].astype(np.float64), dtype=np.float64) ** 2
        numpyVar = np.var(data[:5000], dtype=np.float64)
        self.assertAlmostEqual(var, trueVar)
        self.assertAlmostEqual(var, numpyVar)

        # Test batches with an imbalanced number of elements.
        # The first batch has 4000 nonzero slices, the second - only 1000.
        var = numpy_extras.var_hdf_array_masked(data, mask, np.float64, batchSizeFlat=4000 * 25 * 15 * 4)

        trueVar = np.mean(np.square(data[:5000].astype(np.float64)), dtype=np.float64) - \
                  np.mean(data[:5000].astype(np.float64), dtype=np.float64) ** 2
        self.assertAlmostEqual(var, trueVar)

        # Test a small easy to understand case.
        data = file.create_dataset('tempB', (10, 1), np.uint8)
        mask = file.create_dataset('maskB', (10, 1), np.bool)
        data[...] = np.asarray([0, 0, 0, 0, 66, 66, 10, 10, 10, 10]).reshape(10, 1)
        mask[...] = True
        mask[4:6] = False

        var = numpy_extras.var_hdf_array_masked(data, mask, np.float64, batchSizeFlat=3119)

        self.assertAlmostEqual(var, 25.0)

    def test_get_batch_indices(self):
        shape = (1000, 5, 7)
        dtype = np.float32
        batchSizeFlat = 32000
        expectedBatchSize = 228  # 32000 / (5 * 7 * 4)
        expectedIndices = [(0, 228), (228, 456), (456, 684), (684, 912), (912, 1000)]

        actualIndices = list(numpy_extras.get_batch_indices(shape, dtype, batchSizeFlat=batchSizeFlat))
        self.assertEqual(expectedIndices, actualIndices)

        # Test a very large batch.
        actualIndices = list(numpy_extras.get_batch_indices(shape, dtype, batchSizeFlat=1e10))
        self.assertEqual([(0, 1000)], actualIndices)

        # Test a batch that is too small.
        with self.assertRaises(RuntimeError):
            list(numpy_extras.get_batch_indices(shape, dtype, batchSizeFlat=10))

    def test_numpy_json_encoder(self):
        tempDir = tempfile.gettempdir()
        tempPath = os.path.join(tempDir, 'test_json_encoder.json')

        for dtype in [np.float32, np.float64, np.int32, np.int64, np.uint8]:
            array = np.ones(5, dtype)
            value = array[0]

            with open(tempPath, 'w') as file:  # Overwrite existing.
                json.dump({'key': value}, file, cls=numpy_extras.NumpyJsonEncoder)
            with open(tempPath, 'r') as file:
                contents = json.load(file)
                self.assertEqual(contents['key'], value)

    def test_moving_average_nd(self):
        data = np.array(
            [[0, 1, 2],
             [3, 4, 5],
             [6, 7, 8]])

        expected = np.array(
            [[ 8, 15, 12],
             [21, 36, 27],
             [20, 33, 24]]) / 9.0

        actual = numpy_extras.moving_average_nd(data, kernelSize=3)
        self.assertTrue(np.all(np.less(np.abs(actual - expected), 1e-5)))

        expected = np.array(
            [[0, 1, 3],
             [3, 8, 12],
             [9, 20, 24]]) / 4.0

        actual = numpy_extras.moving_average_nd(data, kernelSize=2)
        self.assertTrue(np.all(np.less(np.abs(actual - expected), 1e-5)))

    def test_sparse_insert_into_nd_array(self):
        tempDir = tempfile.gettempdir()
        tempPath = os.path.join(tempDir, 'test_sparse_insert_into_nd_array.raw')

        if os.path.exists(tempPath):
            os.unlink(tempPath)

        testCaseSize = 2000
        shape = (10, 64, 64, 64)
        dataSizeFlat = numpy_extras.multiply(shape)
        np.random.seed(1771)
        # Double the size, to make sure that the size limit is respected.
        indices = np.random.randint(0, 10, size=(2 * testCaseSize, len(shape)), dtype=np.uint64)
        values = np.random.randint(1, 255, size=(2 * testCaseSize, 1), dtype=np.uint8)

        with BufferedNdArray(tempPath, shape, np.uint8, int(1e5)) as array:
            CppWrapper.sparse_insert_into_nd_array(array.cPointer, indices, values, testCaseSize)

        with open(tempPath, 'rb') as file:
            rawData = np.fromfile(file, np.uint8, dataSizeFlat).reshape(shape)

        correctData = np.zeros(shape, dtype=np.uint8)
        # Repeat the same operations and compare the result, since some values may be overwritten.
        for i in range(testCaseSize):
            indexNd = tuple(indices[i, :])
            correctData[indexNd] = values[i, 0]

        for i in range(testCaseSize):
            indexNd = tuple(indices[i, :])
            self.assertEqual(correctData[indexNd], rawData[indexNd])

        # This test relies on the file being zeroed before writing, which afaik isn't guaranteed.
        self.assertEqual(np.count_nonzero(rawData), np.count_nonzero(correctData))

    def _get_random_hdf(self, shape: Tuple, dtype: type):
        tempDir = tempfile.mkdtemp()
        file = h5py.File(os.path.join(tempDir, 'temp.h5py'))
        dataset = file.create_dataset('temp', shape, dtype)
        dataset[...] = np.random.uniform(0, 1, shape)

        return dataset


if __name__ == '__main__':
    unittest.main()
