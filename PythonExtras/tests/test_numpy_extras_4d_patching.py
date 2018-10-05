import time
import math
import tempfile
import os
import json
from typing import Union, List, Tuple

import unittest
import h5py
import numpy as np

import PythonExtras.numpy_extras as npe
import PythonExtras.volume_tools as volume_tools
from PythonExtras.CppWrapper import CppWrapper


class NumpyExtras4DPatchingTest(unittest.TestCase):

    def setUp(self):
        super().setUp()

        self.data = np.arange(0, 3 * 3 * 3 * 3, dtype=np.uint8).reshape((3, 3, 3, 3))
        self.dataSizeFlat = np.prod(np.array(self.data.shape))  # type: int

        self.patchSize = (2, 2, 2, 2)
        self.patchStride = (2, 1, 1, 1)
        self.patchedAxes = (0, 1, 2, 3)

        self.patchNumber = npe.compute_patch_number(self.data.shape, self.patchedAxes, self.patchSize,
                                                    self.patchStride)
        self.patchSizeX = (self.patchSize[0] - 1,) + self.patchSize[1:]

        fullBatchSize = np.prod(np.asarray(self.patchNumber))  # type: int

        self.batchX = np.empty((fullBatchSize,) + self.patchSizeX, dtype=self.data.dtype)
        self.batchY = np.empty((fullBatchSize, 1), dtype=self.data.dtype)
        self.batchIndices = np.empty((fullBatchSize, self.data.ndim), dtype=np.uint64)

        self.manualResultFull = np.asarray(
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

    def setupComplexWithInnerStride(self):
        self.data = np.arange(0, 8 * 4 * 4 * 4, dtype=np.uint8).reshape((8, 4, 4, 4))
        self.dataSizeFlat = np.prod(np.array(self.data.shape))  # type: int

        self.patchSize = (3, 2, 2, 2)
        self.patchStride = (3, 3, 3, 3)
        self.patchedAxes = (0, 1, 2, 3)
        self.patchInnerStride = (2, 1, 1, 1)

        self.patchNumber = npe.compute_patch_number(self.data.shape, self.patchedAxes, self.patchSize,
                                                    self.patchStride, self.patchInnerStride)
        self.patchSizeX = (self.patchSize[0] - 1,) + self.patchSize[1:]

        fullBatchSize = np.prod(np.asarray(self.patchNumber))  # type: int

        self.batchX = np.empty((fullBatchSize,) + self.patchSizeX, dtype=self.data.dtype)
        self.batchY = np.empty((fullBatchSize, 1), dtype=self.data.dtype)
        self.batchIndices = np.empty((fullBatchSize, self.data.ndim), dtype=np.uint64)

        self.manualResultFull = np.asarray(
            [[[[[0, 1],
                [4, 5]],
               [[16, 17],
                [20, 21]]],
              [[[128, 129],
                [132, 133]],
               [[144, 145],
                [148, 149]]]],

             [[[[192, 193],
                [196, 197]],
               [[208, 209],
                [212, 213]]],
              [[[64, 65],
                 [68, 69]],
                [[80, 81],
                 [84, 85]]]]]
            , dtype=np.uint8)
        self.manualResultFullY = np.asarray([[213], [149]])

    def test_all_in_one_batch(self):

        batchSize = np.prod(np.asarray(self.patchNumber))  # type: int

        patchesExtracted, nextPatchIndex, inputEndReached = \
            npe.extract_patched_training_data_without_empty_4d(self.data, self.data.shape, 0,
                                                               self.dataSizeFlat, self.batchX, self.batchY,
                                                               self.batchIndices, self.patchSize,
                                                               self.patchStride, 0, batchSize,
                                                               skipEmptyPatches=False, emptyValue=0)

        self.assertEqual(self.manualResultFull.shape, self.batchX[:patchesExtracted].shape)
        self.assertTrue(np.all(np.equal(self.manualResultFull, self.batchX[:patchesExtracted])))

    def test_buffered_input(self):

        totalPatchNumber = np.prod(np.asarray(self.patchNumber))
        batchSize = totalPatchNumber  # type: int

        firstPatchEnd = npe.flatten_index(tuple(np.array(self.patchSize) - 1), self.data.shape)

        lastPatchYNd = (1, 2, 2, 2)
        lastPatchYFlat = npe.flatten_index(lastPatchYNd, self.data.shape)

        testCases = [
                     (0, firstPatchEnd + 1, 0, 1),                  # Only the first patch fits.
                     (0, lastPatchYFlat, 0, totalPatchNumber - 1),  # All but the last patch fit.
                     (0, self.dataSizeFlat, 0, totalPatchNumber),   # All patches fit.

                     (9, self.dataSizeFlat, 4, 4),  # Second half of the patches. (9 is the start of the fifth patch)
                     (13, self.dataSizeFlat, 7, 1)  # Only the last patch (it starts with 13).
                     ]

        for dataBufferStart, dataBufferEnd, batchStartIndex, patchesExpected in testCases:
            dataBuffer = self.data.ravel()[dataBufferStart:].copy()
            patchesExtracted, nextPatchIndex, inputEndReached = \
                npe.extract_patched_training_data_without_empty_4d(dataBuffer, self.data.shape,
                                                                   dataBufferStart, dataBufferEnd, self.batchX,
                                                                   self.batchY, self.batchIndices,
                                                                   self.patchSize, self.patchStride,
                                                                   batchStartIndex, batchSize,
                                                                   skipEmptyPatches=False, emptyValue=0)

            self.assertEqual(patchesExpected, patchesExtracted)
            self.assertEqual(patchesExpected != (totalPatchNumber - batchStartIndex), inputEndReached)
            self.assertEqual(self.manualResultFull[:patchesExpected].shape, self.batchX[:patchesExpected].shape)
            self.assertTrue(np.all(np.equal(self.manualResultFull[batchStartIndex:batchStartIndex + patchesExpected],
                                            self.batchX[:patchesExpected])))

    def test_nothing_processed_exception(self):
        batchSize = np.prod(np.asarray(self.patchNumber))  # type: int

        with self.assertRaises(RuntimeError):
            patchesExtracted, nextPatchIndex, inputEndReached = \
                npe.extract_patched_training_data_without_empty_4d(self.data, self.data.shape, 0, 1,
                                                                   self.batchX, self.batchY, self.batchIndices,
                                                                   self.patchSize, self.patchStride, 0,
                                                                   batchSize, skipEmptyPatches=False,
                                                                   emptyValue=0)

    def test_patch_inner_stride(self):

        self.setupComplexWithInnerStride()

        batchSize = np.prod(np.asarray(self.patchNumber))  # type: int

        patchesExtracted, nextPatchIndex, inputEndReached = \
            npe.extract_patched_training_data_without_empty_4d(self.data, self.data.shape, 0,
                                                               self.dataSizeFlat, self.batchX, self.batchY,
                                                               self.batchIndices, self.patchSize,
                                                               self.patchStride, 0, batchSize,
                                                               patchInnerStride=self.patchInnerStride,
                                                               skipEmptyPatches=False, emptyValue=0)

        self.assertEqual(self.manualResultFull.shape, self.batchX[:patchesExtracted].shape)
        self.assertTrue(np.all(np.equal(self.manualResultFull, self.batchX[:patchesExtracted])))
        self.assertTrue(np.all(np.equal(self.manualResultFullY, self.batchY[:patchesExtracted])))

    def test_last_frame_gap(self):
        self.setupComplexWithInnerStride()
        self.lastFrameGap = 2
        self.patchNumber = npe.compute_patch_number(self.data.shape, self.patchedAxes, self.patchSize,
                                                    self.patchStride, self.patchInnerStride,
                                                    lastFrameGap=self.lastFrameGap)

        batchSize = np.prod(np.asarray(self.patchNumber))  # type: int

        patchesExtracted, nextPatchIndex, inputEndReached = \
            npe.extract_patched_training_data_without_empty_4d(self.data, self.data.shape, 0,
                                                               self.dataSizeFlat, self.batchX, self.batchY,
                                                               self.batchIndices, self.patchSize,
                                                               self.patchStride, 0, batchSize,
                                                               patchInnerStride=self.patchInnerStride,
                                                               lastFrameGap=self.lastFrameGap,
                                                               skipEmptyPatches=False, emptyValue=0)

        self.assertEqual(self.manualResultFull.shape, self.batchX[:patchesExtracted].shape)
        self.assertTrue(np.all(np.equal(self.manualResultFull, self.batchX[:patchesExtracted])))
        # Use different Y values, since we changed the last frame gap.
        self.assertTrue(np.all(np.equal(np.asarray([[21], [213]]), self.batchY[:patchesExtracted])))


class NumpyExtras4DPatchingMultithreadedTest(unittest.TestCase):
    """
    Tests for the multithreaded version of the patching function.
    Because it's so cumbersome to create manual test cases (and hard to edit them later),
    the testing is performed against the single-threaded implementation, which is both covered
    by manual tests and has been in production for some time.

    """

    def setUp(self):
        super().setUp()

        # self.data = np.arange(0, 8 * 7 * 9 * 10, dtype=np.uint8).reshape((8, 7, 9, 10))
        # We have to use random values, otherwise patches repeat (due to overflow of uint8),
        # and it becomes hard to check the result for uniqueness.
        self.data = np.random.randint(0, 255, (8, 7, 9, 10), dtype=np.uint8)
        self.dataSizeFlat = npe.multiply(self.data.shape)  # type: int

        self.patchSize = (4, 2, 2, 2)
        self.patchStride = (1, 1, 1, 1)
        self.patchedAxes = (0, 1, 2, 3)
        self.patchInnerStride = (2, 1, 1, 1)
        self.lastFrameGap = 2

        self.patchNumber = npe.compute_patch_number(self.data.shape, self.patchedAxes, self.patchSize,
                                                    self.patchStride, patchInnerStride=self.patchInnerStride,
                                                    lastFrameGap=self.lastFrameGap)
        self.patchSizeX = (self.patchSize[0] - 1,) + self.patchSize[1:]

        self.fullBatchSize = np.prod(np.asarray(self.patchNumber))  # type: int
        self.setUpBuffers(self.fullBatchSize)

        self.dataStartFlat = 0
        self.dataEndFlat = self.dataSizeFlat

        self.batchStartIndex = 0
        self.batchSize = self.fullBatchSize

    def setUpBuffers(self, batchSize):
        self.batchX = np.empty((batchSize,) + self.patchSizeX, dtype=self.data.dtype)
        self.batchY = np.empty((batchSize, 1), dtype=self.data.dtype)
        self.batchIndices = np.empty((batchSize, self.data.ndim), dtype=np.uint64)

    def assertIdenticalToReference(self, data: np.ndarray = None, dataShape: Tuple = None, dataStartFlat: int = None,
                                   dataEndFlat: int = None, outputX: np.ndarray = None, outputY: np.ndarray = None,
                                   outputIndices: np.ndarray = None, patchSize: Tuple = None, patchStride: Tuple = None,
                                   batchStartIndex: int = None, batchSize: int = None,
                                   patchInnerStride: Tuple= None,
                                   lastFrameGap: int = None,
                                   undersamplingProb: float = None,
                                   skipEmptyPatches: bool = None,
                                   emptyValue: int = None):
        """
        Runs the reference single-threaded and the target multithreaded implementations
        and compares their results.
        Arguments can be provided to override the default values.
        """
        
        outputX = outputX if outputX is not None else self.batchX
        outputY = outputY if outputY is not None else self.batchY
        outputIndices = outputIndices if outputIndices is not None else self.batchIndices

        outputX[...] = 0
        outputY[...] = 0
        outputIndices[...] = 0

        # Run the reference single-threaded implementation.
        patchesExtractedRef, nextPatchIndexRef, inputEndReachedRef = \
            npe.extract_patched_training_data_without_empty_4d(
                data if data is not None else self.data,
                dataShape or self.data.shape,
                dataStartFlat or self.dataStartFlat,
                dataEndFlat or self.dataEndFlat,
                outputX,
                outputY,
                outputIndices,
                patchSize or self.patchSize,
                patchStride or self.patchStride,
                batchStartIndex or self.batchStartIndex,
                batchSize or self.batchSize,
                patchInnerStride=patchInnerStride or self.patchInnerStride,
                lastFrameGap=lastFrameGap or self.lastFrameGap,
                skipEmptyPatches=skipEmptyPatches or False,
                emptyValue=emptyValue or 0
            )

        self.referenceX = outputX.copy()
        self.referenceY = outputY.copy()
        self.referenceIndices = outputIndices.copy()

        outputX[...] = 0
        outputY[...] = 0
        outputIndices[...] = 0

        # Run the multithreaded test target.
        patchesExtracted, nextPatchIndex, inputEndReached = \
            npe.extract_patched_training_data_without_empty_4d_multi(
                data if data is not None else self.data,
                dataShape or self.data.shape,
                dataStartFlat or self.dataStartFlat,
                dataEndFlat or self.dataEndFlat,
                outputX,
                outputY,
                outputIndices,
                patchSize or self.patchSize,
                patchStride or self.patchStride,
                batchStartIndex or self.batchStartIndex,
                batchSize or self.batchSize,
                patchInnerStride=patchInnerStride or self.patchInnerStride,
                lastFrameGap=lastFrameGap or self.lastFrameGap,
                skipEmptyPatches=skipEmptyPatches or False,
                emptyValue=emptyValue or 0,
                threadNumber=8)

        self.assertEqual(patchesExtractedRef, patchesExtracted)
        self.assertEqual(nextPatchIndexRef, nextPatchIndexRef)
        self.assertEqual(inputEndReachedRef, inputEndReached)

        # The multithreaded function output can be shuffled, so don't compare directly,
        # but check that every patch is present in the reference output.
        # Since we check that the same number of patches is extracted and
        # the input data is random, we only need to check one way.
        for i in range(patchesExtracted):
            foundMatch = False
            for j in range(patchesExtracted):
                if np.all(np.equal(self.referenceX[i, ...], outputX[j, ...])):
                    np.testing.assert_array_equal(self.referenceY[i, ...], outputY[j, ...])
                    np.testing.assert_array_equal(self.referenceIndices[i, ...], outputIndices[j, ...])

                    foundMatch = True
                    break

            self.assertTrue(foundMatch)

        return patchesExtracted, nextPatchIndex, inputEndReached

    def test_all_in_one_batch(self):

        patchesExtracted, nextPatchIndex, inputEndReached = \
            self.assertIdenticalToReference(batchSize=self.fullBatchSize)

        self.assertEqual(self.fullBatchSize, patchesExtracted)

    def test_buffered_input(self):
        """
        Reworked the test for the single-threaded patching function.
        :return:
        """

        totalPatchNumber = npe.multiply(self.patchNumber)
        batchSize = totalPatchNumber  # type: int

        patchExtent = [self.patchSize[dim] * self.patchInnerStride[dim] for dim in range(len(self.patchSize))]
        patchExtent[0] += self.lastFrameGap - self.patchInnerStride[0]

        firstPatchEnd = npe.flatten_index(tuple(np.array(patchExtent) - 1), self.data.shape)

        lastPatchLower = npe.patch_index_to_data_index(totalPatchNumber - 1, self.data.shape, (0, 1, 2, 3),
                                                       self.patchSize, self.patchStride)

        lastPatchYNd = np.asarray(lastPatchLower) + np.asarray(patchExtent)
        lastPatchYNd[0] -= 1
        lastPatchYFlat = npe.flatten_index(lastPatchYNd, self.data.shape)

        testCases = [
                     (0, firstPatchEnd + 1, 0, 1),                  # Only the first patch fits.
                     (0, lastPatchYFlat, 0, totalPatchNumber - 1),  # All but the last patch fit.
                     (0, self.dataSizeFlat, 0, totalPatchNumber),   # All patches fit.

                     (9, self.dataSizeFlat, 4, 4),  # These test cases don't make much sense (9 and 13 are arbitrary),
                     (13, self.dataSizeFlat, 7, 1)  # but we just check that outputs are identical.
                     ]

        for dataBufferStart, dataBufferEnd, batchStartIndex, patchesExpected in testCases:
            dataBuffer = self.data.ravel()[dataBufferStart:].copy()

            self.assertIdenticalToReference(data=dataBuffer,
                                            dataStartFlat=dataBufferStart, dataEndFlat=dataBufferEnd,
                                            batchStartIndex=batchStartIndex, batchSize=batchSize)


class NumpyExtras4DPatchingMultithreadedLargeScaleTest(unittest.TestCase):

    def setUp(self):
        super().setUp()

        # We have to use random values, otherwise patches repeat (due to overflow of uint8),
        # and it becomes hard to check the result for uniqueness.
        self.dataShape = (300, 32, 32, 32)
        np.random.seed(25620322)
        self.data = np.random.randint(0, 255, self.dataShape, dtype=np.uint8)
        self.dataSizeFlat = npe.multiply(self.dataShape)  # type: int

        self.patchSize = (4, 2, 2, 2)
        self.patchStride = (1, 1, 1, 1)
        self.patchedAxes = (0, 1, 2, 3)
        self.patchInnerStride = (2, 1, 1, 1)
        self.lastFrameGap = 4

        self.skipEmptyPatches = False
        self.emptyValue = 0

        self.patchNumber = npe.compute_patch_number(self.dataShape, self.patchedAxes, self.patchSize,
                                                    self.patchStride, patchInnerStride=self.patchInnerStride,
                                                    lastFrameGap=self.lastFrameGap)
        self.patchNumberFlat = npe.multiply(self.patchNumber)

        self.patchSizeX = (self.patchSize[0] - 1,) + self.patchSize[1:]

        self.fullBatchSize = np.prod(np.asarray(self.patchNumber))  # type: int
        self.batchSize = int(2e6)
        self.setUpBuffers(self.batchSize)

        self.dataStartFlat = 0
        self.dataEndFlat = self.dataSizeFlat

        self.batchStartIndex = 0

    def setUpBuffers(self, batchSize):
        self.batchX = np.empty((batchSize,) + self.patchSizeX, dtype=self.data.dtype)
        self.batchY = np.empty((batchSize, 1), dtype=self.data.dtype)
        self.batchIndices = np.empty((batchSize, self.data.ndim), dtype=np.uint64)

    def assertIdenticalToReference(self, data: np.ndarray = None, dataShape: Tuple = None, dataStartFlat: int = None,
                                   dataEndFlat: int = None, outputX: np.ndarray = None, outputY: np.ndarray = None,
                                   outputIndices: np.ndarray = None, patchSize: Tuple = None, patchStride: Tuple = None,
                                   batchStartIndex: int = None, batchSize: int = None,
                                   patchInnerStride: Tuple = None,
                                   lastFrameGap: int = None,
                                   undersamplingProb: float = None,
                                   skipEmptyPatches: bool = None,
                                   emptyValue: int = None):
        """
        Runs the reference single-threaded and the target multithreaded implementations
        and compares their results.
        Arguments can be provided to override the default values.
        """

        if skipEmptyPatches or self.skipEmptyPatches:
            # This is because the single-threaded code extracts 'batch-size' patches, stepping over empty ones,
            # but the multithreaded code checks 'batch-size' patches, potentially extracting less.
            # In the future this can be checked by checking output over the whole dataset.
            raise RuntimeError("Empty patch skipping cannot be tested against a reference implementation.")

        outputX = outputX if outputX is not None else self.batchX
        outputY = outputY if outputY is not None else self.batchY
        outputIndices = outputIndices if outputIndices is not None else self.batchIndices

        outputX[...] = 0
        outputY[...] = 0
        outputIndices[...] = 0

        # Run the reference single-threaded implementation.
        patchesExtractedRef, nextPatchIndexRef, inputEndReachedRef = \
            npe.extract_patched_training_data_without_empty_4d(
                data if data is not None else self.data,
                dataShape or self.data.shape,
                dataStartFlat or self.dataStartFlat,
                dataEndFlat or self.dataEndFlat,
                outputX,
                outputY,
                outputIndices,
                patchSize or self.patchSize,
                patchStride or self.patchStride,
                batchStartIndex or self.batchStartIndex,
                batchSize or self.batchSize,
                patchInnerStride=patchInnerStride or self.patchInnerStride,
                lastFrameGap=lastFrameGap or self.lastFrameGap,
                skipEmptyPatches=skipEmptyPatches or self.skipEmptyPatches,
                emptyValue=emptyValue or self.emptyValue
            )

        self.referenceX = outputX.copy()
        self.referenceY = outputY.copy()
        self.referenceIndices = outputIndices.copy()

        outputX[...] = 0
        outputY[...] = 0
        outputIndices[...] = 0

        # Run the multithreaded test target.
        patchesExtracted, nextPatchIndex, inputEndReached = \
            npe.extract_patched_training_data_without_empty_4d_multi(
                data if data is not None else self.data,
                dataShape or self.data.shape,
                dataStartFlat or self.dataStartFlat,
                dataEndFlat or self.dataEndFlat,
                outputX,
                outputY,
                outputIndices,
                patchSize or self.patchSize,
                patchStride or self.patchStride,
                batchStartIndex or self.batchStartIndex,
                batchSize or self.batchSize,
                patchInnerStride=patchInnerStride or self.patchInnerStride,
                lastFrameGap=lastFrameGap or self.lastFrameGap,
                skipEmptyPatches=skipEmptyPatches or self.skipEmptyPatches,
                emptyValue=emptyValue or self.emptyValue,
                threadNumber=8)

        self.assertEqual(patchesExtractedRef, patchesExtracted)
        self.assertEqual(nextPatchIndexRef, nextPatchIndexRef)
        self.assertEqual(inputEndReachedRef, inputEndReached)

        # The multithreaded function output can be shuffled, so don't compare directly,
        # but check that every patch is present in the reference output.
        # Since we check that the same number of patches is extracted and
        # the input data is random, we only need to check one way.

        # We can't check all the patches, this would take too long.
        indicesToCheck = [
            0,
            patchesExtracted * 1 / 4,
            patchesExtracted * 2 / 4,
            patchesExtracted * 3 / 4,
            patchesExtracted - 2,
            patchesExtracted - 1
        ]

        for i in indicesToCheck:
            i = int(i)
            if i >= patchesExtracted:
                continue

            foundMatch = False
            for j in range(patchesExtracted):
                if np.all(np.equal(self.referenceX[i, ...], outputX[j, ...])):

                    np.testing.assert_array_equal(self.referenceY[i, ...], outputY[j, ...])
                    np.testing.assert_array_equal(self.referenceIndices[i, ...], outputIndices[j, ...])

                    foundMatch = True
                    break

            self.assertTrue(foundMatch, msg="Patch index: {} \n {}".format(batchStartIndex + i, self.referenceX[i]))

        return patchesExtracted, nextPatchIndex, inputEndReached

    def test_full_identical_output(self):

        nextPatchIndex = 0
        while nextPatchIndex < self.patchNumberFlat:
            print("{} / {} = {:.2f}%".format(nextPatchIndex, self.patchNumberFlat, nextPatchIndex / self.patchNumberFlat * 100))
            patchesExtracted, nextPatchIndex, inputEndReached = \
                self.assertIdenticalToReference(batchStartIndex=nextPatchIndex)
