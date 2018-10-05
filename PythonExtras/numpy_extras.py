import random
import json
import math
import time
import logging
from typing import List, Tuple, Union, Callable, Iterable

import numpy as np
import scipy.signal
import h5py

from PythonExtras.CppWrapper import CppWrapper


def multiply(iterable: Iterable):
    prod = 1
    for x in iterable:
        prod *= x
    return prod


def slice_len(sliceObject: slice, sequenceLength: int):
    return len(range(*sliceObject.indices(sequenceLength)))


def slice_nd(start, end):
    """
    Take a slice of an nd-array along all axes simultaneously..
    :param start:
    :param end:
    :return:
    """
    assert len(start) == len(end)

    result = []
    for i in range(0, len(start)):
        result.append(slice(start[i], end[i]))

    return tuple(result)


def slice_along_axis(index, axis, ndim):
    """
    Return a selector that takes a single-element slice (subtensor) of an nd-array
    along a certain axis (at a given index).
    The result would be an (n-1)-dimensional array. E.g. data[:, :, 5, :].
    The advantage of this function over subscript syntax is that you can
    specify the axis with a variable.

    :param index:
    :param axis:
    :param ndim:
    :return:
    """
    return tuple([index if axis == i else None for i in range(0, ndim)])


def weighted_mean(data, weights, axis=None):

    assert axis == 0
    assert weights.ndim == 1

    weightSum = np.sum(weights)
    weightsNorm = weights / weightSum
    return np.sum(data * weightsNorm[:, np.newaxis], axis=0)


def weighted_variance(data, weights, axis=None, weightedMean=None):
    assert axis == 0
    assert weights.ndim == 1

    if weightedMean is None:
        weightedMean = weighted_mean(data, weights, axis=axis)

    weightSum = np.sum(weights)
    weightsNorm = weights / weightSum

    squaredDeviation = (data - weightedMean) ** 2
    return np.sum(squaredDeviation * weightsNorm[:, np.newaxis], axis=0)


def weighted_std(data, weights, axis=None, weightedMean=None):
    return np.sqrt(weighted_variance(data, weights, axis, weightedMean))


def moving_average_nd(data, kernelSize):
    """
    Perform local averaging of an Nd-array.
    Since averaging kernel is separable, convolution is applied iteratively along each axis..

    :param data:
    :param kernelSize:
    :return:
    """

    kernels = []
    for dim in range(data.ndim):
        kernelShape = tuple((min(kernelSize, data.shape[dim]) if dim == i else 1 for i in range(data.ndim)))
        kernels.append(np.ones(kernelShape) / kernelSize)

    result = data.copy()  # Doesn't work correctly in-place.
    for kernel in kernels:
        result = scipy.signal.convolve(result, kernel, 'same')

    return result


def argsort2d(data):
    """
    Return an array of indices into a sorted version of a 2d-array.
    :param data:
    :return:
    """
    assert(data.ndim == 2)
    origShape = data.shape
    dataFlat = data.reshape(-1)
    indicesSorted = np.argsort(dataFlat)
    indicesMulti = [np.unravel_index(i, origShape) for i in indicesSorted.tolist()]

    return indicesMulti


def unflatten_index(shape, indexFlat):
    """
    Convert a flat index into an N-d index (a tuple).
    :param shape:
    :param indexFlat:
    :return:
    """
    indexNd = tuple()
    for axis, length in enumerate(shape):
        sliceSize = np.prod(shape[axis + 1:], dtype=np.int64)
        axisIndex = int(indexFlat / sliceSize)
        indexNd += (axisIndex,)
        indexFlat -= axisIndex * sliceSize

    return indexNd


def flatten_index(indexTuple, size):
    """
    Converts a Nd index into a flat array index. C order is assumed.
    (The point is that it works with any number of dimensions.)
    :param indexTuple:
    :param size:
    :return:
    """
    ndim = len(size)
    assert(ndim == len(indexTuple))

    sliceSizes = np.empty(ndim, type(size[0]))
    sliceSizes[-1] = 1
    for i in range(ndim - 2, -1, -1):
        sliceSizes[i] = size[i + 1] * sliceSizes[i + 1]

    flatIndex = 0
    for i in range(0, ndim):
        flatIndex += indexTuple[i] * sliceSizes[i]

    return flatIndex


def extract_patches_slow(data: Union[np.ndarray, h5py.Dataset], sourceAxes: Tuple[int], patchSize: Tuple,
                         patchStride: Tuple = None, verbose=False):
    """
    The same method, but it's old non-Cpp version.

    :param data:
    :param sourceAxes:
    :param patchSize:
    :param patchStride:
    :param verbose:
    :return:
    """
    # todo Implement C++ support for different dtypes.

    # By default, extract every patch.
    if patchStride is None:
        patchStride = (1,) * len(sourceAxes)

    result = None

    patchCenters = []  # Store geometrical center of each path. (Can be useful for the caller.)
    patchNumber = compute_patch_number(data.shape, sourceAxes, patchSize, patchStride)
    patchNumberFlat = np.prod(np.asarray(patchNumber), dtype=np.int64)  # type: int

    i = 0
    for patch, patchCenter, patchIndex in extract_patches_gen(data, sourceAxes, patchSize, patchStride,
                                                              verbose):
        if result is None:
            resultShape = (patchNumberFlat,) + patch.shape
            result = np.empty(resultShape, dtype=patch.dtype)
        result[i, ...] = patch
        patchCenters.append(patchCenter)
        i += 1

    return result, patchCenters, patchNumber


def extract_patches(data: Union[np.ndarray, h5py.Dataset], sourceAxes: Tuple, patchSize: Tuple,
                    patchStride: Tuple = None, batchSize=0, verbose=False):
    """
    Extract local patches/windows from the data.
    Source axes specify the dimensions that get cut into patches.
    The source axes get removed, a new axis enumerating patches is added as the first axis.

    E.g. split an 5-channel 2D image into a set of local 2D 5-channel patches;
    or split a dynamic volume into spatiotemporal windows.

    :param data:
    :param sourceAxes: Indices of the dimensions that get cut into slices.
    :param patchSize: Size of a single patch, including skipped axes.
    :param patchStride: Distance in data space between neighboring patches.
    :param batchSize:
    :param verbose: Whether to output progress information.
    :return:
    """

    return CppWrapper.extract_patches(data, sourceAxes, patchSize, patchStride, batchSize, verbose)


def extract_patches_batch(data: np.ndarray, sourceAxes: Tuple, patchSize: Tuple,
                          patchStride: Tuple = None, batchStart: int = 0, batchSize: int = 0):
    """
    The same as normal extract_patches, but extracts a single batch of patches.
    :param batchStart: First patch index in the batch.
    :param batchSize: The number of patches in the batch.
    :return:
    """

    return CppWrapper.extract_patches_batch(data, sourceAxes, patchSize, patchStride, batchStart, batchSize)


def compute_patch_number(dataShape: Tuple, sourceAxes: Tuple, patchSize: Tuple,
                         patchStride: Tuple = None, patchInnerStride: Tuple = None,
                         lastFrameGap: int=1):
    if patchStride is None:
        patchStride = (1,) * len(dataShape)
    if patchInnerStride is None:
        patchInnerStride = (1,) * len(dataShape)

    patchNumber = []
    for i, axis in enumerate(sourceAxes):
        # How many voxels a patch covers.
        if i > 0:
            patchSupport = (patchSize[i] - 1) * patchInnerStride[i] + 1
        else:
            # Last point in time (Y-value) is 'lastFrameGap' frames away from the previous frame.
            # E.g. if 'lastFrameGap' is 1, it immediately follows it.
            patchSupport = (patchSize[i] - 2) * patchInnerStride[i] + 1 + lastFrameGap
        totalPatchNumber = dataShape[axis] - patchSupport + 1
        stride = patchStride[i]
        patchNumber.append(int(math.ceil(totalPatchNumber / stride)))

    return patchNumber


def get_prediction_domain(dataShape: Tuple,
                          patchSize: Tuple,
                          patchInnerStride: Tuple,
                          lastFrameGap: int):
    """
    Given patching parameters, computes the 'predictable' area in the data, i.e.
    voxels which correspond to valid input patches.

    :param dataShape:
    :param patchSize:
    :param patchInnerStride:
    :param lastFrameGap:
    :return:
    """

    patchSize = np.asarray(patchSize)
    patchInnerStride = np.asarray(patchInnerStride)

    lowSpace = patchSize[1:4] // 2 * patchInnerStride[1:4]  # type: np.ndarray
    highSpace = np.asarray(dataShape[1:4]) - (patchSize[1:4] - patchSize[1:4] // 2) * patchInnerStride[1:4] + 1

    lowTime = (patchSize[0] - 2) * patchInnerStride[0] + lastFrameGap
    highTime = dataShape[0]

    # Also, convert to native Python ints from np.int.
    low = (int(lowTime), ) + tuple([np.asscalar(x) for x in lowSpace])
    high = (int(highTime), ) + tuple([np.asscalar(x) for x in highSpace])

    return (low, high)


def tuple_corners_to_slices(low: Tuple, high: Tuple) -> Tuple:
    """
    Convert two tuples representing low and high corners of an nd-box
    into a tuple of slices that can be used to select that box out of a volume.

    :param low:
    :param high:
    :return:
    """

    return tuple(slice(low[dim], high[dim]) for dim in range(len(low)))


def patch_index_to_data_index(patchIndexFlat: int, dataShape: Tuple, sourceAxes: Tuple,
                              patchSize: Tuple, patchStride: Tuple):
    """
    Convert a flat patch index into a data index which points to the lowest corner of the patch.
    Note: We don't need the inner stride or the last frame gap, because the lower corner
    doesn't depend on them.

    :param patchIndexFlat:
    :param dataShape:
    :param sourceAxes:
    :param patchSize:
    :param patchStride:
    :return:
    """

    ndim = len(dataShape)
    assert(sourceAxes == tuple(range(ndim)))  # Axis skipping is not implemented.

    patchNumber = compute_patch_number(dataShape, sourceAxes, patchSize, patchStride)
    patchIndexNd = unflatten_index(patchNumber, patchIndexFlat)

    return tuple((patchIndexNd[dim] * patchStride[dim] for dim in range(ndim)))


def extract_patches_gen(data: Union[np.ndarray, h5py.Dataset], sourceAxes: Tuple[int], patchSize: Tuple,
                        patchStride: Tuple = None, verbose=False):
    """
    The same method but as a generator function.

    :param data:
    :param sourceAxes:
    :param patchSize:
    :param patchStride:
    :param verbose:
    :return:
    """
    # By default, extract every patch.
    if patchStride is None:
        patchStride = (1,) * len(sourceAxes)

    patchNumber = compute_patch_number(data.shape, sourceAxes, patchSize, patchStride)
    patchNumberFlat = np.prod(np.asarray(patchNumber), dtype=np.int64)

    lastPrintTime = time.time()

    # Since the number of traversed dimension is dynamically specified, we cannot use 'for' loops.
    # Instead, iterate a flat index and unflatten it inside the loop. (Easier than recursion.)
    for indexFlat in range(patchNumberFlat):
        patchIndexNd = unflatten_index(patchNumber, indexFlat)
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


def extract_patched_training_data_without_empty_4d(data: np.ndarray, dataShape: Tuple, dataStartFlat: int,
                                                   dataEndFlat: int, outputX: np.ndarray, outputY: np.ndarray,
                                                   outputIndices: np.ndarray, patchSize: Tuple, patchStride: Tuple,
                                                   batchStartIndex: int, batchSize: int,
                                                   patchInnerStride: Tuple = (1, 1, 1, 1),
                                                   lastFrameGap: int = 1,
                                                   undersamplingProb: float = 1.0,
                                                   skipEmptyPatches: bool = True, emptyValue: int = 0):
    """
    Extract patches/windows from a 4-dimensional array.
    Each patch gets split into training data: X and Y.
    X holds the whole hypercube, except for the last frame. Y holds a single scalar
    from the center of the last frame. (Time is the first dimension, C-order is assumed.)
    'Empty' patches are those, where all values in X and the Y value are equal to the 'empty value'.
    Empty patches do not get extracted.
    Extraction is performed in batches, returning control after 'batchSize' patches were extracted.
    """
    return CppWrapper.extract_patched_training_data_without_empty_4d(data, dataShape, dataStartFlat, dataEndFlat,
                                                                     outputX, outputY, outputIndices, patchSize,
                                                                     patchStride, batchStartIndex, batchSize,
                                                                     patchInnerStride=patchInnerStride,
                                                                     lastFrameGap=lastFrameGap,
                                                                     undersamplingProb=undersamplingProb,
                                                                     skipEmptyPatches=skipEmptyPatches,
                                                                     emptyValue=emptyValue)


def extract_patched_training_data_without_empty_4d_multi(data: np.ndarray, dataShape: Tuple, dataStartFlat: int,
                                                         dataEndFlat: int, outputX: np.ndarray, outputY: np.ndarray,
                                                         outputIndices: np.ndarray, patchSize: Tuple,
                                                         patchStride: Tuple,
                                                         batchStartIndex: int, batchSize: int,
                                                         patchInnerStride: Tuple = (1, 1, 1, 1),
                                                         lastFrameGap: int = 1,
                                                         undersamplingProb: float = 1.0,
                                                         skipEmptyPatches: bool = True, emptyValue: int = 0,
                                                         threadNumber: int = 8):
    """
    Same as 'extract_patched_training_data_without_empty_4d', but multithreaded.
    Because of this, the output patches aren't in any particular order,
    though all the patches are guaranteed to be extracted.
    """

    return CppWrapper.extract_patched_training_data_without_empty_4d_multi(data, dataShape, dataStartFlat, dataEndFlat,
                                                                           outputX, outputY, outputIndices,
                                                                           patchSize, patchStride,
                                                                           batchStartIndex, batchSize,
                                                                           patchInnerStride=patchInnerStride,
                                                                           lastFrameGap=lastFrameGap,
                                                                           undersamplingProb=undersamplingProb,
                                                                           skipEmptyPatches=skipEmptyPatches,
                                                                           emptyValue=emptyValue,
                                                                           threadNumber=threadNumber)


def extract_patched_training_data_4d_multithreaded(data: np.ndarray, dataShape: Tuple, dataStartFlat: int,
                                                   dataEndFlat: int, outputX: np.ndarray, outputY: np.ndarray,
                                                   patchSize: Tuple, batchStartIndex: int, batchSize: int,
                                                   patchInnerStride: Tuple = (1, 1, 1, 1), lastFrameGap: int = 1,
                                                   threadNumber: int = 8):
    """
    Same as 'extract_patched_training_data_without_empty_4d', but without empty patch skipping
    which allows multiple threads to be used for extraction, while maintaining correct output order.
    """
    return CppWrapper.extract_patched_training_data_4d_multithreaded(data, dataShape, dataStartFlat, dataEndFlat, outputX,
                                                                     outputY, patchSize,
                                                                     batchStartIndex, batchSize,
                                                                     patchInnerStride=patchInnerStride,
                                                                     lastFrameGap=lastFrameGap,
                                                                     threadNumber=threadNumber)


def extract_patched_all_data_without_empty_4d(data: np.ndarray,
                                              patchSize: Tuple, patchStride: Tuple, emptyValue: int,
                                              batchSize=10000):
    """
    A wrapper around 'extract_patched_training_data_without_empty_4d', takes care of managing
    batching, extracting all available data in one call.
    Also serves as a documentation for how to use the single-batch method.

    :param data:
    :param patchSize:
    :param patchStride:
    :param emptyValue:
    :param batchSize:
    :return:
    """
    dataSizeFlat = multiply(data.shape) * data.dtype.itemsize
    batchSize = min(batchSize, data.size)

    patchSizeX = (patchSize[0] - 1,) + patchSize[1:]

    allDataX = NumpyDynArray((-1,) + patchSizeX, dtype=data.dtype)
    allDataY = NumpyDynArray((-1, 1), dtype=data.dtype)
    allDataIndices = NumpyDynArray((-1, data.ndim), dtype=np.uint64)

    batchX = np.empty((batchSize,) + patchSizeX, dtype=data.dtype)
    batchY = np.empty((batchSize, 1), dtype=data.dtype)
    batchIndices = np.empty((batchSize, data.ndim), dtype=np.uint64)

    patchesExtracted = batchSize
    nextBatchIndex = 0
    while patchesExtracted == batchSize:
        patchesExtracted, nextBatchIndex, dataEndReached = \
            extract_patched_training_data_without_empty_4d(data, data.shape, 0, dataSizeFlat, batchX, batchY,
                                                           batchIndices, patchSize, patchStride, nextBatchIndex,
                                                           batchSize, skipEmptyPatches=True, emptyValue=emptyValue)

        allDataX.append_all(batchX[:patchesExtracted])
        allDataY.append_all(batchY[:patchesExtracted])
        allDataIndices.append_all(batchIndices[:patchesExtracted])

    return allDataX.get_all(), allDataY.get_all(), allDataIndices.get_all()


def get_batch_indices(shape: Tuple, dtype: np.dtype, batchSizeFlat=1e8):
    """
    Batching helper. Returns a pair (start and end) of indices for each batch.
    Batches along axis 0.

    :param shape:
    :param dtype:
    :param batchSizeFlat:
    :return:
    """
    if type(dtype) != np.dtype:
        # noinspection PyCallingNonCallable
        dtype = dtype()  # Instantiate the dtype class.

    sliceSizeFlat = np.prod(np.asarray(shape[1:]), dtype=np.int64) * dtype.itemsize
    batchSize = int(batchSizeFlat // sliceSizeFlat)
    batchSize = min(batchSize, shape[0])

    if batchSize == 0:
        raise RuntimeError("Batch size '{}' is too small to fit a volume slice of size '{}'"
                           .format(batchSizeFlat, sliceSizeFlat))

    for batchStart in range(0, shape[0], batchSize):
        batchEnd = batchStart + min(batchSize, shape[0] - batchStart)
        yield (batchStart, batchEnd)


def shuffle_hdf_arrays_together(dataA: h5py.Dataset, dataB: h5py.Dataset, blockSize: int=1, logger: logging.Logger=None):
    """
    Shuffle rows of two HDF array in sync: corresponding rows end up at the same place.
    Used for shuffling X and Y training data arrays.
    :param dataA:
    :param dataB:
    :param blockSize: Specifies the number of rows that will be moved together as a single element.
                      Rows won't be shuffled inside of a block.
                      Set to one for true shuffling, use a larger value to perform an approximate shuffle.
    :param logger:
    :return:
    """
    # Explicitly make sure we have an HDF dataset, not a numpy array, since we make some assumptions based on it.
    if not hasattr(dataA, 'id') or not hasattr(dataB, 'id'):  # Don't check the type, it's some weird internal class.
        raise ValueError('Provided input is not in HDF datasets.')

    if dataA.shape[0] != dataB.shape[0]:
        raise ValueError('Arguments should have equal length, {} and {} given.'.format(dataA.shape[0], dataB.shape[0]))

    timeStart, timeLastReport = time.time(), time.time()

    length = dataA.shape[0]
    blockNumber = length // blockSize
    for iSource in range(0, blockNumber - 2):  # First to next-to-last. (Simplifies the block size check.)
        iTarget = int(random.uniform(iSource, blockNumber - 1e-10))  # [i, blockNumber)
        # The last block might need to be shortened, if array length is now divisible by the block size.
        actualBlockSize = blockSize if iTarget < blockNumber - 1 else length - iTarget * blockSize

        sourceSelector = slice(iSource * blockSize, iSource * blockSize + actualBlockSize)
        targetSelector = slice(iTarget * blockSize, iTarget * blockSize + actualBlockSize)

        # Swap data in array A.
        temp = dataA[targetSelector]  # This creates a copy, since we read from HDF, no need to do it explicitly.
        dataA[targetSelector] = dataA[sourceSelector]
        dataA[sourceSelector] = temp

        # Swap data synchronously in array B.
        temp = dataB[targetSelector]
        dataB[targetSelector] = dataB[sourceSelector]
        dataB[sourceSelector] = temp

        # Report current progress, but not too often. Use index modulo as heuristic to not query time module often.
        if logger is not None and iSource % 1000 == 0:
            timeCurrent = time.time()
            if timeCurrent - timeLastReport > 60:
                timeLastReport = timeCurrent
                logger.info("Shuffling an HDF array ...{:.2f}% in {:.2f}s."
                            .format(iSource / blockNumber * 100, timeCurrent - timeStart))


def abs_diff_hdf_arrays(dataA: h5py.Dataset, dataB: h5py.Dataset, output: h5py.Dataset,
                        dtype: np.dtype, batchSizeFlat=1e8):
    """
    Compute element-wise |A-B|.
    Computation is performed in batches to decrease memory requirements.

    :param dtype:
    :param batchSizeFlat:
    :param dataA:
    :param dataB:
    :param output:
    :return:
    """
    if dataA.shape != dataB.shape or dataA.shape != output.shape:
        raise ValueError("Arguments should have equal shapes, {}, {} and {} given."
                         .format(dataA.shape, dataB.shape, output.shape))
    if output.dtype != dtype:
        raise ValueError("Output array data type doesn't match the provided type: {} and {}"
                         .format(output.dtype, dtype))

    for batchStart, batchEnd in get_batch_indices(dataA.shape, dtype, batchSizeFlat):
        output[batchStart:batchEnd] = np.abs(dataA[batchStart:batchEnd].astype(dtype) -
                                             dataB[batchStart:batchEnd].astype(dtype))


def abs_diff_hdf_arrays_masked(dataA: h5py.Dataset, dataB: h5py.Dataset, mask: h5py.Dataset, output: h5py.Dataset,
                               dtype: np.dtype, batchSizeFlat=1e8):
    """
    Compute element-wise |A-B| where mask is set to true, zero everywhere else.
    Computation is performed in batches to decrease memory requirements.

    :param dataA:
    :param dataB:
    :param mask:
    :param output:
    :param dtype:
    :param batchSizeFlat:
    :return:
    """
    if dataA.shape != dataB.shape or dataA.shape != output.shape or dataA.shape != mask.shape:
        raise ValueError("Arguments should have equal shapes, {}, {}, {} and {} given."
                         .format(dataA.shape, dataB.shape, mask.shape, output.shape))
    if output.dtype != dtype:
        raise ValueError("Output array data type doesn't match the provided type: {} and {}"
                         .format(output.dtype, dtype))

    for batchStart, batchEnd in get_batch_indices(dataA.shape, dtype, batchSizeFlat):
        batchMask = mask[batchStart:batchEnd]
        batchDiff = batchMask * np.abs(dataA[batchStart:batchEnd].astype(dtype) -
                                       dataB[batchStart:batchEnd].astype(dtype))
        output[batchStart:batchEnd] = batchDiff


def mse_hdf_arrays(dataA: h5py.Dataset, dataB: h5py.Dataset, dtype: np.dtype, batchSizeFlat=1e8):
    """
    Compute MSE between two HDF datasets.
    Computation is performed in batches to decrease memory requirements.

    :param dataA:
    :param dataB:
    :param dtype:
    :param batchSizeFlat:
    :return:
    """
    if dataA.shape != dataB.shape:
        raise ValueError("Arguments should have equal shapes, {} and {} given."
                         .format(dataA.shape, dataB.shape))

    sum = 0.0
    count = 0
    for batchStart, batchEnd in get_batch_indices(dataA.shape, dtype, batchSizeFlat):
        batchA = dataA[batchStart:batchEnd].astype(dtype)
        batchB = dataB[batchStart:batchEnd].astype(dtype)
        sum += np.sum(np.square(batchA - batchB), dtype=dtype)
        count += multiply(batchA.shape)

    return sum / count if count > 0 else float('nan')


def mse_hdf_arrays_masked(dataA: h5py.Dataset, dataB: h5py.Dataset, mask: h5py.Dataset,
                          dtype: np.dtype, batchSizeFlat=1e8):
    """
    Compute MSE between two HDF datasets, considering elements where the mask is set to true
    Computation is performed in batches to decrease memory requirements.
    """
    if dataA.shape != dataB.shape or dataA.shape != mask.shape:
        raise ValueError("Arguments should have equal shapes, {}, {} and {} given."
                         .format(dataA.shape, dataB.shape, mask.shape))

    sum = 0.0
    count = 0
    for batchStart, batchEnd in get_batch_indices(dataA.shape, dtype, batchSizeFlat):
        batchMask = mask[batchStart:batchEnd]
        diff = batchMask * (dataA[batchStart:batchEnd].astype(dtype) - dataB[batchStart:batchEnd].astype(dtype))
        square = np.square(diff)
        nonzeroNumber = np.count_nonzero(batchMask)
        sum += np.sum(square)
        count += nonzeroNumber

    return sum / count if count > 0 else float('nan')


def var_hdf_array_masked(data: h5py.Dataset, mask: h5py.Dataset, dtype: np.dtype, batchSizeFlat=1e8):
    """
    Compute MSE between two HDF datasets, considering elements where the mask is set to true.
    Computation is performed in batches to decrease memory requirements.
    """

    sum = 0.0
    count = 0
    for batchStart, batchEnd in get_batch_indices(data.shape, dtype, batchSizeFlat):
        batchMask = mask[batchStart:batchEnd].astype(dtype)
        batchSum = np.sum((batchMask * data[batchStart:batchEnd].astype(dtype)), dtype=dtype)
        nonzeroNumber = np.count_nonzero(batchMask)
        sum += batchSum
        count += nonzeroNumber
    mean = sum / count if count > 0 else float('nan')

    sum = 0.0
    count = 0
    for batchStart, batchEnd in get_batch_indices(data.shape, dtype, batchSizeFlat):
        batchMask = mask[batchStart:batchEnd].astype(dtype)
        batchSum = np.sum(batchMask * np.square(data[batchStart:batchEnd].astype(dtype) - mean), dtype=dtype)
        nonzeroNumber = np.count_nonzero(batchMask)
        sum += batchSum
        count += nonzeroNumber

    return sum / count if count > 0 else float('nan')


def var_hdf_array(data: h5py.Dataset, dtype: np.dtype, batchSizeFlat=1e8):
    """
    Compute MSE between two HDF datasets.
    Computation is performed in batches to decrease memory requirements.

    :param dtype:
    :param batchSizeFlat:
    :param data:
    :param dataB:
    :param output:
    :return:
    """

    sum = 0.0
    count = 0
    for batchStart, batchEnd in get_batch_indices(data.shape, dtype, batchSizeFlat):
        batch = data[batchStart:batchEnd]
        sum += np.sum(batch.astype(dtype), dtype=dtype)
        count += multiply(batch.shape)
    mean = sum / count if count > 0 else float('nan')

    sum = 0.0
    count = 0
    for batchStart, batchEnd in get_batch_indices(data.shape, dtype, batchSizeFlat):
        batch = data[batchStart:batchEnd]
        sum += np.sum(np.square(batch.astype(dtype) - mean), dtype=dtype)
        count += multiply(batch.shape)

    return sum / count if count > 0 else float('nan')


def reshape_and_pad_volume(inputFlatVolume: h5py.Dataset, outputVolume: h5py.Dataset, targetDomain: Tuple[Tuple, Tuple]):
    """
    Reshapes a flat input array into a 4D volume and 'pastes' it into the output,
    such that it occupies the target domain.

    See also: get_prediction_domain()

    :param inputFlatVolume:
    :param outputVolume:
    :param targetDomain: min and max corners
    :return:
    """

    assert(inputFlatVolume.ndim == 2)
    assert(outputVolume.ndim == 4)

    targetBegin = np.asarray(targetDomain[0])
    targetEnd = np.asarray(targetDomain[1])
    targetShape = targetEnd - targetBegin
    targetSpatialSizeFlat = int(np.prod(targetShape[1:]))  # Cast to python int to avoid overflow.

    # Do frame-by-frame to keep things out-of-core.
    for fInput in range(targetShape[0]):

        outputSpatialSelector = tuple(slice(targetBegin[dim], targetEnd[dim])
                                      for dim in range(1, 4))
        inputFlatSelector = slice(fInput * targetSpatialSizeFlat,
                                  (fInput + 1) * targetSpatialSizeFlat)
        fOutput = targetBegin[0] + fInput

        outputVolume[(fOutput,) + outputSpatialSelector] = inputFlatVolume[inputFlatSelector, :].reshape(targetShape[1:])


def sparse_insert_into_nd_array(bnaPointer, indices: np.ndarray, values: np.ndarray, valueNumber: int):
    return CppWrapper.sparse_insert_into_nd_array(bnaPointer, indices, values, valueNumber)


def sparse_insert_const_into_nd_array(bnaPointer, indices: np.ndarray, value: int, valueNumber: int):
    return CppWrapper.sparse_insert_const_into_nd_array(bnaPointer, indices, value, valueNumber)


def smooth_3d_array_average(data: np.ndarray, kernelRadius: int) -> np.ndarray:
    return CppWrapper.smooth_3d_array_average(data, kernelRadius)


class NumpyJsonEncoder(json.JSONEncoder):
    """
    A custom JSON encoder class that is required for encoding Numpy array values
    without explicitly converting them to Python types.
    Without it, JSON serializer will crash, since Numpy types aren't 'serializable'.
    Credit: https://stackoverflow.com/a/27050186/1545327
    """

    def default(self, o):
        if isinstance(o, np.integer):
            return int(o)
        if isinstance(o, np.floating):
            return float(o)
        if isinstance(o, np.ndarray):
            return o.tolist()

        return super().default(o)


class NumpyDynArray:
    """
    A dynamically resized ND-array that uses NumPy for storage.
    """
    def __init__(self, shape, dtype=None):
        # For now we assume that the first dimension is the variable-sized dimension.
        assert(shape[0] == -1)

        self.size = 0
        self.capacity = 100
        shape = (self.capacity,) + shape[1:]

        self.data = np.empty(shape, dtype=dtype)

    def __str__(self, *args, **kwargs):
        return self.get_all().__str__(*args, **kwargs)

    @property
    def shape(self):
        return self.get_all().shape  # Should be fast, assuming that get_all returns a view.

    def append(self, dataToAdd):
        """
        Append a single row (subtensor?) of data along the dynamic axis.
        :param dataToAdd:
        :return:
        """
        if dataToAdd.shape != self.data.shape[1:]:
            raise ValueError("Cannot append the data. Expected shape: {}. Provided: {}."
                             .format(self.data.shape[1:], dataToAdd.shape))

        # Do we still have free space?
        if self.size >= self.capacity:
            self._allocate_more_space()

        # Add a new 'row' (hunk).
        self.data[self.size, ...] = dataToAdd
        self.size += 1

    def append_all(self, dataToAdd):
        """
        Append multiple rows (subtensors?) of data. Effectively a concatenation
        along the dynamic axis.
        :param dataToAdd:
        :return:
        """
        if dataToAdd.shape[1:] != self.data.shape[1:]:
            raise ValueError("Cannot append the data. Expected shape: {}. Provided: {}."
                             .format(self.data.shape[1:], dataToAdd.shape))

        newRowNumber = dataToAdd.shape[0]
        # Do we still have free space?
        while self.size + newRowNumber >= self.capacity:
            self._allocate_more_space()

        self.data[self.size:self.size + newRowNumber, ...] = dataToAdd
        self.size += newRowNumber

    def _allocate_more_space(self):
        # Allocate a new array of a bigger size.
        self.capacity *= 2
        shape = (self.capacity,) + self.data.shape[1:]
        newData = np.empty(shape, dtype=self.data.dtype)
        # Copy the data to the new array.
        newData[:self.size, ...] = self.data[:self.size, ...]
        self.data = newData

    def get_all(self):
        return self.data[:self.size, ...]

    def __getitem__(self, item):
        return self.data.__getitem__(item)

    def __setitem__(self, key, value):
        raise NotImplementedError()
