import ctypes
from typing import Tuple, Union, List
import h5py
import os
import platform
import time
import math
from typing import Tuple

import numpy as np


# todo Rework this class to just help calling the DLL code, instead of wrapping every function.
# Let the client have the wrapper.
class CppWrapper:
    """
    Wraps functions written in C++ into Python code.
    """

    _Dll = None

    @classmethod
    def _get_c_dll(cls):
        if cls._Dll is None:
            platformName = platform.system()
            if platformName == 'Windows':
                dllPath = os.path.join(os.path.dirname(__file__), 'c_dll', 'PythonExtrasC.dll')
                cls._Dll = ctypes.CDLL(dllPath)
            elif platformName == 'Linux':
                soPath = os.path.join(os.path.dirname(__file__), 'c_dll', 'PythonExtrasC.so')
                cls._Dll = ctypes.CDLL(soPath)
            else:
                raise RuntimeError("Unknown platform: {}".format(platformName))

        return cls._Dll

    @classmethod
    def test(cls, data: np.ndarray):
        func = cls._get_c_dll().test
        func(ctypes.c_void_p(data.ctypes.data), ctypes.c_int32(4), ctypes.c_int32(4))

    @classmethod
    def static_state_test(cls, increment: int):
        func = cls._get_c_dll().static_state_test
        result = ctypes.c_uint64()
        func(ctypes.c_uint64(increment), ctypes.byref(result))

        return result.value

    @classmethod
    def multithreading_test(cls, data: np.ndarray, threadNumber: int):
        func = cls._get_c_dll().multithreading_test
        size = data.shape[0]

        func(data.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)), ctypes.c_uint64(size), ctypes.c_uint64(threadNumber))

        return data

    @classmethod
    def resize_array_point(cls, data: np.ndarray, targetSize: Tuple):
        assert(data.ndim <= 3)

        # Implicitly support 1D and 3D array by adding extra dimensions.
        ndimOrig = data.ndim
        dataShapeOrig = data.shape
        while data.ndim < 3:
            data = data[..., np.newaxis]
            targetSize += (1,)

        if data.dtype == np.float32:
            func = cls._get_c_dll().resize_array_point_float32
        elif data.dtype == np.float64:
            func = cls._get_c_dll().resize_array_point_float64
        elif data.dtype == np.uint8:
            func = cls._get_c_dll().resize_array_point_uint8
        else:
            raise(RuntimeError("Unsupported data type: {}".format(data.dtype)))

        output = np.empty(targetSize, dtype=data.dtype)

        func(ctypes.c_void_p(data.ctypes.data),
             ctypes.c_int32(data.shape[0]),
             ctypes.c_int32(data.shape[1]),
             ctypes.c_int32(data.shape[2]),
             ctypes.c_void_p(output.ctypes.data),
             ctypes.c_int32(targetSize[0]),
             ctypes.c_int32(targetSize[1]),
             ctypes.c_int32(targetSize[2]))

        while output.ndim > ndimOrig:
            output = output[..., 0]

        return output

    @classmethod
    def extract_patches(cls, data: Union[np.ndarray, h5py.Dataset], sourceAxes: Tuple, patchSize: Tuple,
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
            :param batchSize: How many patches to process per C++ call: non-zero values allow for
                              progress reporting.
            :param verbose: Whether to output progress information.
            :return:
            """

        assert(data.dtype == np.uint8)

        # By default, extract every patch.
        if patchStride is None:
            patchStride = (1,) * len(sourceAxes)

        patchShape = cls._compute_patch_shape(data.shape, sourceAxes, patchSize)
        patchNumber = cls._compute_patch_number(data.shape, sourceAxes, patchSize, patchStride)
        patchNumberFlat = np.prod(np.asarray(patchNumber))  # type: int

        inputData = data[...]  # type: np.ndarray
        outputBufferSize = patchNumberFlat * np.prod(np.asarray(patchShape)) * data.dtype.itemsize
        outputData = np.empty((patchNumberFlat,) + tuple(patchShape), dtype=data.dtype)
        outputCenters = np.empty((patchNumberFlat, len(sourceAxes)), dtype=np.uint64)

        func = cls._get_c_dll().extract_patches_batched_uint8

        dataSize = np.array(data.shape, dtype=np.uint64)
        sourceAxes = np.array(sourceAxes, dtype=np.uint64)
        patchSize = np.array(patchSize, dtype=np.uint64)
        patchStride = np.array(patchStride, dtype=np.uint64)

        if batchSize == 0:
            batchSize = patchNumberFlat

        for i in range(0, patchNumberFlat, batchSize):
            patchesInBatch = min(batchSize, patchNumberFlat - i)

            timeStart = time.time()
            func(ctypes.c_void_p(inputData.ctypes.data),
                 ctypes.c_void_p(outputData.ctypes.data),
                 outputCenters.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                 ctypes.c_uint64(inputData.ndim),
                 dataSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                 ctypes.c_uint64(dataSize.size),
                 sourceAxes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                 ctypes.c_uint64(sourceAxes.size),
                 patchSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                 ctypes.c_uint64(patchSize.size),
                 patchStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
                 ctypes.c_uint64(patchStride.size),
                 ctypes.c_uint64(i), ctypes.c_uint64(patchesInBatch), ctypes.c_bool(False))

            if verbose:
                print("Extracting patches: {:02.2f}%, {:02.2f}s. per batch".format(i / patchNumberFlat * 100,
                                                                                   time.time() - timeStart))

        return outputData, outputCenters, patchNumber

    @classmethod
    def extract_patches_batch(cls, data: np.ndarray, sourceAxes: Tuple, patchSize: Tuple,
                              patchStride: Tuple = None, batchStart: int = 0, batchSize: int = 0):
        assert (data.dtype == np.uint8)
        # Because this batched function is called multiple times, we do not manage h5py datasets inside.
        assert (isinstance(data, np.ndarray))

        # By default, extract every patch.
        if patchStride is None:
            patchStride = (1,) * len(sourceAxes)

        patchShape = cls._compute_patch_shape(data.shape, sourceAxes, patchSize)
        patchNumber = cls._compute_patch_number(data.shape, sourceAxes, patchSize, patchStride)
        patchNumberFlat = np.prod(np.asarray(patchNumber))  # type: int

        if batchSize == 0:
            batchSize = patchNumberFlat

        patchesInBatch = min(batchSize, patchNumberFlat - batchStart)

        inputData = data
        outputData = np.empty((patchesInBatch,) + tuple(patchShape), dtype=data.dtype)
        outputCenters = np.empty((patchesInBatch, len(sourceAxes)), dtype=np.uint64)

        func = cls._get_c_dll().extract_patches_batched_uint8

        dataSize = np.array(data.shape, dtype=np.uint64)
        sourceAxes = np.array(sourceAxes, dtype=np.uint64)
        patchSize = np.array(patchSize, dtype=np.uint64)
        patchStride = np.array(patchStride, dtype=np.uint64)

        func(ctypes.c_void_p(inputData.ctypes.data),
             ctypes.c_void_p(outputData.ctypes.data),
             outputCenters.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(inputData.ndim),
             dataSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(dataSize.size),
             sourceAxes.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(sourceAxes.size),
             patchSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(patchSize.size),
             patchStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(patchStride.size),
             ctypes.c_uint64(batchStart), ctypes.c_uint64(patchesInBatch), ctypes.c_bool(True))

        return outputData, outputCenters, patchNumber

    @classmethod
    def extract_patched_training_data_without_empty_4d(cls, dataBuffer: np.ndarray, dataShape: Tuple,
                                                       dataStartFlat: int, dataEndFlat: int, outputX: np.ndarray,
                                                       outputY: np.ndarray, outputIndices: np.ndarray, patchSize: Tuple,
                                                       patchStride: Tuple, batchStartIndex: int, batchSize: int,
                                                       patchInnerStride: Tuple = (1, 1, 1, 1), lastFrameGap:int = 1,
                                                       undersamplingProb: float = 1.0,
                                                       skipEmptyPatches: bool = True, emptyValue: int = 0):
        assert(len(dataShape) == 4)
        # C++ code copies data column-by-column along the last axis,
        # this means that the last axis must be continuous.
        assert(patchInnerStride[-1] == 1)
        assert(dataBuffer.dtype == np.uint8)
        # Because this batched function is called multiple times, we do not manage h5py datasets inside.
        assert(isinstance(dataBuffer, np.ndarray))

        func = cls._get_c_dll().extract_patched_training_data_without_empty_4d_uint8

        dataSize = np.array(dataShape, dtype=np.uint64)
        patchSize = np.array(patchSize, dtype=np.uint64)
        patchStride = np.array(patchStride, dtype=np.uint64)
        patchInnerStride = np.array(patchInnerStride, dtype=np.uint64)

        patchesExtracted = ctypes.c_uint64()
        nextBatchIndex = ctypes.c_uint64()
        inputEndReached = ctypes.c_bool()

        func(dataBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             ctypes.c_uint64(dataStartFlat), ctypes.c_uint64(dataEndFlat),

             dataSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             patchSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             patchStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             patchInnerStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(lastFrameGap),

             ctypes.c_bool(skipEmptyPatches), ctypes.c_uint8(emptyValue),
             ctypes.c_uint64(batchStartIndex), ctypes.c_uint64(batchSize),
             ctypes.c_float(undersamplingProb),

             outputX.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             outputY.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             outputIndices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.byref(patchesExtracted),
             ctypes.byref(nextBatchIndex),
             ctypes.byref(inputEndReached)
             )

        if nextBatchIndex.value == batchStartIndex:
            raise RuntimeError("No patches have been processed, does the input buffer contain the next patch?")

        return patchesExtracted.value, nextBatchIndex.value, inputEndReached.value

    @classmethod
    def extract_patched_training_data_without_empty_4d_multi(
                                                       cls, dataBuffer: np.ndarray, dataShape: Tuple,
                                                       dataStartFlat: int, dataEndFlat: int, outputX: np.ndarray,
                                                       outputY: np.ndarray, outputIndices: np.ndarray, patchSize: Tuple,
                                                       patchStride: Tuple, batchStartIndex: int, batchSize: int,
                                                       patchInnerStride: Tuple = (1, 1, 1, 1), lastFrameGap: int = 1,
                                                       undersamplingProb: float = 1.0,
                                                       skipEmptyPatches: bool = True, emptyValue: int = 0,
                                                       threadNumber: int = 8):
        assert (len(dataShape) == 4)
        # C++ code copies data column-by-column along the last axis,
        # this means that the last axis must be continuous.
        assert (patchInnerStride[-1] == 1)
        assert (dataBuffer.dtype == np.uint8)
        assert (outputIndices.dtype == np.uint64)
        # Because this batched function is called multiple times, we do not manage h5py datasets inside.
        assert (isinstance(dataBuffer, np.ndarray))

        func = cls._get_c_dll().extract_patched_training_data_without_empty_4d_multithreaded_uint8

        dataSize = np.array(dataShape, dtype=np.uint64)
        patchSize = np.array(patchSize, dtype=np.uint64)
        patchStride = np.array(patchStride, dtype=np.uint64)
        patchInnerStride = np.array(patchInnerStride, dtype=np.uint64)

        patchesExtracted = ctypes.c_uint64()
        nextBatchIndex = ctypes.c_uint64()
        inputEndReached = ctypes.c_bool()

        func(dataBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             ctypes.c_uint64(dataStartFlat), ctypes.c_uint64(dataEndFlat),

             dataSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             patchSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             patchStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             patchInnerStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.c_uint64(lastFrameGap),

             ctypes.c_bool(skipEmptyPatches), ctypes.c_uint8(emptyValue),
             ctypes.c_uint64(batchStartIndex), ctypes.c_uint64(batchSize),
             ctypes.c_float(undersamplingProb),
             ctypes.c_uint64(threadNumber),


             outputX.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             outputY.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             outputIndices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             ctypes.byref(patchesExtracted),
             ctypes.byref(nextBatchIndex),
             ctypes.byref(inputEndReached)
             )

        if nextBatchIndex.value == batchStartIndex:
            raise RuntimeError("No patches have been processed, does the input buffer contain the next patch?")

        return patchesExtracted.value, nextBatchIndex.value, inputEndReached.value

    @classmethod
    def extract_patched_training_data_4d_multithreaded(cls, dataBuffer: np.ndarray, dataShape: Tuple, dataStartFlat: int,
                                                       dataEndFlat: int, outputX: np.ndarray, outputY: np.ndarray,
                                                       patchSize: Tuple, batchStartIndex: int, batchSize: int,
                                                       patchInnerStride: Tuple = (1, 1, 1, 1), lastFrameGap: int = 1,
                                                       threadNumber: int = 8):
        assert (len(dataShape) == 4)
        # C++ code copies data column-by-column along the last axis,
        # this means that the last axis must be continuous.
        assert (patchInnerStride[-1] == 1)
        assert (dataBuffer.dtype == np.uint8)
        # Because this batched function is called multiple times, we do not manage h5py datasets inside.
        assert (isinstance(dataBuffer, np.ndarray))

        func = cls._get_c_dll().extract_patched_training_data_multithreaded_uint8

        dataSize = np.array(dataShape, dtype=np.uint64)
        patchSize = np.array(patchSize, dtype=np.uint64)
        patchInnerStride = np.array(patchInnerStride, dtype=np.uint64)

        patchesExtracted = ctypes.c_uint64()
        nextBatchIndex = ctypes.c_uint64()
        inputEndReached = ctypes.c_bool()

        func(
            dataBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.c_uint64(dataStartFlat),
            ctypes.c_uint64(dataEndFlat),
            dataSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            patchSize.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            patchInnerStride.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_uint64(lastFrameGap),
            ctypes.c_uint64(batchStartIndex),
            ctypes.c_uint64(batchSize),
            ctypes.c_uint64(threadNumber),
            outputX.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            outputY.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            ctypes.byref(patchesExtracted),
            ctypes.byref(nextBatchIndex),
            ctypes.byref(inputEndReached)
             )

        if nextBatchIndex.value == batchStartIndex:
            raise RuntimeError("No patches have been processed, does the input buffer contain the next patch?")

        return patchesExtracted.value, nextBatchIndex.value, inputEndReached.value

    @classmethod
    def sparse_insert_into_nd_array(cls, bnaPointer, indices: np.ndarray, values: np.ndarray, valueNumber: int):
        func = cls._get_c_dll().sparse_insert_into_nd_array_uint8

        assert (indices.dtype == np.uint64)
        assert (values.dtype == np.uint8)

        func(
            ctypes.c_void_p(bnaPointer),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            values.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
            valueNumber
        )

    @classmethod
    def sparse_insert_const_into_nd_array(cls, bnaPointer, indices: np.ndarray, value: int, valueNumber: int):
        func = cls._get_c_dll().sparse_insert_const_into_nd_array_uint8

        assert (indices.dtype == np.uint64)

        func(
            ctypes.c_void_p(bnaPointer),
            indices.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_uint8(value),
            valueNumber
        )

    @classmethod
    def smooth_3d_array_average(cls, data: np.ndarray, kernelRadius: int) -> np.ndarray:
        assert(data.dtype == np.float32)

        func = cls._get_c_dll().smooth_3d_array_average_float

        shape = np.asarray(data.shape, dtype=np.uint64)
        output = np.empty_like(data)
        func(
            data.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            shape.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_uint64(kernelRadius),
            output.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
        )

        return output

    @classmethod
    def bna_construct(cls, filepath: str, shape: Tuple, maxBufferSize: int):
        func = cls._get_c_dll().bna_construct
        func.restype = ctypes.c_void_p

        shapeArray = np.asarray(shape, dtype=np.uint64)

        bnaPointer = func(
            filepath,
            shapeArray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
            ctypes.c_uint64(len(shape)),
            ctypes.c_uint64(maxBufferSize)
        )

        return bnaPointer

    @classmethod
    def bna_destruct(cls, bnaPointer):
        # todo All bna_xxxx methods can use the magic _as_parameter_ property to avoid explicit pointers.
        func = cls._get_c_dll().bna_destruct
        func(ctypes.c_void_p(bnaPointer))

    @classmethod
    def bna_flush(cls, bnaPointer, flushOsBuffer: bool):
        func = cls._get_c_dll().bna_flush
        func(ctypes.c_void_p(bnaPointer), ctypes.c_bool(flushOsBuffer))

    @classmethod
    def bna_read_slice(cls, bnaPointer, outputBuffer: np.ndarray, sliceIndex: int):
        func = cls._get_c_dll().bna_read_slice
        func(ctypes.c_void_p(bnaPointer),
             outputBuffer.ctypes.data_as(ctypes.POINTER(ctypes.c_uint8)),
             ctypes.c_uint64(sliceIndex))

    @classmethod
    def bna_fill_box(cls, bnaPointer, value, cornerLow: Tuple, cornerHigh: Tuple):

        cornerLowArray = np.asarray(cornerLow, dtype=np.uint64)
        cornerHighArray = np.asarray(cornerHigh, dtype=np.uint64)

        func = cls._get_c_dll().bna_fill_box
        func(ctypes.c_void_p(bnaPointer),
             ctypes.c_uint8(value),
             cornerLowArray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)),
             cornerHighArray.ctypes.data_as(ctypes.POINTER(ctypes.c_uint64)))

    def _compute_patch_number(dataShape: Tuple, sourceAxes: Tuple, patchSize: Tuple,
                              patchStride: Tuple = None):
        patchNumber = []
        for i, axis in enumerate(sourceAxes):
            totalPatchNumber = dataShape[axis] - patchSize[sourceAxes.index(axis)] + 1
            stride = patchStride[i]
            patchNumber.append(int(math.ceil(totalPatchNumber / stride)))

        return patchNumber

    def _compute_patch_shape(dataShape: Tuple, sourceAxes: Tuple, patchSize: Tuple):
        patchShape = []
        for axis in range(len(dataShape)):
            patchShape.append(patchSize[sourceAxes.index(axis)] if axis in sourceAxes else dataShape[axis])
        return patchShape