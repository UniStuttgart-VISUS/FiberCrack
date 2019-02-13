import os.path as path
import os
import warnings
import json
import gzip

from typing import Union, Callable, Tuple, List, Dict, Any

import numpy as np
import h5py


def load_volume_data_from_dat(datPath,
                              outputAllocator: Callable[[Tuple, np.dtype], Union[h5py.Dataset, np.array]] = None,
                              timestepsToLoad: List[int] = None,
                              printFn: Callable[[Any], None] = None):
    """
    Reads a provided .dat metadata file, and loads a corresponding volume or a volume sequence.
    Use the allocator callback to flexibly define how the output show be stored.

    :param datPath:
    :param outputAllocator:
    :param timestepsToLoad:
    :param printFn:
    :return:
    """

    if not os.path.exists(datPath):
        raise RuntimeError("Path does not exist: '{}'".format(datPath))

    filename, extension = os.path.splitext(datPath)
    if not extension.lower() == '.dat':
        raise RuntimeError("Path does no point to a .dat file: '{}'".format(datPath))

    # Read out the metadata.
    baseDirPath = os.path.dirname(datPath)
    metadata = _read_metadata_from_dat(datPath)
    volumeDirName = metadata['ObjectFileName'.lower()]
    volumeSpatialSize = metadata['Resolution'.lower()]
    volumeDataType = metadata['Format'.lower()].lower()
    dtype = _string_to_dtype(volumeDataType)

    componentNumber = 1
    compression = 'none'
    if 'ComponentNumber'.lower() in metadata:
        componentNumber = metadata['ComponentNumber'.lower()]
    if 'Compression'.lower() in metadata:
        compression = metadata['Compression'.lower()].lower()

    # The 'ObjectFileName' parameter specifies either the dir where all files should be loaded,
    # or a wildcard pattern, matching files with a certain extension, e.g. "/path/to/dir/*.raw".
    volumeExtensionFilter = None
    starIndex = volumeDirName.find('*')
    if starIndex > 0:
        if volumeDirName[starIndex + 1] != '.':
            raise RuntimeError("Invalid volume wildcard provided: '{}'".format(volumeDirName))

        volumeExtensionFilter = volumeDirName[starIndex + 1:]
        volumeDirName = volumeDirName[:starIndex]

    dataPath = os.path.join(baseDirPath, volumeDirName)

    # If we don't have a sequence, parse a single static volume.
    if not os.path.isdir(dataPath):
        volumeFilePath = dataPath
        shape = tuple(reversed(volumeSpatialSize))
        if componentNumber > 1:
            shape += (componentNumber,)

        assert(outputAllocator is None)  # We don't support output allocation for static volumes.
        frameData = _read_binary_file(volumeFilePath, dtype, compression)

        return frameData.reshape(shape)

    assert(componentNumber == 1)  # Multi-channel volume sequences are not supported.

    if outputAllocator is None:
        outputAllocator = lambda shape, dtype: np.zeros(shape, dtype)

    # Figure out exactly which files to load.
    filenames = [f for f in os.listdir(dataPath)]
    if volumeExtensionFilter is not None:
        filenames = [f for f in filenames if os.path.splitext(f)[1] == volumeExtensionFilter]
    if timestepsToLoad is not None:
        filenames = [filenames[f] for f in timestepsToLoad]

    # Spatial shape is reversed, because volume data is stored in Z,Y,X C-order.
    shape = (len(filenames),) + tuple(reversed(volumeSpatialSize))
    data = outputAllocator(shape, dtype)
    for i, filename in enumerate(filenames):
        if printFn:
            printFn("Reading file {}".format(filename))

        filepath = os.path.join(dataPath, filename)
        frameData = _read_binary_file(filepath, dtype, compression)

        frameData = frameData.reshape(shape[1:])
        data[i, ...] = frameData

    return data


def _read_binary_file(volumeFilePath: str, dtype: np.dtype, compression: str = 'none') -> np.ndarray:
    if compression == 'none' and os.path.splitext(volumeFilePath)[1] == '.gz':
        warnings.warn("Reading '.gz' volume file as uncompressed. Is the metadata outdated?")

    if compression == 'none':
        with open(volumeFilePath, 'rb') as file:
            frameData = np.fromfile(file, dtype)
    elif compression == 'gzip':
        with gzip.open(volumeFilePath, 'rb') as file:
            frameData = np.frombuffer(file.read(), dtype=dtype)
    else:
        raise RuntimeError("Unknown compression method: '{}'".format(compression))

    return frameData


def read_metadata_from_dat(datPath):
    return _read_metadata_from_dat(datPath)


def _read_metadata_from_dat(datPath):
    data = {}

    with open(datPath, 'r') as file:
        for line in file.readlines():
            chunks = [c.strip() for c in line.split(':')]
            key, valueString = chunks[0].lower(), chunks[1]
            if key == 'ComponentNumber'.lower():
                value = int(valueString)
            elif key == 'Resolution'.lower():
                value = tuple([int(c.strip()) for c in valueString.split()])  # e.g. "256 256 256"
            else:
                value = valueString.strip()

            data[key] = value

    return data


def load_volume_sequence_from_dir(dir, size, frameSlice, preloadedDataPath=None, printFn=None):

    filenames = [f for f in os.listdir(dir)]
    width, height, depth = size

    if not os.path.isfile(preloadedDataPath):
        if printFn:
            printFn("Reading data from {}".format(dir))

        size4d = (len(filenames),) + size
        data = np.empty(size4d, dtype=np.uint8)
        for i, filename in enumerate(filenames):
            if printFn:
                printFn("Reading file {}".format(filename))

            filepath = os.path.join(dir, filename)
            with open(filepath, 'rb') as file:
                frameData = np.fromfile(file, np.uint8, width * height * depth)

            frameData = frameData.reshape((depth, height, width))
            data[i, ...] = frameData

        h5File = h5py.File(preloadedDataPath, 'w')
        h5Data = h5File.create_dataset('volume-data', data=data, dtype='uint8')
        h5Data[...] = data
        data = data[frameSlice, ...]

    else:
        if printFn:
            printFn("Reading preloaded data from {}".format(preloadedDataPath))
        h5File = h5py.File(preloadedDataPath, 'r')
        data = h5File['volume-data'][frameSlice, ...]

    return data


def write_volume_sequence(dir: str, data: Union[np.ndarray, h5py.Dataset],
                          timeAxis: int = 0,
                          filenameProvider: Callable[[int], str] = None,
                          clip: Tuple = None, dtype: np.dtype = None,
                          compress: bool = False,
                          printFn: Callable[[Any], None] = None):

    assert(timeAxis == 0)  # HDF files do not support fancy slicing. For now just assume that time is the first axis.

    if filenameProvider is None:
        filenameProvider = lambda x: 'frame_{:04d}.raw'.format(x)

    if dtype is None:
        dtype = data.dtype

    extension = os.path.splitext(dir)[1]
    # Strip the extension, if a path to a .dat file was provided.
    if extension == '.dat':
        dir = dir[:-len('.dat')]

    if not os.path.exists(dir):
        os.makedirs(dir)

    basePath = os.path.dirname(dir)  # Parent dir of the volume dir.
    metaPath = os.path.join(basePath, os.path.basename(dir) + '.dat')

    if printFn:
        printFn("Writing volume sequence data to '{}'".format(dir))

    for f in range(0, data.shape[timeAxis]):
        frameData = data[f, ...]
        if clip is not None:
            frameData = np.clip(frameData, clip[0], clip[1])
        frameData = frameData.astype(dtype)

        filename = filenameProvider(f)
        filepath = os.path.join(dir, filename)
        if not compress:
            frameData.tofile(filepath)
        else:
            with gzip.open(filepath + '.gz', 'wb', compresslevel=6) as f:
                f.write(frameData.tobytes())

    if printFn:
        printFn("Writing volume sequence metadata to '{}'".format(metaPath))

    compression = 'gzip' if compress else 'none'
    _write_metadata_to_dat(metaPath, data.shape[1:], os.path.basename(dir), dtype=dtype, compression=compression)


def write_volume_sequence_from_generator(dirOrDatPath: str, frameGenerator,
                                         filenameProvider: Callable[[int], str] = None,
                                         clip: Tuple = None, dtype: np.dtype = None,
                                         printFn: Callable[[Any], None] = None):

    if filenameProvider is None:
        filenameProvider = lambda x: 'frame_{:04d}.raw'.format(x)

    extension = os.path.splitext(dirOrDatPath)[1]
    # Strip the extension, if a path to a .dat file was provided.
    dirPath = dirOrDatPath[:-len('.dat')] if extension == '.dat' else dirOrDatPath

    if not os.path.exists(dirPath):
        os.makedirs(dirPath)

    basePath = os.path.dirname(dirPath)  # Parent dir of the volume dir.
    metaPath = os.path.join(basePath, os.path.basename(dirPath) + '.dat')

    if printFn:
        printFn("Writing volume sequence data to '{}'".format(dirPath))

    spatialShape = None

    for f, frameData in enumerate(frameGenerator):

        if dtype is None:
            dtype = frameData.dtype
        if spatialShape is None:
            spatialShape = frameData.shape
        elif (spatialShape != frameData.shape):
            raise RuntimeError("Data has frames of different shape: {} ({} before)"
                               .format(frameData.shape, spatialShape))

        if clip is not None:
            frameData = np.clip(frameData, clip[0], clip[1])
        frameData = frameData.astype(dtype)

        filename = filenameProvider(f)
        frameData.tofile(os.path.join(dirPath, filename))

    if printFn:
        printFn("Writing volume sequence metadata to '{}'".format(metaPath))

    _write_metadata_to_dat(metaPath, spatialShape, os.path.basename(dirPath), dtype)


def write_volume_to_datraw(data: np.ndarray, rawOrDatPath: str, dataOrder: str = 'zyx',
                           printFn: Callable[[Any], None] = None):
    """
    Datraw: raw format, with a metadata .dat file describing the resolution.
    :param data:
    :param rawOrDatPath:
    :param dataOrder:
    :param printFn:
    :return:
    """

    beforeExtension, extension = path.splitext(rawOrDatPath)

    assert data.dtype == np.uint8

    if dataOrder == 'zyx':
        pass
    elif dataOrder == 'xyz':
        data = data.swapaxes(0, 2)
    else:
        raise RuntimeError("Unsupported data order: {}".format(dataOrder))

    rawPath = beforeExtension + '.raw'
    metaPath = beforeExtension + '.dat'

    if printFn:
        printFn("Writing raw volume data to '{}'".format(rawPath))

    data.tofile(rawPath)

    if printFn:
        printFn("Writing volume metadata to '{}'".format(metaPath))

    _write_metadata_to_dat(metaPath, data.shape, os.path.basename(rawPath), dtype=data.dtype)


def _write_metadata_to_dat(metaPath, shape, dataFilename, dtype: np.dtype, compression='none'):
    volumeSizeString = ' '.join(reversed([str(x) for x in shape[0:3]]))  # Specified as X Y Z
    componentNumber = 1 if len(shape) == 3 else shape[3]

    with open(metaPath, 'w', encoding='ascii') as file:
        file.write('ObjectFileName: {}\n'.format(dataFilename))
        file.write('Resolution: {}\n'.format(volumeSizeString))
        file.write('Format: {}\n'.format(_dtype_to_string(dtype)))
        file.write('ComponentNumber: {}\n'.format(componentNumber))
        file.write('Compression: {}\n'.format(compression))


def _dtype_to_string(dtype: np.dtype):
    if dtype == np.uint8:
        return 'uchar'
    elif dtype == np.float32:
        return 'float'
    else:
        raise RuntimeError("Unsupported volume data type: '{}'".format(dtype))


def _string_to_dtype(dtypeString: str) -> np.dtype:
    if dtypeString == 'uchar':
        return np.dtype(np.uint8)
    elif dtypeString == 'float':
        return np.dtype(np.float32)
    else:
        raise RuntimeError("Unsupported volume data type: '{}'".format(dtypeString))


def write_parallax_seven_config(configPath, series: List[Dict[str, Any]]):
    for seriesDesc in series:
        if 'path' not in seriesDesc:
            raise RuntimeError("Series must specify a path.")

        if 'slice' in seriesDesc:
            sliceObject = seriesDesc['slice']
            if sliceObject and (sliceObject.start or sliceObject.stop or sliceObject.step):
                sliceString = '{},{},{}'.format(sliceObject.start, sliceObject.stop, sliceObject.step)
                seriesDesc['slice'] = sliceString
            else:
                del seriesDesc['slice']

    with open(configPath, 'w', encoding='utf-8') as file:
        json.dump({'series': series}, file)