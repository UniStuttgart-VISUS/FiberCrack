import os
import re
import sys
from os import path

import h5py
import numpy as np
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.transform
import skimage.util
from PIL import Image

from FiberCrack.Dataset import Dataset
from PythonExtras.common_data_tools import read_csv_data, read_tiff_data

__all__ = ['DataImportConfig', 'load_csv_data', 'load_tiff_data']


class DataImportConfig:
    """
    Stores information on how and where from data should be loaded.
    """
    def __init__(self,
                 basePath=None,
                 dataDir=None,
                 imageDir=None,
                 groundTruthDir=None,
                 metadataFilename=None,
                 dataFormat=None,
                 imageBaseName=None,
                 imageFilenameFormat=None,
                 preloadedDataDir=None,
                 preloadedDataFilename=None,
                 crackAreaGroundTruthPath=None,
                 reloadOriginalData=False,
                 maxFrames=sys.maxsize,
                 ):

        self.groundTruthDir = groundTruthDir  # Path to binary images with true crack shape.
        self.basePath = basePath
        self.dataDir = dataDir
        self.imageDir = imageDir
        self.metadataFilename = metadataFilename
        self.dataFormat = dataFormat
        self.imageBaseName = imageBaseName
        self.imageFilenameFormat = imageFilenameFormat
        self.preloadedDataDir = preloadedDataDir
        self.preloadedDataFilename = preloadedDataFilename
        self.crackAreaGroundTruthPath = crackAreaGroundTruthPath  # Path to a CSV file with true measured crack area.
        self.reloadOriginalData = reloadOriginalData
        self.maxFrames = maxFrames

    def load_from_dict(self, configDict):
        for key, value in configDict.items():
            if key in self.__dict__:
                self.__dict__[key] = value
            else:
                raise RuntimeError("Unknown data-import config parameter: '{}'".format(key))


def load_csv_data(config: 'DataImportConfig'):
    if config.preloadedDataFilename is None:
        h5Filename = config.metadataFilename.replace('.csv', '.hdf5')
    else:
        h5Filename = config.preloadedDataFilename

    h5File = None

    preloadedDataPath = path.join(config.preloadedDataDir, h5Filename)
    if config.reloadOriginalData or not path.isfile(preloadedDataPath):
        print('Loading data from CSV files.')

        # We need to know the size of data that will be added later to allocate an hdf5 file.
        extraFeatureNumber = Dataset.get_data_feature_number()
        originalFeatureNumber = None

        # Load the metadata, describing each frame of the experiment.
        metadata, metaheader = read_csv_data(path.join(config.basePath, config.metadataFilename))

        dataFilenameList = os.listdir(path.join(config.basePath, config.dataDir))
        frameNumber = min(len(dataFilenameList), config.maxFrames)

        h5File = h5py.File(preloadedDataPath, 'w')
        h5Data = None
        frameMap = []
        header = None

        # Load data for each available frame.
        for i, filename in enumerate(dataFilenameList):

            if i >= frameNumber:
                break

            filepath = path.join(config.basePath, config.dataDir, filename)
            frameData, frameHeader = read_csv_data(filepath)

            # Figure out the size of the frame.
            # Count its width by finding where the value of 'y' changes the first time.
            frameWidth = 0
            yIndex = frameHeader.index('y')
            for row in frameData:
                if int(row[yIndex]) != int(frameData[0, yIndex]):
                    break
                frameWidth += 1

            frameSize = (frameWidth, int(frameData.shape[0] / frameWidth))
            frameData = frameData.reshape((frameSize[1], frameSize[0], -1)).swapaxes(0, 1)

            # Make sure we don't have any extra/missing data.
            assert(frameData.shape[2] == len(frameHeader))

            if h5Data is None:
                dataShape = (
                frameNumber, frameData.shape[0], frameData.shape[1], frameData.shape[2] + extraFeatureNumber)
                originalFeatureNumber = frameData.shape[2]
                h5Data = h5File.create_dataset('data', dataShape, dtype='float32')
                header = frameHeader

            # Check the data dimensions.
            expectedDataShape = (h5Data.shape[1], h5Data.shape[2], originalFeatureNumber)
            if frameData.shape == expectedDataShape:
                h5Data[i, ..., 0:originalFeatureNumber] = frameData

                # Extract the frame index from the filename (not all frames are there)
                regexMatch = re.search('[^\-]+-(\d+).*', filename)
                frameIndex = int(regexMatch.group(1))
                frameMap.append(frameIndex)
            else:
                raise RuntimeError("Frame data has wrong shape: {}, expected: {}".format(
                    frameData.shape, expectedDataShape))

            print("Read {}".format(filename))

        # Metadata describes all the frames, but we only have CSVs for some of them,
        # so discard the redundant metadata.
        metadata = metadata[frameMap, ...]

        # Metadata and the headers are resizable, since we will augment the data later.
        h5File.create_dataset('metadata', data=metadata, maxshape=(metadata.shape[0], None))
        h5File.create_dataset('metaheader', data=np.array(metaheader, dtype='|S128'), maxshape=(None,))
        h5File.create_dataset('header', data=np.array(header, dtype='|S128'), maxshape=(None,))
        h5File.create_dataset('frameMap', data=frameMap)
        h5File.create_group('arrayAttributes')

    else:
        print("Reading preloaded data from {}".format(preloadedDataPath))
        h5File = h5py.File(preloadedDataPath, 'r+')

    return Dataset(h5File)


def load_tiff_data(config: 'DataImportConfig'):
    """
    Loads data from TIFF and BMP images.
    A sequence of BMP images contains camera image at different time steps.
    Each TIFF image contains data for one feature, with multiple frames in each file.
    :return:
    """

    # Must specify the preloaded data file manually.
    if config.preloadedDataFilename is None:
        raise RuntimeError('Preloaded data filename is missing.')

    h5Filename = config.preloadedDataFilename
    h5File = None

    preloadedDataPath = path.join(config.preloadedDataDir, h5Filename)
    if config.reloadOriginalData or not path.isfile(preloadedDataPath):
        print('Loading data from TIFF files.')

        # We need to know the size of data that will be added later to allocate an hdf5 file.
        extraFeatureNumber = Dataset.get_data_feature_number()

        # Files with feature data, each contains N frames, one per time step.
        tiffFileMap = {
            'exx': 'epsxx.tif',
            'exy': 'epsxy.tif',
            'eyy': 'epsyy.tif',
            'u': 'u_smoothed.tif',
            'v': 'v_smoothed.tif',
            'sigma': 'zncc.tif',
        }

        # Peek into the files to determine the number of frames and size of 'data frames'.
        firstFilePath = path.join(config.basePath, config.dataDir, tiffFileMap['exx'])
        firstFileData = read_tiff_data(firstFilePath)
        frameNumber = firstFileData.shape[0]
        dataFrameSize = firstFileData.shape[1:3]

        # Peek into the camera images to determine camera image size. (Different from the data frame size.)
        cameraImageFiles = []
        for file in os.listdir(path.join(config.basePath, config.imageDir)):
            if file.endswith('.bmp'):
                cameraImageFiles.append(path.join(config.basePath, config.imageDir, file))

        firstCameraImage = skimage.util.img_as_float(Image.open(cameraImageFiles[0]))
        imageSize = firstCameraImage.shape

        # Figure out which camera pixels are described in the data files. (The data resolution is much smaller.)
        dataMappingStep = np.floor(np.asarray(imageSize) / np.asarray(dataFrameSize))
        pixelsCoveredByData = dataMappingStep * dataFrameSize
        minPixel = ((imageSize - pixelsCoveredByData) / 2).astype(np.int)
        maxPixel = minPixel + pixelsCoveredByData

        # There's no metadata accompanying tiff data, create empty arrays.
        metadata = np.empty((frameNumber, 0))
        metaheader = np.empty((0,))

        # Create and initialize the data file and buffers.
        h5File = h5py.File(preloadedDataPath, 'w')

        originalFeatureNumber = len(tiffFileMap) + 2  # Plus x and y features that are generated manually.
        dataShape = (frameNumber, dataFrameSize[0], dataFrameSize[1], originalFeatureNumber + extraFeatureNumber)

        h5Data = h5File.create_dataset('data', dataShape, dtype='float32')
        frameMap = np.arange(0, frameNumber)  # No missing frames for tiff data
        header = []

        # Load data for each available feature.
        featureIndex = -1
        for featureName, filename in tiffFileMap.items():
            featureIndex += 1

            filepath = path.join(config.basePath, config.dataDir, filename)
            featureData = read_tiff_data(filepath)
            header.append(featureName)

            # Check the data dimensions.
            if featureData.shape[0:3] == dataShape[0:3]:
                h5Data[..., featureIndex] = featureData
            else:
                raise RuntimeError("Frame data has wrong shape: {}, expected: {}".format(
                    featureData.shape[0:3], dataShape[0:3]))

            print("Read {}".format(filename))

        print("Generating X and Y features")
        # Manually generate the x and y features.
        header.append('x')
        header.append('y')

        # It's faster to generate the full arrays in memory, and then copy, then to write each value individually.
        xFeature = np.tile(np.arange(minPixel[0], maxPixel[0], dataMappingStep[0], dtype=np.float).reshape(-1, 1, 1),
                           (1, dataShape[2], 1))
        yFeature = np.tile(np.arange(minPixel[0], maxPixel[1], dataMappingStep[1], dtype=np.float).reshape(1, -1, 1),
                           (dataShape[1], 1, 1))
        featureIndex = originalFeatureNumber - 2
        h5Data[:, :, :, featureIndex:featureIndex + 2] = np.concatenate((xFeature, yFeature), axis=2)

        # Metadata and the headers are resizable, since we will augment the data later.
        h5File.create_dataset('metadata', data=metadata, maxshape=(metadata.shape[0], None))
        h5File.create_dataset('metaheader', data=np.array(metaheader, dtype='|S20'), maxshape=(None,))
        h5File.create_dataset('header', data=np.array(header, dtype='|S20'), maxshape=(None,))
        h5File.create_dataset('frameMap', data=frameMap)
        h5File.create_group('arrayAttributes')

    else:
        print("Reading preloaded data from {}".format(preloadedDataPath))
        h5File = h5py.File(preloadedDataPath, 'r+')

    return Dataset(h5File)