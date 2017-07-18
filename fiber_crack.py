import numpy as np

from numpy_extras import slice_along_axis

np.random.seed(13)  # Fix the seed for reproducibility.
# import tensorflow as tf
# tf.set_random_seed(13)

import time
import warnings
import inspect
import hashlib
import argparse

from data_loading import readDataFromCsv, readDataFromTiff
import os, math
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import colorsys
import skimage.measure, skimage.transform, skimage.util, skimage.filters, skimage.feature, skimage.morphology
import scipy.ndimage.morphology, scipy.stats
from PIL import Image
from os import path
import h5py

from volume_tools import write_volume_to_datraw

############### Configuration ###############

preloadedDataDir = 'C:/preloaded_data'
preloadedDataFilename = None  # By default, decide automatically.
dataFormat = 'csv'
imageFilenameFormat = '{}-{:04d}_0.tif'
outDir = 'T:/projects/SimtechOne/out/fiber'

dicKernelSize = 55
plotCrackAreaGroundTruth = False
crackAreaGroundTruthPath = ''

globalParams = {
    'textureKernelMultiplier': 1.0,
    'entropyThreshold': 1.0,
    'varianceThreshold': 0.003,
    'unmatchedPixelsPadding': 0.0,
    'unmatchedAndEntropyKernelMultiplier': 0.5,
    'exportedVolumeTimestepWidth': 3,
    'exportedVolumeSkippedFrames': 5
}

# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Steel-Epoxy'
# metadataFilename = 'Steel-Epoxy.csv'
# dataDir = 'data_export_tstep3'
# imageDir = 'raw_images'
# imageBaseName = 'Spec054'
# dicKernelSize = 85

# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Steel-Epoxy'
# metadataFilename = 'Steel-Epoxy.csv'
# dataDir = 'data_export'
# imageDir = 'raw_images'
# imageBaseName = 'Spec054'
# dicKernelSize = 85
# preloadedDataFilename = 'Steel-Epoxy-low-t-res.hdf5'

# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Steel-ModifiedEpoxy'
# metadataFilename = 'Steel-ModifiedEpoxy.csv'
# dataDir = 'data_export'
# imageDir = 'raw_images'
# imageBaseName = 'Spec010'
# dicKernelSize = 55

basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/PTFE-Epoxy'
metadataFilename = 'PTFE-Epoxy.csv'
dataDir = 'data_export'
# dataDir = 'data_export_fine'
imageDir = 'raw_images'
imageBaseName = 'Spec048'
dicKernelSize = 81
plotCrackAreaGroundTruth = True
crackAreaGroundTruthPath = 'spec_048_area.csv'

# # PTFE-Epoxy with fine spatial and temporal resolutions, mid-experiment.
# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Spec48'
# metadataFilename = '../PTFE-Epoxy/PTFE-Epoxy.csv'
# dataDir = 'data_grid_sparse_filtered'
# imageDir = '../PTFE-Epoxy/raw_images'
# imageBaseName = 'Spec048'
# preloadedDataFilename = 'PTFE-Epoxy-fine-mid.hdf5'
# dicKernelSize = 81

# dataFormat = 'tiff'
# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/micro_epoxy-hole'
# dataDir = ''
# imageDir = ''
# imageBaseName = 'test'
# imageFilenameFormat = '{}_{:03d}.bmp'
# metadataFilename = ''
# preloadedDataFilename = 'micro_epoxy-hole'

maxFrames = 99999
reloadOriginalData = False
recomputeResults = False


#############################################

class Dataset:

    @staticmethod
    def get_data_feature_number():
        """
        Specifies how many columns should be created when allocating an hdf5 array.
        :return:
        """
        # When allocating a continuous hdf5 file we need to know the final size
        # of the data in advance.

        augmentedFeatureNumber = 4
        resultsFeatureNumber = 15  # Approximately, leave some empty.

        return augmentedFeatureNumber + resultsFeatureNumber

    def __init__(self, h5File):
        self.h5Data = h5File['data']
        self.h5Header = h5File['header']
        self.h5FrameMap = h5File['frameMap']
        self.h5Metadata = h5File['metadata']
        self.h5Metaheader = h5File['metaheader']

    def unpack_vars(self):
        header = self.get_header()
        frameMap = self.h5FrameMap[:].tolist()
        metadata = self.h5Metadata[:].tolist()
        metaheader = self.get_metaheader()
        return self.h5Data, header, frameMap, metadata, metaheader

    def create_or_get_column(self, newColumnName):
        """
        Appends a new empty column to the data. If the column already exists, returns its index.

        Note: we do not insert the data directly, because typically we don't want to
        copy it into a single huge array. The size would be to big.
        :param newColumnName:
        :return:
        """

        # Check if the column already exists.
        header = self.get_header()
        if newColumnName in header:
            return header.index(newColumnName)

        # Make sure we still have preallocated space available.
        assert(self.h5Data.shape[-1] > self.h5Header.shape[0])

        self.h5Header.resize((self.h5Header.shape[0] + 1,))
        self.h5Header[-1] = newColumnName.encode('ascii')

        newColumnIndex = self.h5Header.shape[0] - 1
        return newColumnIndex

    def create_or_update_metadata_column(self, newColumnName, newColumnData):

        assert(newColumnData.shape[0] == self.h5Metadata.shape[0])

        # Check if the column already exists.
        metaheader = self.get_metaheader()
        if newColumnName not in metaheader:
            self.h5Metadata.resize(self.h5Metadata.shape[1] + 1, axis=1)
            self.h5Metaheader.resize(self.h5Metaheader.shape[0] + 1, axis=0)
            self.h5Metadata[:, -1] = newColumnData
            self.h5Metaheader[-1] = newColumnName.encode('ascii')

            return self.h5Metaheader.shape[0] - 1
        else:
            self.h5Metadata[:, metaheader.index(newColumnName)] = newColumnData
            return metaheader.index(newColumnName)

    def get_column_at_frame(self, frame, columnName) -> np.ndarray:
        return self.h5Data[frame, ..., self.get_header().index(columnName)]

    def get_data_image_mapping(self):
        return (self.get_attr('mappingMin'), self.get_attr('mappingMax'), self.get_attr('mappingStep'))

    def get_image_shift(self):
        metaheader = self.get_metaheader()
        indices = [metaheader.index('imageShiftX'), metaheader.index('imageShiftY')]
        return self.h5Metadata[:, indices].astype(np.int)

    def get_frame_number(self):
        return self.h5Data.shape[0]

    def get_frame_size(self):
        return tuple(self.h5Data.shape[1:3])

    def get_metadata_val(self, frame, columnName):
        return self.h5Metadata[frame, self.get_metaheader().index(columnName)]

    def get_metadata_column(self, columnName):
        return self.h5Metadata[:, self.get_metaheader().index(columnName)]

    def set_attr(self, attrName, attrValue):
        self.h5Data.attrs[attrName] = attrValue

    def get_attr(self, attrName):
        return self.h5Data.attrs[attrName]

    def has_attr(self, attrName):
        return attrName in self.h5Data.attrs

    def get_header(self):
        return self.h5Header[:].astype(np.str).tolist()

    def get_metaheader(self):
        return self.h5Metaheader[:].astype(np.str).tolist()

    def get_frame_map(self):
        return self.h5FrameMap[:].tolist()


def load_csv_data():
    if preloadedDataFilename is None:
        h5Filename = metadataFilename.replace('.csv', '.hdf5')
    else:
        h5Filename = preloadedDataFilename

    h5File = None

    preloadedDataPath = path.join(preloadedDataDir, h5Filename)
    if reloadOriginalData or not path.isfile(preloadedDataPath):
        print('Loading data from CSV files.')

        # We need to know the size of data that will be added later to allocate an hdf5 file.
        extraFeatureNumber = Dataset.get_data_feature_number()
        originalFeatureNumber = None

        # Load the metadata, describing each frame of the experiment.
        metadata, metaheader = readDataFromCsv(path.join(basePath, metadataFilename))

        dataFilenameList = os.listdir(path.join(basePath, dataDir))
        frameNumber = min(len(dataFilenameList), maxFrames)

        h5File = h5py.File(preloadedDataPath, 'w')
        h5Data = None
        frameMap = []
        header = None

        # Load data for each available frame.
        for i, filename in enumerate(dataFilenameList):

            if i >= frameNumber:
                break

            filepath = path.join(basePath, dataDir, filename)
            frameData, frameHeader = readDataFromCsv(filepath)

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

    else:
        print("Reading preloaded data from {}".format(preloadedDataPath))
        h5File = h5py.File(preloadedDataPath, 'r+')

    return Dataset(h5File)


def load_tiff_data():
    """
    Loads data from TIFF and BMP images.
    A sequence of BMP images contains camera image at different time steps.
    Each TIFF image contains data for one feature, with multiple frames in each file.
    :return:
    """

    # Must specify the preloaded data file manually.
    if preloadedDataFilename is None:
        raise RuntimeError('Preloaded data filename is missing.')

    h5Filename = preloadedDataFilename
    h5File = None

    preloadedDataPath = path.join(preloadedDataDir, h5Filename)
    if reloadOriginalData or not path.isfile(preloadedDataPath):
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
        firstFilePath = path.join(basePath, dataDir, tiffFileMap['exx'])
        firstFileData = readDataFromTiff(firstFilePath)
        frameNumber = firstFileData.shape[0]
        dataFrameSize = firstFileData.shape[1:3]

        # Peek into the camera images to determine camera image size. (Different from the data frame size.)
        cameraImageFiles = []
        for file in os.listdir(path.join(basePath, imageDir)):
            if file.endswith('.bmp'):
                cameraImageFiles.append(path.join(basePath, imageDir, file))

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

            filepath = path.join(basePath, dataDir, filename)
            featureData = readDataFromTiff(filepath)
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

    else:
        print("Reading preloaded data from {}".format(preloadedDataPath))
        h5File = h5py.File(preloadedDataPath, 'r+')

    return Dataset(h5File)


def load_data():
    if dataFormat == 'csv':
        return load_csv_data()
    elif dataFormat == 'tiff':
        return load_tiff_data()
    else:
        raise ValueError("Unknown data format: {}".format(dataFormat))


def rgba_to_rgb(rgba):
    a = rgba[3]
    return np.array([rgba[0] * a, rgba[1] * a, rgba[2] * a])


def color_map_hsv(X, Y, maxNorm):
    assert(X.ndim == 2)
    assert(X.shape == Y.shape)

    result = np.empty(X.shape + (3,))
    for i in range(0, X.shape[0]):
        for j in range(0, X.shape[1]):
            x = X[i, j]
            y = Y[i, j]

            norm = math.sqrt(x * x + y * y)
            angle = math.atan2(y, x)
            angle = angle if angle >= 0.0 else angle + 2 * math.pi  # Convert [-pi, pi] -> [0, 2*pi]
            hue = angle / (2 * math.pi)  # Map [0, 2*pi] -> [0, 1]
            v = norm / maxNorm
            rgb = colorsys.hsv_to_rgb(hue, 1.0, v)

            result[i, j, :] = rgb

    return result


def morphology_prune(data, iter):
    # Reference: http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm

    # The structuring element for pruning.
    # Here '2' means empty cell (match anything).
    selems = [
        np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 2, 2],])
    ]

    # Another basic selem is a mirrored version of the first one.
    selems.append(np.flip(selems[0], axis=0))
    # Append all rotations of the basic two selems.
    for i in range(0, 3):
        selems.append(np.rot90(selems[2 * i]))
        selems.append(np.rot90(selems[2 * i + 1]))

    output = data.copy()
    for i in range(0, iter):
        for selem in selems:
            removedPixels = scipy.ndimage.morphology.binary_hit_or_miss(output, selem == 1, selem == 0)
            output = np.logical_xor(output, removedPixels)

    return output


def image_variance_filter(data, windowRadius):
    windowLength = windowRadius * 2 + 1
    windowShape = (windowLength, windowLength)

    mean = scipy.ndimage.uniform_filter(data, windowShape)
    meanOfSquare = scipy.ndimage.uniform_filter(data ** 2, windowShape)
    return meanOfSquare - mean ** 2


def image_entropy_filter(data, windowRadius):
    dataUint = (data * 16).astype(np.uint8)
    windowLength = windowRadius * 2 + 1
    windowMask = np.ones((windowLength, windowLength), dtype=np.bool)
    # windowMask = skimage.morphology.disk(windowRadius)

    # Important: data range and the number of histogram bins must be equal (skimage expects it so).
    histograms = skimage.filters.rank.windowed_histogram(dataUint, selem=windowMask, n_bins=16)

    return np.apply_along_axis(scipy.stats.entropy, axis=2, arr=histograms)


def append_camera_image(dataset):
    """
    Appends a column containing grayscale data from the camera.
    The data is cropped from the frame according to the mapping between the data and the image.

    :param dataset:
    :return:
    """

    h5Data, header, frameMap, *r = dataset.unpack_vars()
    imageShift = dataset.get_image_shift()

    min, max, step = dataset.get_data_image_mapping()

    hasCameraImageMask = np.zeros((h5Data.shape[0]))

    columnIndex = dataset.create_or_get_column('camera')
    frameNumber = h5Data.shape[0]
    for f in range(0, frameNumber):
        print("Frame {}/{}".format(f, frameNumber))
        frameIndex = frameMap[f]
        cameraImagePath = path.join(basePath, imageDir, imageFilenameFormat.format(imageBaseName, frameIndex))
        cameraImageAvailable = os.path.isfile(cameraImagePath)
        if cameraImageAvailable:
            cameraImage = np.array(skimage.util.img_as_float(Image.open(cameraImagePath)))

            # Crop a rectangle from the camera image, accounting for the overall shift of the specimen.
            relMin = np.clip(min + imageShift[f, ...], [0, 0], cameraImage.shape)
            relMax = np.clip(max + imageShift[f, ...], [0, 0], cameraImage.shape)

            size = np.ceil((relMax - relMin) / step).astype(np.int)
            h5Data[f, 0:size[0], 0:size[1], columnIndex] = \
                cameraImage[relMin[1]:relMax[1]:step[1], relMin[0]:relMax[0]:step[0]].transpose()

            dataset.set_attr('cameraImageSize', cameraImage.shape)
            hasCameraImageMask[f] = True

    dataset.create_or_update_metadata_column('hasCameraImage', hasCameraImageMask)

    return dataset


def append_matched_pixels(dataset):
    h5Data, header, frameMap, *r = dataset.unpack_vars()
    imageShift = dataset.get_image_shift()
    frameSize = dataset.get_frame_size()
    min, max, step = dataset.get_data_image_mapping()

    matchedColumnIndex = dataset.create_or_get_column('matched')
    uBackColumnIndex = dataset.create_or_get_column('u_back')
    vBackColumnIndex = dataset.create_or_get_column('v_back')

    frameNumber = h5Data.shape[0]
    for f in range(0, frameNumber):
        matchedPixels = np.zeros(frameSize)
        backwardFlow = np.zeros(frameSize + (2,))
        print("Frame {}/{}".format(f, frameNumber))
        frameData = h5Data[f, :, :, :]
        frameFlow = frameData[:, :, [header.index('u'), header.index('v')]]
        frameMask = frameData[:, :, header.index('sigma')]
        for x in range(0, frameSize[0]):
            for y in range(0, frameSize[1]):
                # Don't consider pixels that lost tracking.
                if frameMask[x, y] < 0:
                    continue

                newX = x + int(round((frameFlow[x, y, 0] - imageShift[f, 0]) / step[0]))
                newY = y + int(round((frameFlow[x, y, 1] - imageShift[f, 1]) / step[1]))
                if (0 < newX < frameSize[0]) and (0 < newY < frameSize[1]):
                    matchedPixels[newX, newY] = 1.0
                    backwardFlow[newX, newY, :] = (imageShift[f, :] - frameFlow[x, y, :]) / step

        h5Data[f, ..., matchedColumnIndex] = matchedPixels
        h5Data[f, ..., [uBackColumnIndex, vBackColumnIndex]] = backwardFlow


def zero_pixels_without_tracking(dataset):
    h5Data, header, frameMap, *r = dataset.unpack_vars()
    # frameSize = dataset.get_frame_size()
    for f in range(0, h5Data.shape[0]):
        data = h5Data[f, ...]
        filter = data[..., header.index('sigma')] >= 0
        # flowIndices = [header.index('u'), header.index('v')]
        data[..., header.index('u')] *= filter
        data[..., header.index('v')] *= filter
        h5Data[f, ...] = data
        # todo is it faster to use an intermediate buffer? or should I write to hdf5 directly?


def plot_optic_flow(pdf, dataset, frameIndex):
    h5Data, header, frameMap, *r = dataset.unpack_vars()
    imageShift = dataset.get_image_shift()
    frameSize = dataset.get_frame_size()
    min, max, step = dataset.get_data_image_mapping()

    imageShift = imageShift[frameIndex]

    frameFlow = h5Data[frameIndex, :, :, :][:, :, [header.index('u'), header.index('v')]]
    frameFlow[:, :, 0] = np.rint((frameFlow[:, :, 0] - imageShift[0]) / step[0])
    frameFlow[:, :, 1] = np.rint((frameFlow[:, :, 1] - imageShift[1]) / step[1])


    # Shift the array to transform to the current frame coordinates.
    cropMin = np.maximum(np.rint(imageShift / step).astype(np.int), 0)
    cropMax = np.minimum(np.rint(frameSize + imageShift / step).astype(np.int), frameSize)
    cropSize = cropMax - cropMin
    plottedFlow = np.zeros(frameFlow.shape)
    plottedFlow[0:cropSize[0], 0:cropSize[1], :] = frameFlow[cropMin[0]:cropMax[0], cropMin[1]:cropMax[1], :]
    plottedFlow = plottedFlow.swapaxes(0, 1)

    X, Y = np.meshgrid(np.arange(0, frameSize[0]), np.arange(0, frameSize[1]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.quiver(X, Y, plottedFlow[:, :, 0], plottedFlow[:, :, 1], angles='uv', scale_units='xy', scale=1,
              width=0.003, headwidth=2)
    # pdf.savefig(fig)
    # plt.cla()
    plt.show()


def append_data_image_mapping(dataset):
    """
    Fetch the min/max pixel coordinates of the data, in original image space (2k*2k image)
    Important, since the data only covers every ~fifth pixel of some cropped subimage of the camera image.

    :param dataset:
    :return: (min, max, step)
    """
    h5Data, header, *r = dataset.unpack_vars()

    minX = int(h5Data[0, 0, 0, header.index('x')])
    maxX = int(h5Data[0, -1, 0, header.index('x')])
    minY = int(h5Data[0, 0, 0, header.index('y')])
    maxY = int(h5Data[0, 0, -1, header.index('y')])
    stepX = round((maxX - minX) / h5Data.shape[1])
    stepY = round((maxY - minY) / h5Data.shape[2])

    dataset.set_attr('mappingMin', np.array([minX, minY]))
    dataset.set_attr('mappingMax', np.array([maxX, maxY]))
    dataset.set_attr('mappingStep', np.array([stepX, stepY]))


def append_physical_frame_size(dataset):
    h5Data, header, *r = dataset.unpack_vars()

    # We cannot simply look at the corner pixels, since sometimes
    # the physical coordinates are reported incorrectly (at least near the edges).
    # Instead, find literally the minimum and maximum coordinates present in the data.
    xColumn = h5Data[0, :, :, header.index('X')]
    yColumn = h5Data[0, :, :, header.index('Y')]
    minX = np.min(xColumn)
    maxX = np.max(xColumn)
    minY = np.min(yColumn)
    maxY = np.max(yColumn)

    physicalDomain = np.array([[minX, minY], [maxX, maxY]])

    physicalSize = np.abs(np.array([maxX - minX, maxY - minY]))
    print('Computed physical size: {}'.format(physicalSize))
    print('Physical domain: {}'.format(physicalDomain))
    dataset.set_attr('physicalFrameSize', physicalSize)


def compute_avg_flow(dataset):
    """
    Sample the flow/shift (u,v) at a few points to determine the average shift of each frame
    relative to the base frame.

    :param dataset:
    :return:
    """

    h5Data, header, *r = dataset.unpack_vars()

    # Select points which will be sampled to determine the overall shift relative to the ref. frame
    # (This is used to align the camera image with the reference frame (i.e. data space)
    sampleX = np.linspace(0 + 20, h5Data.shape[1] - 20, 4)
    sampleY = np.linspace(0 + 20, h5Data.shape[2] - 20, 4)
    samplePointsX, samplePointsY = np.array(np.meshgrid(sampleX, sampleY)).astype(np.int)
    # Convert to an array of 2d points.
    samplePoints = np.concatenate((samplePointsX[:, :, np.newaxis], samplePointsY[:, :, np.newaxis]), 2).reshape(-1, 2)

    avgFlow = np.empty((h5Data.shape[0], 2))
    for f in range(0, h5Data.shape[0]):
        frameData = h5Data[f, ...]
        samples = np.array([frameData[tuple(p)] for p in samplePoints])
        uvSamples = samples[:, [header.index('u'), header.index('v'), header.index('sigma')]]
        uvSamples = uvSamples[uvSamples[:, 2] >= 0, :]  # Filter out points that lost tracking
        avgFlow[f, :] = np.mean(uvSamples[:, 0:2], axis=0).astype(np.int) if uvSamples.shape[0] > 0 else np.array(
            [0, 0])

    return avgFlow


def append_crack_from_unmatched_pixels(dataset, dicKernelRadius):

    frameWidth, frameHeight = dataset.get_frame_size()
    header = dataset.get_header()

    mappingMin, mappingMax, mappingStep = dataset.get_data_image_mapping()

    # Prepare columns for the results.
    index1 = dataset.create_or_get_column('matchedPixelsGauss')
    index2 = dataset.create_or_get_column('matchedPixelsGaussThres')
    index3 = dataset.create_or_get_column('matchedPixelsGaussThresClean')

    selem = scipy.ndimage.morphology.generate_binary_structure(2, 2)
    for frameIndex in range(0, dataset.get_frame_number()):
        frameData = dataset.h5Data[frameIndex, ...]
        matchedPixels = frameData[:, :, header.index('matched')]

        ### Gaussian smoothing.
        matchedPixelsGauss = skimage.filters.gaussian(matchedPixels, 2.0)

        ### Binary thresholding.
        thresholdBinary = lambda t: lambda x: 1.0 if x >= t else 0.0
        matchedPixelsGaussThres = np.vectorize(thresholdBinary(0.5))(matchedPixelsGauss)
        # matchedPixelsGaussThres = skimage.morphology.binary_dilation(matchedPixels, selem)

        ### Morphological filtering.

        # Suppress warnings from remove_small_objects/holes which occur when there's a single object/hole.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            maxObjectSize = int(frameWidth * frameHeight / 50)
            tempResult = skimage.morphology.remove_small_objects(
                matchedPixelsGaussThres.astype(np.bool), min_size=maxObjectSize)

            cropSelector = (slice(int(frameWidth * 0.1), int(frameWidth * 0.9)),
                            slice(int(frameHeight * 0.1), int(frameHeight * 0.9)))
            holePixelNumber = np.count_nonzero(tempResult[cropSelector] == False)
            tempResult = skimage.morphology.remove_small_holes(tempResult, min_size=holePixelNumber / 6.0)

            tempResult = skimage.morphology.binary_erosion(tempResult, selem)
            tempResult = skimage.morphology.binary_erosion(tempResult, selem)
            tempResult = skimage.morphology.remove_small_objects(tempResult, min_size=maxObjectSize)
            tempResult = skimage.morphology.binary_dilation(tempResult, selem)
            tempResult = skimage.morphology.binary_dilation(tempResult, selem)
            tempResult = skimage.morphology.binary_dilation(tempResult, selem)
            tempResult = skimage.morphology.binary_dilation(tempResult, selem)
            tempResult = skimage.morphology.remove_small_holes(tempResult, min_size=holePixelNumber / 6.0)

            # Don't erode back: instead, compensate for the kernel used during DIC.
            currentDilation = 2  # Because we dilated twice without eroding back.
            for i in range(currentDilation, dicKernelRadius + 1):
                tempResult = skimage.morphology.binary_dilation(tempResult, selem)

            matchedPixelsGaussThresClean = tempResult

        ### Write the results.
        dataset.h5Data[frameIndex, :, :, index1] = matchedPixelsGauss
        dataset.h5Data[frameIndex, :, :, index2] = matchedPixelsGaussThres
        dataset.h5Data[frameIndex, :, :, index3] = matchedPixelsGaussThresClean


def append_crack_from_variance(dataset, textureKernelSize, varianceThreshold=0.003):
    print("Computing cracks from variance.")

    frameWidth, frameHeight = dataset.get_frame_size()
    header = dataset.get_header()

    selem = scipy.ndimage.morphology.generate_binary_structure(2, 2)

    index1 = dataset.create_or_get_column('cameraImageVar')
    index2 = dataset.create_or_get_column('cameraImageVarFiltered')

    crackAreaData = np.zeros((dataset.get_frame_number()))
    for frameIndex in range(0, dataset.get_frame_number()):
        frameData = dataset.h5Data[frameIndex, ...]
        cameraImage = frameData[..., header.index('camera')]

        # Compute variance.
        cameraImageVar = image_variance_filter(cameraImage, textureKernelSize)

        # Threshold.
        varianceBinary = cameraImageVar < varianceThreshold

        # Clean up.
        varianceFiltered = varianceBinary.copy()
        for i in range(0, math.ceil(textureKernelSize / 2.0)):
            varianceFiltered = skimage.morphology.binary_dilation(varianceFiltered, selem)

        # Determine the crack area.
        if dataset.get_metadata_val(frameIndex, 'hasCameraImage'):
            totalArea = np.count_nonzero(varianceFiltered)
            crackAreaData[frameIndex] = totalArea

        dataset.h5Data[frameIndex, ..., index1] = cameraImageVar
        dataset.h5Data[frameIndex, ..., index2] = varianceFiltered

    frameArea = frameWidth * frameHeight
    physicalFrameSize = dataset.get_attr('physicalFrameSize')
    physicalFrameArea = physicalFrameSize[0] * physicalFrameSize[1]

    dataset.create_or_update_metadata_column('crackAreaVariance', crackAreaData)
    dataset.create_or_update_metadata_column('crackAreaVariancePhysical', crackAreaData / frameArea * physicalFrameArea)


def append_crack_from_entropy(dataset, textureKernelSize, entropyThreshold=1.0):
    # todo a lot of repetition between the variance and the entropy implementations.
    # refactor, if more features are added.
    print("Computing cracks from entropy.")

    frameWidth, frameHeight = dataset.get_frame_size()
    header = dataset.get_header()

    selem = scipy.ndimage.morphology.generate_binary_structure(2, 2)

    index1 = dataset.create_or_get_column('cameraImageEntropy')
    index2 = dataset.create_or_get_column('cameraImageEntropyFiltered')

    crackAreaData = np.zeros((dataset.get_frame_number()))
    for frameIndex in range(0, dataset.get_frame_number()):
        frameData = dataset.h5Data[frameIndex, ...]
        cameraImage = frameData[..., header.index('camera')]

        #  Compute entropy.
        cameraImageEntropy = image_entropy_filter(cameraImage, textureKernelSize)

        # Threshold.
        entropyBinary = cameraImageEntropy < entropyThreshold

        # Clean up.
        entropyFiltered = entropyBinary.copy()
        for i in range(0, math.ceil(textureKernelSize / 2.0)):
            entropyFiltered = skimage.morphology.binary_dilation(entropyFiltered, selem)

        # Determine the crack area.
        if dataset.get_metadata_val(frameIndex, 'hasCameraImage'):
            totalArea = np.count_nonzero(entropyFiltered)
            crackAreaData[frameIndex] = totalArea

        dataset.h5Data[frameIndex, ..., index1] = cameraImageEntropy
        dataset.h5Data[frameIndex, ..., index2] = entropyFiltered

    frameArea = frameWidth * frameHeight
    physicalFrameSize = dataset.get_attr('physicalFrameSize')
    physicalFrameArea = physicalFrameSize[0] * physicalFrameSize[1]

    dataset.create_or_update_metadata_column('crackAreaEntropy', crackAreaData)
    dataset.create_or_update_metadata_column('crackAreaEntropyPhysical', crackAreaData / frameArea * physicalFrameArea)


def append_crack_from_unmatched_and_entropy(dataset, textureKernelSize, unmatchedAndEntropyKernelMultiplier,
                                            entropyThreshold, unmatchedPixelsPadding=0.1):

    header = dataset.get_header()
    selem = scipy.ndimage.morphology.generate_binary_structure(2, 2)

    frameWidth, frameHeight = dataset.get_frame_size()
    # Use a smaller entropy kernel size, since we narrow the search area using unmatched pixels.
    entropyFilterRadius = int(textureKernelSize * unmatchedAndEntropyKernelMultiplier)

    index = dataset.create_or_get_column('cracksFromUnmatchedAndEntropy')

    crackAreaData = np.zeros((dataset.get_frame_number()))
    for frameIndex in range(0, dataset.get_frame_number()):
        frameData = dataset.h5Data[frameIndex, :, :, :]
        # Fetch the unmatched pixels.
        unmathedPixels = frameData[..., header.index('matchedPixelsGaussThresClean')]  == 0

        # Manually remove the unmatched pixels that are pushing into the image from the sides.
        if unmatchedPixelsPadding > 0.0:
            paddingWidth = int(frameWidth * unmatchedPixelsPadding)
            unmathedPixels[:paddingWidth, :] = False
            unmathedPixels[-paddingWidth:, :] = False

        # Dilate the unmatched pixels crack, and later use it as a search filter.
        unmathedPixels = skimage.morphology.binary_dilation(unmathedPixels, selem)
        unmathedPixels = skimage.morphology.binary_dilation(unmathedPixels, selem)
        unmathedPixels = skimage.morphology.binary_dilation(unmathedPixels, selem)

        # todo reuse entropy code (copy-pasting right now), but note that we might need a different kernel size.
        imageEntropy = image_entropy_filter(frameData[..., header.index('camera')], int(entropyFilterRadius))
        entropyBinary = imageEntropy < entropyThreshold

        entropyFiltered = entropyBinary.copy()
        for i in range(0, math.ceil(entropyFilterRadius / 2.0)):
            entropyFiltered = skimage.morphology.binary_dilation(entropyFiltered, selem)

        # Crack = low entropy near unmatched pixels.
        cracks = np.logical_and(unmathedPixels, entropyFiltered)

        # Determine the crack area.
        if dataset.get_metadata_val(frameIndex, 'hasCameraImage'):
            totalArea = np.count_nonzero(cracks)
            crackAreaData[frameIndex] = totalArea

        dataset.h5Data[frameIndex, ..., index] = cracks

    frameArea = frameWidth * frameHeight
    physicalFrameSize = dataset.get_attr('physicalFrameSize')
    physicalFrameArea = physicalFrameSize[0] * physicalFrameSize[1]

    dataset.create_or_update_metadata_column('crackAreaUnmatchedAndEntropy', crackAreaData)
    dataset.create_or_update_metadata_column('crackAreaUnmatchedAndEntropyPhysical', crackAreaData / frameArea * physicalFrameArea)


def append_reference_frame_crack(dataset, dicKernelRadius):

    frameSize = dataset.get_frame_size()

    index1 = dataset.create_or_get_column('sigmaFiltered')
    index2 = dataset.create_or_get_column('sigmaSkeleton')
    index3 = dataset.create_or_get_column('sigmaSkeletonPruned')

    for frameIndex in range(0, dataset.get_frame_number()):

        # Get the original sigma plot, with untrackable pixels as 'ones' (cracks).
        binarySigma = dataset.get_column_at_frame(frameIndex, 'sigma') < 0

        binarySigmaFiltered = binarySigma.copy()
        maxObjectSize = int(frameSize[0] * frameSize[1] / 1000)

        # Erode the cracks to remove smaller noise.
        selem = scipy.ndimage.morphology.generate_binary_structure(binarySigma.ndim, 2)
        for i in range(0, math.ceil(dicKernelRadius / 2)):
            binarySigmaFiltered = skimage.morphology.binary_erosion(binarySigmaFiltered, selem)

        # Remove tiny disconnected chunks of cracks as unnecessary noise.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            binarySigmaFiltered = skimage.morphology.remove_small_objects(binarySigmaFiltered, maxObjectSize)

        # Compute the skeleton.
        binarySigmaSkeleton = skimage.morphology.skeletonize(binarySigmaFiltered)

        # Prune the skeleton from small branches.
        binarySigmaSkeletonPruned = morphology_prune(binarySigmaSkeleton, int(frameSize[0] / 100))

        dataset.h5Data[frameIndex, ..., index1] = binarySigmaFiltered
        dataset.h5Data[frameIndex, ..., index2] = binarySigmaSkeleton
        dataset.h5Data[frameIndex, ..., index3] = binarySigmaSkeletonPruned


def apply_function_if_code_changed(dataset, function):
    """
    Calls a function that computes and writes data to the dataset.
    Stores the has of the function's source code as metadata.
    If the function has not changed, it isn't applied to the data.

    :param dataset:
    :param function:
    :return:
    """
    # Get a string containing the full function source.
    sourceLines = inspect.getsourcelines(function)
    functionSource = ''.join(sourceLines[0])
    functionName = function.__name__

    callSignature = inspect.signature(function)
    callArguments = {}
    for callParameter in callSignature.parameters:
        if callParameter in globalParams:
            callArguments[callParameter] = globalParams[callParameter]

    callArgumentsString = ''.join([key + str(callArguments[key]) for key in sorted(callArguments)])

    attrName = '_functionHash_' + functionName
    currentHash = hashlib.sha1((functionSource + callArgumentsString).encode('utf-8')).hexdigest()

    if 'cameraImageVar' in dataset.get_header() and not recomputeResults:
        oldHash = dataset.get_attr(attrName) if dataset.has_attr(attrName) else None

        if currentHash == oldHash:
            print("Function {} has not changed, skipping.".format(functionName))
            return

    print("Applying function {} to the dataset.".format(functionName))

    callArguments['dataset'] = dataset
    function(**callArguments)

    dataset.set_attr(attrName, currentHash)


def augment_data(dataset):
    header = dataset.get_header()
    metaheader = dataset.get_metaheader()

    if 'imageShiftX' in metaheader and 'camera' in header and 'matched' in header:
        print("Data already augmented, skipping.")
        return dataset

    # For now we require that all the data is present, or none of it.
    assert('imageShiftX' not in metaheader)
    assert('camera' not in header)
    assert('matched' not in header)

    # Add the data to image mapping to the dataset.
    append_data_image_mapping(dataset)

    # Add the physical dimensions of the data (in millimeters).
    append_physical_frame_size(dataset)

    # Add the image shift to the metadata.
    imageShift = compute_avg_flow(dataset)

    dataset.create_or_update_metadata_column('imageShiftX', imageShift[..., 0])
    dataset.create_or_update_metadata_column('imageShiftY', imageShift[..., 1])

    print("Adding the camera image...")
    append_camera_image(dataset)
    print("Adding the matched pixels...")
    append_matched_pixels(dataset)

    print("Zeroing the pixels that lost tracking.")
    zero_pixels_without_tracking(dataset)

    return dataset


def compute_and_append_results(dataset):
    # Compute derived parameters.
    mappingMin, mappingMax, mappingStep = dataset.get_data_image_mapping()
    dicKernelRadius = int((dicKernelSize - 1) / 2 / mappingStep[0])
    globalParams['dicKernelRadius'] = dicKernelRadius
    globalParams['textureKernelSize'] = int(dicKernelRadius * globalParams['textureKernelMultiplier'])

    apply_function_if_code_changed(dataset, append_crack_from_unmatched_pixels)

    apply_function_if_code_changed(dataset, append_crack_from_variance)
    apply_function_if_code_changed(dataset, append_crack_from_entropy)

    apply_function_if_code_changed(dataset, append_crack_from_unmatched_and_entropy)
    apply_function_if_code_changed(dataset, append_reference_frame_crack)


def plot_contour_overlay(axes, backgroundImage, binaryImage):
    axes.imshow(backgroundImage.transpose(), origin='lower', cmap='gray')
    entropyContours = skimage.measure.find_contours(binaryImage.transpose(), 0.5)
    for n, contour in enumerate(entropyContours):
        axes.plot(contour[:, 1], contour[:, 0], linewidth=1, color='white')


def plot_original_data_for_frame(axes, frameData, header):
    if 'W' in header:
        imageData0 = frameData[:, :, header.index('W')]
        axes[0].imshow(imageData0.transpose(), origin='lower', cmap='gray')
    axes[1].imshow(color_map_hsv(frameData[..., header.index('u')],
                                 frameData[..., header.index('v')], maxNorm=50.0)
                   .swapaxes(0, 1), origin='lower')
    # print("Sigma-plot")
    axes[2].imshow(frameData[:, :, header.index('sigma')].transpose(), origin='lower', cmap='gray')
    # print("Camera image")
    cameraImage = frameData[:, :, header.index('camera')]
    axes[3].imshow(cameraImage.transpose(), origin='lower', cmap='gray')


def plot_unmatched_cracks_for_frame(axes, frameData, header):
    matchedPixels = frameData[..., header.index('matched')]
    cameraImageData = frameData[..., header.index('camera')]

    axes[0].imshow(matchedPixels.transpose(), origin='lower', cmap='gray')
    matchedPixelsGauss = frameData[..., header.index('matchedPixelsGauss')]

    axes[1].imshow(matchedPixelsGauss.transpose(), origin='lower', cmap='gray')
    matchedPixelsGaussThres = frameData[..., header.index('matchedPixelsGaussThres')]

    axes[2].imshow(matchedPixelsGaussThres.transpose(), origin='lower', cmap='gray')
    matchedPixelsGaussThresClean = frameData[..., header.index('matchedPixelsGaussThresClean')]

    axes[3].imshow(matchedPixelsGaussThresClean.transpose().astype(np.float), origin='lower', cmap='gray')

    plot_contour_overlay(axes[4], cameraImageData, matchedPixelsGaussThresClean)


def plot_image_cracks_for_frame(axes, frameData, header):
    cameraImageData = frameData[..., header.index('camera')]

    # Variance-based camera image crack extraction.
    cameraImageVar = frameData[..., header.index('cameraImageVar')]
    varianceFiltered = frameData[..., header.index('cameraImageVarFiltered')]

    axes[0].imshow(cameraImageVar.transpose(), origin='lower', cmap='gray')
    plot_contour_overlay(axes[1], cameraImageData, varianceFiltered)

    # Entropy-based camera image crack extraction.
    cameraImageEntropy = frameData[..., header.index('cameraImageEntropy')]
    entropyFiltered = frameData[..., header.index('cameraImageEntropyFiltered')]

    axes[2].imshow(cameraImageEntropy.transpose(), origin='lower', cmap='gray')
    plot_contour_overlay(axes[3], cameraImageData, entropyFiltered)

    # Unmatched pixels + entropy crack extraction.
    cracksFromUnmatchedAndEntropy = frameData[..., header.index('cracksFromUnmatchedAndEntropy')]
    plot_contour_overlay(axes[4], cameraImageData, cracksFromUnmatchedAndEntropy)


def plot_reference_crack_for_frame(axes, frameData, header):

    binarySigmaFiltered = frameData[..., header.index('sigmaFiltered')]
    binarySigmaSkeleton = frameData[..., header.index('sigmaSkeleton')]
    binarySigmaSkeletonPruned = frameData[..., header.index('sigmaSkeletonPruned')]

    axes[0].imshow(binarySigmaFiltered.transpose(), origin='lower', cmap='gray')
    axes[1].imshow(binarySigmaSkeleton.transpose(), origin='lower', cmap='gray')
    axes[2].imshow(binarySigmaSkeletonPruned.transpose(), origin='lower', cmap='gray')


def plot_feature_histograms_for_frame(axes, frameData, header):
    cameraImageVar = frameData[..., header.index('cameraImageVar')]
    cameraImageEntropy = frameData[..., header.index('cameraImageEntropy')]

    # Plot variance and entropy histograms for the image.
    varianceHist = np.histogram(cameraImageVar, bins=16, range=(0, np.max(cameraImageVar)))[0]
    entropyHist = np.histogram(cameraImageEntropy, bins=16, range=(0, np.max(cameraImageEntropy)))[0]

    varHistDomain = np.linspace(0, np.max(cameraImageVar), 16)
    entropyHistDomain = np.linspace(0, np.max(cameraImageEntropy), 16)

    axes[0].bar(varHistDomain * 100, varianceHist)
    axes[1].bar(entropyHistDomain, entropyHist)

    axes[0].axis('on')
    axes[1].axis('on')
    axes[0].get_yaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)


def plot_crack_area_chart(dataset):
    frameMap = dataset.get_frame_map()

    fig = plt.figure()
    fig.suptitle("Crack area in the current frame")
    ax = fig.add_subplot(1, 1, 1)

    ax.grid(which='both')
    ax.set_ylabel('Millimeter^2')

    # Plot the estimated crack area.
    ax.plot(dataset.get_metadata_column('crackAreaVariancePhysical'), label='Variance estimation')
    ax.plot(dataset.get_metadata_column('crackAreaEntropyPhysical'), label='Entropy  estimation')
    ax.plot(dataset.get_metadata_column('crackAreaUnmatchedAndEntropyPhysical'), label='Unmatched&Entropy  estimation')

    # A hacky way to convert x axis labels from frame indices to frame numbers.
    locs, labels = plt.xticks()
    for i in range(len(labels)):
        frameIndex = int(locs[i])
        frameNumber = str(frameMap[frameIndex]) if 0 <= frameIndex < len(frameMap) else ''
        labels[i].set_text(frameNumber)
        plt.xticks(locs, labels)

    if (plotCrackAreaGroundTruth):
        # Convert frame numbers into frame indices, i.e. align with X axis.
        def get_closest_frame_index(frameNumber):
            return np.count_nonzero(np.array(frameMap, dtype=np.int)[:-1] < frameNumber)

        groundTruth = np.genfromtxt(os.path.join(basePath, crackAreaGroundTruthPath), delimiter=',')
        groundTruth[:, 0] = [get_closest_frame_index(frameNumber) for frameNumber in groundTruth[:, 0]]

        # Compute crack area percentage error for each frame with ground truth available.
        percentageError = np.zeros(groundTruth.shape[0])
        crackAreaEntropy = dataset.get_metadata_column('crackAreaUnmatchedAndEntropyPhysical')
        for i, frameIndex in enumerate(groundTruth[:, 0]):
            truth = groundTruth[i, 1]
            estimated = crackAreaEntropy[int(frameIndex)]
            percentageError[i] = math.fabs((truth - estimated) / truth) * 100

        # Plot the ground truth.
        ax.scatter(groundTruth[:, 0], groundTruth[:, 1], marker='x', label='Measured by hand')

        # Plot the percentage error on the second Y-axis.
        ax2 = ax.twinx()
        ax2.scatter(groundTruth[:, 0], percentageError, marker='o', s=8, c='green', label='Percentage error (entropy)')
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel('Error, %')

    plt.legend()

    return fig


def plot_data_mapping(dataset):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    mappingText = 'Data to camera image mapping\n'
    mappingText += 'Data size: {} Image size: {}\n'.format(
        list(dataset.get_frame_size()), dataset.get_attr('cameraImageSize'))
    mappingText += 'Physical size: {}\n'.format(dataset.get_attr('physicalFrameSize'))
    mappingText += 'Min: {} Max: {} Step: {}\n'.format(*dataset.get_data_image_mapping())

    ax.text(0.1, 0.1, mappingText)

    return fig


def plot_data(dataset):
    h5Data, header, frameMap, *r = dataset.unpack_vars()

    # Prepare for plotting
    pdfPath = os.path.join(outDir, 'fiber-crack.pdf')
    print("Plotting to {}".format(pdfPath))
    pdf = PdfPages(pdfPath)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Prepare a figure with subfigures.
    fig = plt.figure()
    axes = []
    for f in range(0, 20):
        axes.append(fig.add_subplot(4, 5, f + 1))
        axes[f].axis('off')

    fig.subplots_adjust(hspace=0.025, wspace=0.025)

    # Draw the frame plots.
    for f in range(0, dataset.get_frame_number()):
        timeStart = time.time()
        frameIndex = frameMap[f]
        print("Plotting frame {}".format(frameIndex))
        fig.suptitle("Frame {}".format(frameIndex))

        frameData = h5Data[f, :, :, :]

        plot_original_data_for_frame(axes[0:5], frameData, header)
        plot_unmatched_cracks_for_frame(axes[5:10], frameData, header)
        plot_image_cracks_for_frame(axes[10:15], frameData, header)
        plot_reference_crack_for_frame(axes[15:18], frameData, header)
        plot_feature_histograms_for_frame(axes[18:20], frameData, header)

        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        for a in axes:
            a.clear()
            a.axis('off')

        print("Rendered in {:.2f} s.".format(time.time() - timeStart))

    # Crack area figure.
    fig = plot_crack_area_chart(dataset)
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    # Print the data-to-camera mapping.
    fig = plot_data_mapping(dataset)
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    pdf.close()


def export_crack_volume(dataset: 'Dataset'):
    """
    Build a volume by concatenating crack areas from each frame,
    save to the disk in datraw format.
    :param dataset:
    :return:
    """
    frameSize = dataset.get_frame_size()

    framesToSkip = globalParams['exportedVolumeSkippedFrames']
    frameWidth = globalParams['exportedVolumeTimestepWidth']
    frameNumber = dataset.get_frame_number() - framesToSkip
    # The volume should be exported in Z,Y,X C-order.
    volume = np.empty((frameNumber * frameWidth, frameSize[1], frameSize[0]), dtype=np.uint8)

    for f in range(0, frameNumber):
        crackArea = dataset.get_column_at_frame(f, 'cracksFromUnmatchedAndEntropy')
        crackAreaUint8 = np.zeros_like(crackArea, dtype=np.uint8)
        crackAreaUint8[crackArea == 1.0] = 255

        volumeSlabSelector = slice_along_axis(slice(f * frameWidth, f * frameWidth + frameWidth), 0, volume.ndim)
        volume[volumeSlabSelector] = crackAreaUint8.transpose()

    volume = scipy.ndimage.filters.gaussian_filter(volume, 0.5)

    write_volume_to_datraw(volume, os.path.join(outDir, 'crack-volume.raw'))


def main():

    # Parse the arguments.
    parser = argparse.ArgumentParser('Fiber crack.')
    parser.add_argument('-c', '--command', default='plot', choices=['plot', 'export-crack-volume'])
    args = parser.parse_args()

    timeStart = time.time()
    print("Loading the data.")
    dataset = load_data()
    print("Data loaded in {:.3f} s. Shape: {} Columns: {}".format(time.time() - timeStart, dataset.h5Data.shape, dataset.get_header()))

    timeStart = time.time()
    print("Augmenting the data.")
    augment_data(dataset)
    print("Data augmented in {:.3f} s.".format(time.time() - timeStart))

    timeStart = time.time()
    compute_and_append_results(dataset)
    print("Results computed and appended in {:.3f} s.".format(time.time() - timeStart))

    timeStart = time.time()
    print("Executing command: {}".format(args.command))
    commandMap = {
        'plot': lambda: plot_data(dataset),
        'export-crack-volume': lambda: export_crack_volume(dataset),
    }
    commandMap[args.command]()
    print("Command executed in {:.3f} s.".format(time.time() - timeStart))

    # timeStart = time.time()
    # print("Making a prediction.")
    # predict(dataset)
    # print("Prediction finished in {:.3f} s.".format(time.time() - timeStart))

    # https://github.com/tensorflow/tensorflow/issues/3388
    # keras.backend.clear_session()


main()
