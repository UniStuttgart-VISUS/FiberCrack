import numpy as np

np.random.seed(13)  # Fix the seed for reproducibility.
# import tensorflow as tf
# tf.set_random_seed(13)

import time
import warnings

from data_loading import readDataFromCsv, readDataFromTiff
import os, math
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import colorsys
import pickle
import keras
import skimage.measure, skimage.transform, skimage.util, skimage.filters, skimage.feature, skimage.morphology
import scipy.ndimage.morphology, scipy.stats
from keras.layers import Dense
from keras.constraints import maxnorm
from PIL import Image
from os import path
import h5py

from numpy_extras import slice_nd
from Normalizer import Normalizer
from image_tools import masked_gaussian_filter

############### Configuration ###############

preloadedDataDir = 'C:/preloaded_data'
preloadedDataFilename = None  # By default, decide automatically.
dataFormat = 'csv'
imageFilenameFormat = '{}-{:04d}_0.tif'

dicKernelSize = 55

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


#############################################

# todo Sigma-filtering when searching for unmatched pixels
# todo Strip NaNs from the metadata/metaheader
# todo PTFE-Epoxy frame 4465 data is incomplete.


class Dataset:

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

    def append_column(self, newColumnName):
        # Make sure we still have preallcoated space available.
        assert(self.h5Data.shape[-1] > self.h5Header.shape[0])

        self.h5Header.resize((self.h5Header.shape[0] + 1,))
        self.h5Header[-1] = newColumnName.encode('ascii')

        newColumnIndex = self.h5Header.shape[0] - 1
        return newColumnIndex

    def get_image_shift(self):
        metaheader = self.get_metaheader()
        indices = [metaheader.index('imageShiftX'), metaheader.index('imageShiftY')]
        return self.h5Metadata[:, indices].astype(np.int)

    def get_frame_size(self):
        return tuple(self.h5Data.shape[1:3])

    def get_header(self):
        return self.h5Header[:].astype(np.str).tolist()

    def get_metaheader(self):
        return self.h5Metaheader[:].astype(np.str).tolist()


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
        extraFeatureNumber = augment_data_extra_feature_number()
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
        h5File.create_dataset('metaheader', data=np.array(metaheader, dtype='|S20'), maxshape=(None,))
        h5File.create_dataset('header', data=np.array(header, dtype='|S20'), maxshape=(None,))
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
        extraFeatureNumber = augment_data_extra_feature_number()

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
    # return np.sum(histograms, axis=2)
    # histograms = histograms.astype(np.float)
    return np.apply_along_axis(scipy.stats.entropy, axis=2, arr=histograms)


def train_net(XTrain, yTrain, XVal, yVal, patchSize):
    timeStart = time.time()
    model = keras.models.Sequential()

    # model.add(
    #     Convolution3D(32, patchSize[0], 3, 3, input_shape=(patchSize[0], patchSize[1], patchSize[2], XTrain.shape[-1]),
    #                   border_mode="same",
    #                   activation="relu", W_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    # model.add(Convolution3D(32, patchSize[0], 3, 3, activation="relu", border_mode="same"))
    # model.add(MaxPooling3D(pool_size=(patchSize[0], 2, 2)))
    # model.add(Convolution3D(64, 1, 3, 3, activation="relu", border_mode="same"))
    # # model.add(Dropout(0.2))
    # # model.add(Convolution3D(64, patchSize[0], 3, 3, activation="relu", border_mode="same"))
    # # model.add(MaxPooling3D(pool_size=(patchSize[0], 2, 2)))
    # # model.add(Convolution3D(128, 3, 3, activation="relu", border_mode="same"))
    # # model.add(Dropout(0.2))
    # # model.add(Convolution3D(128, 3, 3, activation="relu", border_mode="same"))
    # # model.add(MaxPooling3D(pool_size=(2, 2)))
    # model.add(Flatten())
    #
    # model.add(Dropout(0.2))
    # model.add(Dense(256, activation="relu", W_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    # model.add(Dense(128, activation="relu", W_constraint=maxnorm(3)))
    # model.add(Dropout(0.2))
    # model.add(Dense(1))
    #
    # model.add(Dense(512, input_dim=XTrain.shape[-1], activation="relu", W_constraint=maxnorm(3)))
    # model.add(Dense(256, activation="relu", W_constraint=maxnorm(3)))
    # model.add(Dense(1))

    model.add(Dense(256, input_dim=XTrain.shape[-1], activation="relu", W_constraint=maxnorm(3)))
    model.add(Dense(128, activation="relu", W_constraint=maxnorm(3)))
    model.add(Dense(32, activation="relu", W_constraint=maxnorm(3)))
    model.add(Dense(1))

    epochNum = 5
    learningRate = 0.01
    batchSize = 32

    optimizer = keras.optimizers.RMSprop(lr=learningRate)
    model.compile(loss='mse', optimizer=optimizer)
    print("Finished compiling the net in {:.3f} s.".format(time.time() - timeStart))

    history = model.fit(XTrain, yTrain, validation_data=(XVal, yVal), nb_epoch=epochNum, batch_size=batchSize,
                        verbose=0)

    print("Finished training a net in {:.3f} s.)".format(time.time() - timeStart))
    print("Training loss: {}".format(history.history['loss']))
    print("Val      loss: {}".format(history.history['val_loss']))

    return model, history


def predict_frame(data, header, targetFrame, timeWindow, targetFeature, features):
    patchSize = (4, 16, 16)
    # Whether should collapse all spatial and temporal dimensions and use a 1D vector representation.
    flattenInput = True

    featureIndices = [header.index(f) for f in features]
    data = data[targetFrame - timeWindow:targetFrame + 1, :, :, featureIndices]

    patchNumber = (
        (data.shape[0] - patchSize[0] + 1 - 1),  # We want to predict the next frame, so we can't use the last frame // todo actually, I think I should include it, since we later 'cut off' the test data.
        (data.shape[1] - patchSize[1] + 1),
        (data.shape[2] - patchSize[2] + 1))
    patchNumberFlat = patchNumber[0] * patchNumber[1] * patchNumber[2]
    netDataX = np.empty((patchNumberFlat, patchSize[0], patchSize[1], patchSize[2], len(features)))
    netDataY = np.empty((patchNumberFlat, 1))

    for t in range(0, patchNumber[0]):
        for x in range(0, patchNumber[1]):
            for y in range(0, patchNumber[2]):
                patchIndex = t * patchNumber[1] * patchNumber[2] + x * patchNumber[2] + y
                netDataX[patchIndex, :, :, :, :] = data[t:t + patchSize[0], x:x + patchSize[1], y:y + patchSize[2], :]
                netDataY[patchIndex, 0] = data[t + patchSize[0],
                                               x + int(patchSize[1] / 2),
                                               y + int(patchSize[2] / 2),
                                               features.index(targetFeature)]

    if flattenInput:
        netDataX = netDataX.reshape((patchNumberFlat, -1))
        netDataY = netDataY.reshape((patchNumberFlat, -1))

    startTestIndex = (patchNumber[0] - 1) * patchNumber[1] * patchNumber[2]

    trainSetRatio = 0.95
    permutation = np.random.permutation(startTestIndex)
    trainIndices = permutation[:int(netDataX.shape[0] * trainSetRatio) - 1]
    valIndices = permutation[:int(netDataY.shape[0] * (1.0 - trainSetRatio))]

    XTrain = netDataX[trainIndices]
    yTrain = netDataY[trainIndices]
    XVal = netDataX[valIndices]
    yVal = netDataY[valIndices]

    normalizerX = Normalizer().fit(XTrain, XTrain.ndim - 1)
    XTrain = normalizerX.scale(XTrain)
    # yTrain = normalizer.scale(yTrain)
    XVal = normalizerX.scale(XVal)
    # yVal = normalizer.scale(yVal)

    model, history = train_net(XTrain, yTrain, XVal, yVal, patchSize)
    XTest = normalizerX.scale(netDataX[startTestIndex:,...])
    yTest = netDataY[startTestIndex:, ...]

    prediction = model.predict(XTest)
    testScore = model.evaluate(XTest, yTest, verbose=0)
    print("Test score: {}".format(testScore))

    # Reshape for plotting a 2D image.
    prediction = np.array(prediction).reshape((patchNumber[1], -1))

    # We can't make a prediction near image edges, pad the prediction.
    predictionImage = np.zeros((data.shape[1], data.shape[2]))
    patchRX = int(patchSize[1] / 2)
    patchRY = int(patchSize[2] / 2)
    predictionImage[patchRX:data.shape[1] - patchSize[1] + patchRX + 1,
    patchRY:data.shape[2] - patchSize[2] + patchRY + 1] = prediction
    predictionImage = predictionImage.transpose()

    return predictionImage, testScore


def predict(dataset):
    raise RuntimeError('Code not tested since refactoring to H5Py.')

    timeWindow = 6

    data, header, frameMap, *r = dataset.unpack_vars()

    vmin = -0.5
    vmax = 0.5
    pdf = PdfPages('..\out\prediction.pdf')
    fig = plt.figure()
    axes = [fig.add_subplot(2, 4, i + 1) for i in range(0, 8)]

    for frameIndex in range(int(data.shape[0] / 2 + timeWindow - 1), int(data.shape[0] / 2)):
    # for frameIndex in range(timeWindow + 1, data.shape[0]):
        print("Making a prediction for frame {}".format(frameIndex))
        timeStart = time.time()

        # Train a model and make a prediction.
        predictionImageW, testScoreW = predict_frame(data, header, frameIndex, timeWindow, 'W',
                                                     ['U', 'V', 'sigma', 'W'])
        predictionImageSigma, testScoreSigma = predict_frame(data, header, frameIndex, timeWindow, 'sigma',
                                                             ['U', 'V', 'sigma', 'W'])

        # Plot the actual data.
        frameData = data[frameIndex, :, :, :].swapaxes(0, 1)  # Transpose.
        prevFrameData = data[frameIndex - 1, :, :, :].swapaxes(0, 1)

        dataW = frameData[:, :, header.index('W')]
        dataSigma = frameData[:, :, header.index('sigma')]

        predictionDiffW = np.absolute(dataW - predictionImageW)
        predictionDiffSigma = np.absolute(dataSigma - predictionImageSigma)

        dataDiffW = np.absolute(dataW - prevFrameData[:, :, header.index('W')])
        dataDiffSigma = np.absolute(dataSigma - prevFrameData[:, :, header.index('sigma')])

        fig.suptitle("Frame {}. Score: {:.3f}, {:.3f}".format(frameMap[frameIndex], testScoreW, testScoreSigma))

        axes[0].imshow(dataW, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        axes[4].imshow(dataSigma, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')

        # Plot the prediction.
        axes[1].imshow(predictionImageW, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')
        axes[5].imshow(predictionImageSigma, origin='lower', cmap='gray', vmin=vmin, vmax=vmax, interpolation='nearest')

        # Plot the difference with prediction.
        axes[2].imshow(predictionDiffW, origin='lower', cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')
        axes[6].imshow(predictionDiffSigma, origin='lower', cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')

        # Plot the difference with prev frame.
        axes[3].imshow(dataDiffW, origin='lower', cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')
        axes[7].imshow(dataDiffSigma, origin='lower', cmap='gray', vmin=0.0, vmax=1.0, interpolation='nearest')

        print("Processed a frame in {} s.".format(time.time() - timeStart))

        pdf.savefig(fig, bbox_inches='tight')
        for a in axes:
            a.clear()

    pdf.close()


def append_camera_image(dataset):
    """
    Appends a column containing grayscale data from the camera.
    The data is cropped from the frame according to the mapping between the data and the image.

    :param dataset:
    :return:
    """

    h5Data, header, frameMap, *r = dataset.unpack_vars()
    imageShift = dataset.get_image_shift()

    min, max, step = compute_data_image_mapping(dataset)

    columnIndex = dataset.append_column('camera')
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

            dataset.h5Data.attrs['cameraImageSize'] = cameraImage.shape

    return dataset


def append_matched_pixels(dataset):
    h5Data, header, frameMap, *r = dataset.unpack_vars()
    imageShift = dataset.get_image_shift()
    frameSize = dataset.get_frame_size()
    min, max, step = compute_data_image_mapping(dataset)

    matchedColumnIndex = dataset.append_column('matched')
    uBackColumnIndex = dataset.append_column('u_back')
    vBackColumnIndex = dataset.append_column('v_back')

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
    min, max, step = compute_data_image_mapping(dataset)

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


def compute_data_image_mapping(dataset):
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

    return np.array([minX, minY]), np.array([maxX, maxY]), np.array([stepX, stepY])


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


def augment_data_extra_feature_number():
    # When allocating a continuous hdf5 file we need to know the final size
    # of the data in advance.

    return 4


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

    imageShift = compute_avg_flow(dataset)

    # Add two columns to the metadata.
    dataset.h5Metadata.resize((dataset.h5Metadata.shape[0], dataset.h5Metadata.shape[1] + 2))
    dataset.h5Metaheader.resize((dataset.h5Metaheader.shape[0] + 2,))

    dataset.h5Metadata[:, [-2, -1]] = imageShift
    dataset.h5Metaheader[[-2, -1]] = [b'imageShiftX', b'imageShiftY']

    print("Adding the camera image...")
    append_camera_image(dataset)
    print("Adding the matched pixels...")
    append_matched_pixels(dataset)

    print("Zeroing the pixels that lost tracking.")
    zero_pixels_without_tracking(dataset)

    return dataset


def plot_data(dataset):
    h5Data, header, frameMap, *r = dataset.unpack_vars()

    # Prepare for plotting
    pdf = PdfPages('../../out/fiber-crack.pdf')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Draw the color wheel
    colorWheel = np.empty((h5Data.shape[1], h5Data.shape[2], 2))
    center = np.array(list(colorWheel.shape[0:2]), dtype=np.int) / 2
    for x in range(0, colorWheel.shape[0]):
        for y in range(0, colorWheel.shape[1]):
            colorWheel[x, y, :] = (-center + [x, y]) / center

    ax.imshow(color_map_hsv(colorWheel[..., 0], colorWheel[..., 1], 2.0).swapaxes(0, 1))
    fig.suptitle("Color Wheel")
    pdf.savefig(fig)
    plt.cla()

    # plot_optic_flow(pdf, dataset, 130)
    # pdf.close()
    #
    # return

    mappingMin, mappingMax, mappingStep = compute_data_image_mapping(dataset)

    fig = plt.figure()
    axes = []
    for f in range(0, 20):
        axes.append(fig.add_subplot(4, 5, f + 1))
        axes[f].axis('off')

    fig.subplots_adjust(hspace=0.025, wspace=0.025)

    frameNumber = h5Data.shape[0]
    crackAreaData = np.zeros((frameNumber, 2))

    # Draw the frame plots.
    for f in range(0, frameNumber):
        timeStart = time.time()
        frameIndex = frameMap[f]
        print("Frame {}".format(frameIndex))
        fig.suptitle("Frame {}".format(frameIndex))

        frameData = h5Data[f, :, :, :]
        frameWidth = frameData.shape[0]
        frameHeight = frameData.shape[1]

        matchedPixels = h5Data[f, :, :, header.index('matched')]
        cameraImageData = h5Data[f, :, :, header.index('camera')]

        assert isinstance(matchedPixels, np.ndarray)
        assert isinstance(cameraImageData, np.ndarray)

        # print("W-plot")
        if 'W' in header:
            imageData0 = frameData[:, :, header.index('W')]
            axes[0].imshow(imageData0.transpose(), origin='lower', cmap='gray')

        # print("UV-plot")
        # axes[1].imshow(color_map_hsv(frameData[..., header.index('U')],
        #                              frameData[..., header.index('V')], maxNorm=2.0)
        #                .swapaxes(0, 1), origin='lower', cmap='gray')
        axes[1].imshow(color_map_hsv(frameData[..., header.index('u')],
                                     frameData[..., header.index('v')], maxNorm=50.0)
                       .swapaxes(0, 1), origin='lower')

        # print("Sigma-plot")
        axes[2].imshow(frameData[:, :, header.index('sigma')].transpose(), origin='lower', cmap='gray')

        # def strain_to_norm(row):
        #     strain = row[[header.index('exx'), header.index('exy'), header.index('eyy')]]
        #     return np.linalg.norm(strain)

        # print("Strain-plot")
        # imageData3 = np.apply_along_axis(strain_to_norm, 2, frameData)
        # axes[4].imshow(imageData3, origin='lower', cmap='gray')

        # print("Camera image")
        cameraImage = frameData[:, :, header.index('camera')]
        axes[3].imshow(cameraImage.transpose(), origin='lower', cmap='gray')

        axes[5].imshow(matchedPixels.transpose(), origin='lower', cmap='gray')

        # print("Matched pixels, mean convolution.")
        matchedPixelsGauss = skimage.filters.gaussian(matchedPixels, 5.0 / 3.0)
        axes[6].imshow(matchedPixelsGauss.transpose(), origin='lower', cmap='gray')

        # print("Matched pixels, closing.")
        # matchedPixelsClosing = scipy.ndimage.morphology.grey_closing(matchedPixels, (3, 3))
        # axes[6].imshow(matchedPixelsClosing.transpose(), origin='lower', cmap='gray')

        # print("Matched pixels, downsample and threshold")
        thresholdBinary = lambda t: lambda x: 1.0 if x >= t else 0.0
        matchedPixelsGaussThres = np.vectorize(thresholdBinary(0.5))(matchedPixelsGauss)

        axes[7].imshow(matchedPixelsGaussThres.transpose(), origin='lower', cmap='gray')

        # print("Matched pixels, close, downsample, threshold, erode")
        # matchedPixelsDown = skimage.measure.block_reduce(matchedPixelsClosing, (3, 3), np.mean)
        # matchedPixelsDown = np.vectorize(threshold(0.5))(matchedPixelsDown)
        # matchedPixelsUp = skimage.transform.resize(matchedPixelsDown, (frameWidth, frameHeight), order=0)

        # matchedPixelsMorph = scipy.ndimage.morphology.grey_erosion(matchedPixelsUp, (3, 3))
        # axes[8].imshow(matchedPixelsMorph.transpose(), origin='lower', cmap='gray')

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

            selem = scipy.ndimage.morphology.generate_binary_structure(tempResult.ndim, 2)
            tempResult = skimage.morphology.binary_erosion(tempResult, selem)
            tempResult = skimage.morphology.binary_erosion(tempResult, selem)
            tempResult = skimage.morphology.remove_small_objects(tempResult, min_size=maxObjectSize)
            tempResult = skimage.morphology.binary_dilation(tempResult, selem)
            tempResult = skimage.morphology.binary_dilation(tempResult, selem)
            tempResult = skimage.morphology.binary_dilation(tempResult, selem)
            tempResult = skimage.morphology.binary_dilation(tempResult, selem)
            tempResult = skimage.morphology.remove_small_holes(tempResult, min_size=holePixelNumber / 6.0)

            # Don't erode back: instead, compensate for the kernel used during DIC.
            dicKernelRadius = int((dicKernelSize - 1) / 2 / mappingStep[0])
            currentDilation = 2  # Because we dilated twice without eroding back.
            for i in range(currentDilation, dicKernelRadius + 1):
                tempResult = skimage.morphology.binary_dilation(tempResult, selem)

            print("Applied {} extra dilation rounds to compensate for the DIC kernel."
                  .format(dicKernelRadius - currentDilation))

            matchedPixelsGaussThresClean = tempResult

        axes[8].imshow(matchedPixelsGaussThresClean.transpose().astype(np.float), origin='lower', cmap='gray')
        axes[9].imshow(cameraImageData.transpose(), origin='lower', cmap='gray')
        contours = skimage.measure.find_contours(matchedPixelsGaussThresClean.transpose(), 0.5)
        for n, contour in enumerate(contours):
            axes[9].plot(contour[:, 1], contour[:, 0], linewidth=1, color='white')

        # uBack = frameData[..., header.index('u_back')]
        # vBack = frameData[..., header.index('v_back')]
        # axes[9].imshow(color_map_hsv(uBack, vBack, maxNorm=50.0).swapaxes(0, 1), origin='lower')
        #
        # sourceMask = matchedPixels > 0.0
        # targetMask = matchedPixels == 0.0
        #
        # uBackFilledIn = masked_gaussian_filter(uBack, sourceMask, targetMask, 50.0 / 3.0)
        # vBackFilledIn = masked_gaussian_filter(vBack, sourceMask, targetMask, 50.0 / 3.0)
        #
        # axes[10].imshow(color_map_hsv(uBackFilledIn, vBackFilledIn, maxNorm=50.0).swapaxes(0, 1), origin='lower')
        # # axes[11].imshow((uBackFilledIn ** 2 + vBackFilledIn ** 2).swapaxes(0, 1), origin='lower')
        #
        # frameSize = frameData.shape[0:2]
        # matchedPixelsRef = np.zeros(frameSize)
        # for x in range(0, frameSize[0]):
        #     for y in range(0, frameSize[1]):
        #         if matchedPixelsGaussThresClean[x, y] == 1.0:
        #             continue
        #
        #         newX = int(round(x + uBackFilledIn[x, y]))
        #         newY = int(round(y + vBackFilledIn[x, y]))
        #         if newX >= 0 and newX < frameSize[0] and newY >= 0 and newY < frameSize[1]:
        #             matchedPixelsRef[newX, newY] = 1.0
        #
        # axes[11].imshow(matchedPixelsRef.transpose(), origin='lower', cmap='gray')

        ###  Variance-based camera image crack extraction.
        varFilterRadius = int(dicKernelRadius / 2)
        cameraImageVar = image_variance_filter(cameraImage, varFilterRadius)

        # print("Variance: from {} to {}".format(np.min(cameraImageVar), np.max(cameraImageVar)))
        axes[10].imshow(cameraImageVar.transpose(), origin='lower', cmap='gray')

        varianceBinary = cameraImageVar < 0.003
        # varianceObjectSize = int(frameWidth * frameHeight / 4000)

        varianceFiltered = varianceBinary.copy()
        for i in range(0, math.ceil(dicKernelRadius / 2)):
            varianceFiltered = skimage.morphology.binary_dilation(varianceFiltered, selem)
        # with warnings.catch_warnings():
        #     warnings.filterwarnings('ignore')
        #     varianceFiltered = skimage.morphology.remove_small_objects(varianceBinary, varianceObjectSize)

        # axes[11].imshow(varianceFiltered.transpose(), origin='lower', cmap='gray')

        totalArea = np.count_nonzero(varianceFiltered)
        crackAreaData[f, 0] = totalArea

        axes[11].imshow(cameraImageData.transpose(), origin='lower', cmap='gray')
        varianceContours = skimage.measure.find_contours(varianceFiltered.transpose(), 0.5)
        for n, contour in enumerate(varianceContours):
            axes[11].plot(contour[:, 1], contour[:, 0], linewidth=1, color='white')

        ### Entropy-based camera image crack extraction.
        entropyFilterRadius = int(dicKernelRadius / 2)
        cameraImageEntropy = image_entropy_filter(cameraImage, entropyFilterRadius)

        axes[12].imshow(cameraImageEntropy.transpose(), origin='lower', cmap='gray')

        print("Entropy: from {} to {}".format(np.min(cameraImageEntropy), np.max(cameraImageEntropy)))

        entropyBinary = cameraImageEntropy < 1.0

        entropyFiltered = entropyBinary.copy()
        for i in range(0, math.ceil(dicKernelRadius / 2)):
            entropyFiltered = skimage.morphology.binary_dilation(entropyFiltered, selem)

        axes[13].imshow(entropyFiltered.transpose(), origin='lower', cmap='gray')

        totalArea = np.count_nonzero(entropyFiltered)
        crackAreaData[f, 1] = totalArea

        axes[14].imshow(cameraImageData.transpose(), origin='lower', cmap='gray')
        entropyContours = skimage.measure.find_contours(entropyFiltered.transpose(), 0.5)
        for n, contour in enumerate(entropyContours):
            axes[14].plot(contour[:, 1], contour[:, 0], linewidth=1, color='white')


        ### todo A quick hack for datasets where not all images are present:
        if np.count_nonzero(cameraImage) <= 0:
            crackAreaData[f, :] = 0

        ### Plot variance and entropy histograms for the image.
        varianceHist = np.histogram(cameraImageVar, bins=16, range=(0, np.max(cameraImageVar)))[0]
        entropyHist = np.histogram(cameraImageEntropy, bins=16, range=(0, np.max(cameraImageEntropy)))[0]

        axes[18].bar(np.linspace(0, np.max(cameraImageVar), 16), varianceHist)
        axes[19].bar(np.linspace(0, np.max(cameraImageEntropy), 16), entropyHist)

        axes[18].axis('on')
        axes[19].axis('on')
        axes[18].get_yaxis().set_visible(False)
        axes[19].get_yaxis().set_visible(False)

        ### Extract 1-pixel crack paths from the sigma plot through skeletonization (thinning).

        # Get the original sigma plot, with untrackable pixels as 'ones' (cracks).
        binarySigma = frameData[..., header.index('sigma')] < 0

        frameSize = frameData.shape[0:2]
        binarySigmaFiltered = binarySigma.copy()
        maxObjectSize = int(frameWidth * frameHeight / 1000)

        dicRadius = int((dicKernelSize - 1) / 2 / mappingStep[0])

        # Erode the cracks to remove smaller noise.
        selem = scipy.ndimage.morphology.generate_binary_structure(binarySigma.ndim, 2)
        for i in range(0, math.ceil(dicRadius / 2)):
            binarySigmaFiltered = skimage.morphology.binary_erosion(binarySigmaFiltered, selem)

        # Remove tiny disconnected chunks of cracks as unnecessary noise.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            binarySigmaFiltered = skimage.morphology.remove_small_objects(binarySigmaFiltered, maxObjectSize)

        axes[15].imshow(binarySigmaFiltered.transpose(), origin='lower', cmap='gray')

        # Compute the skeleton.
        binarySigmaSkeleton = skimage.morphology.skeletonize(binarySigmaFiltered)

        axes[16].imshow(binarySigmaSkeleton.transpose(), origin='lower', cmap='gray')

        # Prune the skeleton from small branches.
        binarySigmaSkeletonPruned = morphology_prune(binarySigmaSkeleton, int(frameSize[0] / 100))

        axes[17].imshow(binarySigmaSkeletonPruned.transpose(), origin='lower', cmap='gray')

        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        for a in axes:
            a.clear()

            a.axis('off')

        print("Rendered in {:.2f} s.".format(time.time() - timeStart))

    # Crack area figure.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    fig.suptitle("Crack area in the current frame")
    ax.plot(crackAreaData[:, 0], label='Variance estimation')
    ax.plot(crackAreaData[:, 1], label='Entropy  estimation')
    # ax.plot(np.sqrt(crackAreaData))
    ax.grid(True)
    plt.ylabel('Pixels')
    plt.legend()

    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    # Print the data-to-camera mapping.
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    mappingText = 'Data to camera image mapping \n'
    mappingText += 'Data size: {} Image size: {} \n'.format(h5Data.shape[1:3], dataset.h5Data.attrs['cameraImageSize'])
    mappingText += 'Min: {} Max: {} Step: {} \n'.format(mappingMin, mappingMax, mappingStep)

    ax.text(0.1, 0.1, mappingText)

    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    pdf.close()


def main():

    timeStart = time.time()
    print("Loading the data.")
    dataset = load_data()
    print("Data loaded in {:.3f} s. Shape: {} Columns: {}".format(time.time() - timeStart, dataset.h5Data.shape, dataset.get_header()))

    min, max, step = compute_data_image_mapping(dataset)
    print("Data to image mapping step: {}".format(step))

    timeStart = time.time()
    print("Augmenting the data.")
    augment_data(dataset)
    print("Data augmented in {:.3f} s.".format(time.time() - timeStart))

    timeStart = time.time()
    print("Plotting the data.")
    plot_data(dataset)
    print("Data plotted in {:.3f} s.".format(time.time() - timeStart))

    # timeStart = time.time()
    # print("Making a prediction.")
    # predict(dataset)
    # print("Prediction finished in {:.3f} s.".format(time.time() - timeStart))

    # https://github.com/tensorflow/tensorflow/issues/3388
    # keras.backend.clear_session()


main()
