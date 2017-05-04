import numpy as np

np.random.seed(13)  # Fix the seed for reproducibility.
# import tensorflow as tf
# tf.set_random_seed(13)

import time
import warnings

from Normalizer import Normalizer
from data_loading import readDataFromCsv
import os, math
import re
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import colorsys
import pickle
import keras
import skimage.measure, skimage.transform, skimage.util, skimage.filters, skimage.feature, skimage.morphology
import scipy.ndimage.morphology
from keras.layers import Dense
from keras.constraints import maxnorm
from PIL import Image
from os import path
import h5py

from numpy_extras import slice_nd

############### Configuration ###############

preloadedDataDir = 'C:\\preloaded_data'


# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Steel-Epoxy'
# metadataFilename = 'Steel-Epoxy.csv'
# dataDir = 'data_export_tstep3'
# imageDir = 'raw_images'
# imageBaseName = 'Spec054'

# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Steel-ModifiedEpoxy'
# metadataFilename = 'Steel-ModifiedEpoxy.csv'
# dataDir = 'data_export'
# imageDir = 'raw_images'
# imageBaseName = 'Spec010'

basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/PTFE-Epoxy'
metadataFilename = 'PTFE-Epoxy.csv'
dataDir = 'data_export'
# dataDir = 'data_export_fine'
imageDir = 'raw_images'
imageBaseName = 'Spec048'

maxFrames = 99999
reloadOriginalData = False


#############################################

# todo Sigma-filtering when searching for unmatched pixels
# todo Strip NaNs from the metadata/metaheader
# todo PTFE-Epoxy frame 4465 data is incomplete.


class Dataset:
    h5Data = None
    h5Header = None
    h5FrameMap = None
    h5Metadata = None
    h5Metaheader = None

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


def load_data():
    h5Filename = metadataFilename.replace('.csv', '.hdf5')
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
            rgb = colorsys.hsv_to_rgb(hue, 1.0, 1.0)
            a = norm / maxNorm

            result[i, j, :] = [rgb[0] * a, rgb[1] * a, rgb[2] * a]

    return result


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
        cameraImagePath = path.join(basePath, imageDir, '{}-{:04d}_0.tif'.format(imageBaseName, frameIndex))
        cameraImageAvailable = os.path.isfile(cameraImagePath)
        if cameraImageAvailable:
            # Crop a rectangle from the camera image, accounting for the overall shift of the specimen.
            relMin = min + imageShift[f, ...]
            relMax = max + imageShift[f, ...]

            size = ((relMax - relMin) / step).astype(np.int)
            cameraImage = skimage.util.img_as_float(Image.open(cameraImagePath))
            h5Data[f, 0:size[0], 0:size[1], columnIndex] = \
                np.array(cameraImage)[relMin[1]:relMax[1]:step[1], relMin[0]:relMax[0]:step[0]].transpose()

    return dataset


def append_matched_pixels(dataset):
    h5Data, header, frameMap, *r = dataset.unpack_vars()
    imageShift = dataset.get_image_shift()
    frameSize = dataset.get_frame_size()
    min, max, step = compute_data_image_mapping(dataset)

    columnIndex = dataset.append_column('matched')
    frameNumber = h5Data.shape[0]
    for f in range(0, frameNumber):
        print("Frame {}/{}".format(f, frameNumber))
        frameFlow = h5Data[f, :, :, :][:, :, [header.index('u'), header.index('v'), header.index('sigma')]]
        for x in range(0, frameSize[0]):
            for y in range(0, frameSize[1]):
                newX = x + int(round((frameFlow[x, y, 0] - imageShift[f, 0]) / step[0]))
                newY = y + int(round((frameFlow[x, y, 1] - imageShift[f, 1]) / step[1]))
                if (0 < newX < frameSize[0]) and (0 < newY < frameSize[1]):
                    h5Data[f, newX, newY, columnIndex] = 1.0


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

    return 2


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

    return dataset


def plot_data(dataset):
    h5Data, header, frameMap, *r = dataset.unpack_vars()

    # Prepare for plotting
    pdf = PdfPages('../../out/fiber-crack.pdf')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Draw the color wheel
    colorWheel = np.empty((h5Data.shape[1], h5Data.shape[2], 3))
    for x in range(0, colorWheel.shape[0]):
        for y in range(0, colorWheel.shape[1]):
            angle = math.atan2(y - colorWheel.shape[1] / 2, x - colorWheel.shape[0] / 2)
            angle = angle if angle >= 0.0 else angle + 2 * math.pi  # Convert [-pi, pi] -> [0, 2*pi]
            hue = angle / (2 * math.pi)  # Map [0, 2*pi] -> [0, 1]
            colorWheel[x, y, :] = list(colorsys.hsv_to_rgb(hue, 1.0, 1.0))
    ax.imshow(colorWheel.swapaxes(0, 1))
    fig.suptitle("Color Wheel")
    pdf.savefig(fig)
    plt.cla()

    # plot_optic_flow(pdf, dataset, 130)
    # pdf.close()
    #
    # return

    fig = plt.figure()
    axes = []
    for f in range(0, 9):
        axes.append(fig.add_subplot(3, 3, f + 1))
        axes[f].axis('off')

    fig.subplots_adjust(hspace=0.025, wspace=0.025)

    # Draw the frame plots.
    for f in range(0, h5Data.shape[0]):
        frameIndex = frameMap[f]
        print("Frame {}".format(frameIndex))
        fig.suptitle("Frame {}".format(frameIndex))

        frameData = h5Data[f, :, :, :]
        frameWidth = frameData.shape[0]
        frameHeight = frameData.shape[1]

        matchedPixels = h5Data[f, :, :, header.index('matched')]
        cameraImageData = h5Data[f, :, :, header.index('camera')]

        # print("W-plot")
        imageData0 = frameData[:, :, header.index('W')]
        axes[0].imshow(imageData0.transpose(), origin='lower', cmap='gray')

        # print("UV-plot")
        axes[1].imshow(color_map_hsv(frameData[..., header.index('U')],
                                     frameData[..., header.index('V')], maxNorm=2.0)
                       .swapaxes(0, 1), origin='lower', cmap='gray')

        # print("Sigma-plot")
        axes[2].imshow(frameData[:, :, header.index('sigma')].transpose(), origin='lower', cmap='gray')

        # def strain_to_norm(row):
        #     strain = row[[header.index('exx'), header.index('exy'), header.index('eyy')]]
        #     return np.linalg.norm(strain)

        # print("Strain-plot")
        # imageData3 = np.apply_along_axis(strain_to_norm, 2, frameData)
        # axes[4].imshow(imageData3, origin='lower', cmap='gray')

        # print("Camera image")
        axes[3].imshow(frameData[:, :, header.index('camera')].transpose(), origin='lower', cmap='gray')

        axes[4].imshow(matchedPixels.transpose(), origin='lower', cmap='gray')

        # print("Matched pixels, mean convolution.")
        matchedPixelsGauss = skimage.filters.gaussian(matchedPixels, 5.0 / 3.0)
        axes[5].imshow(matchedPixelsGauss.transpose(), origin='lower', cmap='gray')

        # print("Matched pixels, closing.")
        # matchedPixelsClosing = scipy.ndimage.morphology.grey_closing(matchedPixels, (3, 3))
        # axes[6].imshow(matchedPixelsClosing.transpose(), origin='lower', cmap='gray')

        # print("Matched pixels, downsample and threshold")
        threshold = lambda t: lambda x: x if x >= t else 0.0
        thresholdBinary = lambda t: lambda x: 1.0 if x >= t else 0.0
        matchedPixelsGaussThres = np.vectorize(thresholdBinary(0.5))(matchedPixelsGauss)

        axes[6].imshow(matchedPixelsGaussThres.transpose(), origin='lower', cmap='gray')

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
            # Don't erode back: an extra round of dilation compensates for the kernel used during DIC.
            # tempResult = skimage.morphology.binary_erosion(tempResult, selem)
            # tempResult = skimage.morphology.binary_erosion(tempResult, selem)

            matchedPixelsGaussThresClean = tempResult

        axes[7].imshow(matchedPixelsGaussThresClean.transpose().astype(np.float), origin='lower', cmap='gray')
        axes[8].imshow(cameraImageData.transpose(), origin='lower', cmap='gray')
        contours = skimage.measure.find_contours(matchedPixelsGaussThresClean.transpose(), 0.8)
        for n, contour in enumerate(contours):
            axes[8].plot(contour[:, 1], contour[:, 0], linewidth=1, color='white')


        # axes[11].imshow(cameraImageData.transpose(), origin='lower', cmap='gray')
        # contours = skimage.measure.find_contours(matchedPixelsMorph.transpose(), 0.8)
        # for n, contour in enumerate(contours):
        #     axes[11].plot(contour[:, 1], contour[:, 0], linewidth=1, color='white')

        pdf.savefig(fig, bbox_inches='tight')
        for a in axes:
            a.clear()
            a.axis('off')
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
    # print("Prediction finihsed in {:.3f} s.".format(time.time() - timeStart))

    # https://github.com/tensorflow/tensorflow/issues/3388
    # keras.backend.clear_session()


main()
