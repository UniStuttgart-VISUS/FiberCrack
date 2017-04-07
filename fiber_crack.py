import numpy as np

np.random.seed(13)  # Fix the seed for reproducibility.
# import tensorflow as tf
# tf.set_random_seed(13)

import time
import warnings

from FiberCrack.Normalizer import Normalizer
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

############### Configuration ###############


basePath = 'W:\Experiments\Steel-Epoxy'
metadataFilename = 'Steel-Epoxy.csv'
dataDir = 'data_export_tstep3'
imageDir = 'raw_images'
imageBaseName = 'Spec054'

# basePath = 'W:\Experiments\Steel-ModifiedEpoxy'
# metadataFilename = 'Steel-ModifiedEpoxy.csv'
# dataDir = 'data_export'
# imageDir = 'raw_images'
# imageBaseName = 'Spec010'

# basePath = 'W:\Experiments\PTFE-Epoxy'
# metadataFilename = 'PTFE-Epoxy.csv'
# dataDir = 'data_export'
# imageDir = 'raw_images'
# imageBaseName = 'Spec048'

reloadOriginalData = False


#############################################

# todo Sigma-filtering when searching for unmatched pixels
# todo Strip NaNs from the metadata/metaheader
# todo PTFE-Epoxy frame 4465 data is incomplete.


class Dataset:
    data = None
    header = None
    frameMap = None
    metadata = None
    metaheader = None

    def __init__(self, data, header, frameMap, metadata, metaheader):
        self.data = data
        self.header = header
        self.frameMap = frameMap
        self.metadata = metadata
        self.metaheader = metaheader

    def unpack_vars(self):
        return self.data, self.header, self.frameMap, self.metadata, self.metaheader

    def append_to_data(self, newColumn, newColumnName):
        # assert(newColumn.ndim == 3)
        assert (newColumn.shape == self.data.shape[0:-1])

        self.data = np.concatenate((self.data, newColumn[..., np.newaxis]), axis=-1)
        self.header.append(newColumnName)

    def get_image_shift(self):
        indices = [self.metaheader.index('imageShiftX'), self.metaheader.index('imageShiftY')]
        return self.metadata[:, indices].astype(np.int)

    def get_frame_size(self):
        return tuple(self.data.shape[1:3])


def load_data():
    # Load the metadata, describing each frame of the experiment.
    metadata, metaheader = readDataFromCsv(path.join(basePath, metadataFilename))
    frameMap = []
    data = None
    header = None
    preloadedDataFilename = metadataFilename.replace('.csv', '.npy')
    preloadedMetadataFilename = metadataFilename.replace('.csv', '.pickle')
    if reloadOriginalData or not path.isfile(path.join(basePath, preloadedDataFilename)):
        # Load data for each available frame.
        for filename in os.listdir(path.join(basePath, dataDir)):
            filepath = path.join(basePath, dataDir, filename)
            frameData, frameHeader = readDataFromCsv(filepath)

            # Extract the frame index from the filename (not all frames are there)
            regexMatch = re.search('[^\-]+-(\d+).*', filename)
            frameIndex = int(regexMatch.group(1))

            # Figure out the size of the frame.
            # Count its width by finding where the value of 'y' changes the first time.
            frameWidth = 0
            frameMap.append(frameIndex)
            yIndex = frameHeader.index('y')
            for row in frameData:
                if int(row[yIndex]) != int(frameData[0, yIndex]):
                    break
                frameWidth += 1

            frameSize = (frameWidth, int(frameData.shape[0] / frameWidth))
            frameData = frameData.reshape((frameSize[1], frameSize[0], -1)).swapaxes(0, 1)

            # Stack the data along the first axis
            appendedData = frameData[np.newaxis, :, :]

            if data is None:
                data = appendedData
                header = frameHeader
            else:
                if frameData.shape == tuple(data.shape[1:]):
                    data = np.concatenate((data, appendedData), 0)
                else:
                    frameMap.pop()
                    warnings.warn("Frame data has wrong shape: {}, expected: {}".format(
                        frameData.shape, tuple(data.shape[1:])))
            print("Read {}".format(filename))

        np.save(path.join(basePath, preloadedDataFilename), data)
        with open(path.join(basePath, preloadedMetadataFilename), 'wb') as metadataFile:
            pickle.dump((header, frameMap), metadataFile)
    else:
        print("Reading preloaded data from {}".format(preloadedDataFilename))
        data = np.load(path.join(basePath, preloadedDataFilename))
        # firstFilename = os.listdir(path.join(basePath, dataDir))[0]
        # header = readHeaderFromCsv(path.join(basePath, dataDir, firstFilename))
        with open(path.join(basePath, preloadedMetadataFilename), 'rb') as metadataFile:
            header, frameMap = pickle.load(metadataFile)

    # Metadata describes all the frames, but we only have CSVs for some of them,
    # so discard the redundant metadata.
    metadata = metadata[frameMap, ...]

    return Dataset(data, header, frameMap, metadata, metaheader)


def rgba_to_rgb(rgba):
    a = rgba[3]
    return np.array([rgba[0] * a, rgba[1] * a, rgba[2] * a])


def color_map_hsv(xCol, yCol, maxNorm, sigmaIndex=-1):
    def result(row):
        # Paint black the pixels that lost tracking
        if sigmaIndex != -1 and row[sigmaIndex] < 0:
            return [0.0, 0.0, 0.0]

        dir = row[[xCol, yCol]]
        norm = np.linalg.norm(dir)
        if norm > maxNorm:
            raise ValueError("Encountered norm of {} while {} was specified as max!".format(norm, maxNorm))

        angle = math.atan2(dir[1], dir[0])
        angle = angle if angle >= 0.0 else angle + 2 * math.pi  # Convert [-pi, pi] -> [0, 2*pi]
        hue = angle / (2 * math.pi)  # Map [0, 2*pi] -> [0, 1]
        rgb = list(colorsys.hsv_to_rgb(hue, 1.0, 1.0)) + [norm / maxNorm]
        return rgba_to_rgb(rgb)

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

    data, header, frameMap, *r = dataset.unpack_vars()
    imageShift = dataset.get_image_shift()

    min, max, step = compute_data_image_mapping(dataset)

    cameraImageData = np.zeros(tuple(data.shape[0:3]))
    for f in range(0, data.shape[0]):
        frameIndex = frameMap[f]
        cameraImagePath = path.join(basePath, imageDir, '{}-{:04d}_0.tif'.format(imageBaseName, frameIndex))
        cameraImageAvailable = os.path.isfile(cameraImagePath)
        if cameraImageAvailable:
            # Crop a rectangle from the camera image, accounting for the overall shift of the specimen.
            relMin = min + imageShift[f, ...]
            relMax = max + imageShift[f, ...]

            size = ((relMax - relMin) / step).astype(np.int)
            cameraImage = skimage.util.img_as_float(Image.open(cameraImagePath))
            cameraImageData[f, 0:size[0], 0:size[1]] = \
                np.array(cameraImage)[relMin[1]:relMax[1]:step[1], relMin[0]:relMax[0]:step[0]].transpose()

    dataset.append_to_data(cameraImageData, 'camera')

    return dataset


def append_matched_pixels(dataset):
    data, header, frameMap, *r = dataset.unpack_vars()
    imageShift = dataset.get_image_shift()
    frameSize = dataset.get_frame_size()
    min, max, step = compute_data_image_mapping(dataset)

    matchedPixels = np.zeros(tuple(data.shape[0:3]))
    for f in range(0, data.shape[0]):
        frameFlow = data[f, :, :, :][:, :, [header.index('u'), header.index('v'), header.index('sigma')]]
        for x in range(0, frameSize[0]):
            for y in range(0, frameSize[1]):
                newX = x + int(round((frameFlow[x, y, 0] - imageShift[f, 0]) / step[0]))
                newY = y + int(round((frameFlow[x, y, 1] - imageShift[f, 1]) / step[1]))
                if (0 < newX < frameSize[0]) and (0 < newY < frameSize[1]):
                    matchedPixels[f, newX, newY] = 1.0

    dataset.append_to_data(matchedPixels, 'matched')


def compute_data_image_mapping(dataset):
    """
    Fetch the min/max pixel coordinates of the data, in original image space (2k*2k image)
    Important, since the data only covers every ~fifth pixel of some cropped subimage of the camera image.

    :param dataset:
    :return: (min, max, step)
    """
    data, header, *r = dataset.unpack_vars()

    minX = int(data[0, 0, 0, header.index('x')])
    maxX = int(data[0, -1, 0, header.index('x')])
    minY = int(data[0, 0, 0, header.index('y')])
    maxY = int(data[0, 0, -1, header.index('y')])
    stepX = round((maxX - minX) / data.shape[1])
    stepY = round((maxY - minY) / data.shape[2])

    return np.array([minX, minY]), np.array([maxX, maxY]), np.array([stepX, stepY])


def compute_avg_flow(dataset):
    """
    Sample the flow/shift (u,v) at a few points to determine the average shift of each frame
    relative to the base frame.

    :param dataset:
    :return:
    """

    data, header, *r = dataset.unpack_vars()

    # Select points which will be sampled to determine the overall shift relative to the ref. frame
    # (This is used to align the camera image with the reference frame (i.e. data space)
    sampleX = np.linspace(0 + 20, data.shape[1] - 20, 4)
    sampleY = np.linspace(0 + 20, data.shape[2] - 20, 4)
    samplePointsX, samplePointsY = np.array(np.meshgrid(sampleX, sampleY)).astype(np.int)
    # Convert to an array of 2d points.
    samplePoints = np.concatenate((samplePointsX[:, :, np.newaxis], samplePointsY[:, :, np.newaxis]), 2).reshape(-1, 2)

    avgFlow = np.empty((data.shape[0], 2))
    for f in range(0, data.shape[0]):
        frameData = data[f, ...]
        samples = np.array([frameData[tuple(p)] for p in samplePoints])
        uvSamples = samples[:, [header.index('u'), header.index('v'), header.index('sigma')]]
        uvSamples = uvSamples[uvSamples[:, 2] >= 0, :]  # Filter out points that lost tracking
        avgFlow[f, :] = np.mean(uvSamples[:, 0:2], axis=0).astype(np.int) if uvSamples.shape[0] > 0 else np.array(
            [0, 0])

    return avgFlow


def augment_data(dataset):
    imageShift = compute_avg_flow(dataset)
    dataset.metadata = np.concatenate((dataset.metadata, imageShift[:, 0, None], imageShift[:, 1, None]), axis=1)
    dataset.metaheader.extend(['imageShiftX', 'imageShiftY'])

    append_camera_image(dataset)
    append_matched_pixels(dataset)

    return dataset


def plot_data(dataset):
    data, header, frameMap, *r = dataset.unpack_vars()

    # Prepare for plotting
    pdf = PdfPages('..\out\data.pdf')
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Draw the color wheel
    colorWheel = np.empty((data.shape[1], data.shape[2], 3))
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

    fig = plt.figure()
    axes = []
    for f in range(0, 9):
        axes.append(fig.add_subplot(3, 3, f + 1))
        axes[f].axis('off')

    fig.subplots_adjust(hspace=0.025, wspace=0.025)

    # Draw the frame plots.
    for f in range(0, data.shape[0]):
        frameIndex = frameMap[f]
        print("Frame {}".format(frameIndex))
        fig.suptitle("Frame {}".format(frameIndex))

        frameData = data[f, :, :, :]
        frameWidth = frameData.shape[0]
        frameHeight = frameData.shape[1]

        matchedPixels = data[f, :, :, header.index('matched')]
        cameraImageData = data[f, :, :, header.index('camera')]

        # print("W-plot")
        imageData0 = frameData[:, :, header.index('W')]
        axes[0].imshow(imageData0.transpose(), origin='lower', cmap='gray')

        # print("UV-plot")
        colorMap = color_map_hsv(header.index('U'), header.index('V'), 2.0, -1)
        axes[1].imshow(np.apply_along_axis(colorMap, 2, frameData).swapaxes(0, 1), origin='lower', cmap='gray')

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
    print("Data loaded in {:.3f} s. Shape: {} Columns: {}".format(time.time() - timeStart, dataset.data.shape, dataset.header))

    timeStart = time.time()
    print("Augmenting the data.")
    augment_data(dataset)
    print("Data augmented in {:.3f} s.".format(time.time() - timeStart))

    # timeStart = time.time()
    # print("Plotting the data.")
    # plot_data(dataset)
    # print("Data plotted in {:.3f} s.".format(time.time() - timeStart))

    timeStart = time.time()
    print("Making a prediction.")
    predict(dataset)
    print("Prediction finihsed in {:.3f} s.".format(time.time() - timeStart))

    # https://github.com/tensorflow/tensorflow/issues/3388
    # keras.backend.clear_session()


main()
