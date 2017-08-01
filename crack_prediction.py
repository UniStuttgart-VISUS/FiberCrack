from typing import Callable, Dict, List

import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

import FiberCrack.Dataset as Dataset
from PythonExtras.Normalizer import Normalizer
from PythonExtras.numpy_extras import NumpyDynArray, extract_patches


__all__ = ['append_crack_prediction_spatial', 'append_crack_prediction_simple']


def append_crack_prediction_spatial(dataset: 'Dataset', allTextureKernelSizes,
                                    textureFilters: List[str]):
    """
    Make a crack prediction based on many texture features, using a local neighborhood around a pixel.
    Unmatched with entropy detector is used as ground truth.

    Note: some config parameters are imported just to create a dependency.

    :param dataset:
    :param allTextureKernelSizes:
    :param textureFilters:
    :return:
    """

    frameNumber = dataset.get_frame_number()
    header = dataset.get_header()

    textureFeatureNames = dataset.get_str_array_attr('textureFeatureNames')

    targetFeature = 'cracksFromUnmatchedAndEntropy'  # What we're going to predict.
    featureNames = textureFeatureNames + ['camera', targetFeature]
    featureIndices = [header.index(name) for name in featureNames]

    # Note: relative to already selected features.
    xFeatureIndices = [i for i, name in enumerate(featureNames) if name != targetFeature]

    patchSize = [5, 5]

    print("Collecting training data...")

    # One of the features is Y.
    featureNumberFlat = (len(featureNames) - 1) * patchSize[0] * patchSize[1]

    rawDataX = NumpyDynArray((-1, featureNumberFlat))
    rawDataY = NumpyDynArray((-1, 1))

    for f in range(frameNumber):
        # Fetch all relevant data for this frame.
        features = dataset.h5Data[f, ...][..., featureIndices]
        patches, *r = extract_patches(features, [0, 1], patchSize)

        # Select features that are used as input for the net.
        patchesX = patches[..., xFeatureIndices]
        # Flatten the patch to get rid of the spatial dimensions.
        patchesX = patchesX.reshape((patchesX.shape[0], -1))

        # Select the target feature, take only one pixel at the center of the patch.
        patchesY = patches[..., int(patchSize[0] / 2), int(patchSize[1] / 2), featureNames.index(targetFeature)]

        rawDataX.append_all(patchesX)
        rawDataY.append_all(patchesY[:, np.newaxis])

    dataX = rawDataX.get_all()
    dataY = rawDataY.get_all()

    # Don't use all the data for now. !!!!!!!!!!
    dataX = dataX[::10, ...]
    dataY = dataY[::10, ...]

    print("Training...")

    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=.2)

    normalizerX = Normalizer().fit(trainX, -1)
    trainX = normalizerX.scale(trainX)
    testX = normalizerX.scale(testX)

    model = Sequential()
    model.add(Dense(128, activation='relu', input_dim=trainX.shape[-1]))
    model.add(Dense(64, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    history = model.fit(trainX, trainY, validation_data=(testX, testY), nb_epoch=2, batch_size=128)

    print("Predicting")

    outputIndex1 = dataset.create_or_get_column('crackPredictionSpatial')
    outputIndex2 = dataset.create_or_get_column('crackPredictionSpatialBinary')

    for f in range(0, frameNumber):
        print("Predicting crack for frame {}".format(f))

        # Fetch all relevant data for this frame.
        features = dataset.h5Data[f, ...][..., featureIndices]
        patches, patchCenters, patchNumber = extract_patches(features, [0, 1], patchSize)

        # Select features that are used as input for the net.
        patchesX = patches[..., xFeatureIndices]
        # Flatten the patch.
        patchesX = patchesX.reshape((patchesX.shape[0], -1))

        predictionFlat = model.predict(normalizerX.scale(patchesX))
        prediction = predictionFlat.reshape(tuple(patchNumber))

        min = patchCenters[0]
        max = patchCenters[-1]

        dataset.h5Data[f, min[0]:max[0]+1, min[1]:max[1]+1, outputIndex1] = prediction[...]
        dataset.h5Data[f, min[0]:max[0]+1, min[1]:max[1]+1, outputIndex2] = prediction[...] > 0.5


def append_crack_prediction_simple(dataset: 'Dataset'):

    frameNumber = dataset.get_frame_number()
    frameSize = dataset.get_frame_size()
    header = dataset.get_header()
    featureNames = ['cameraImageEntropy', 'cameraImageVar', 'camera']
    featureIndices = [header.index(name) for name in featureNames]
    featureNumber = len(featureNames)

    # Collect training data.

    rawDataX = NumpyDynArray((-1, featureNumber))
    rawDataY = NumpyDynArray((-1, 1))

    for f in range(frameNumber):
        crackImage = dataset.get_column_at_frame(f, 'cracksFromUnmatchedAndEntropy')
        features = dataset.h5Data[f, ...][..., featureIndices]

        mask = crackImage == 1.0
        crackFeatures = features[mask]
        nonCrackFeatures = features[np.logical_not(mask)]

        rawDataX.append_all(crackFeatures)
        rawDataY.append_all(np.ones((crackFeatures.shape[0], 1)))

        rawDataX.append_all(nonCrackFeatures)
        rawDataY.append_all(np.zeros((nonCrackFeatures.shape[0], 1)))

    dataX = rawDataX.get_all()
    dataY = rawDataY.get_all()

    # Don't use all the data for now. !!!!!!!!!!
    dataX = dataX[::10, ...]
    dataY = dataY[::10, ...]

    # Train a model.

    trainX, testX, trainY, testY = train_test_split(dataX, dataY, test_size=.2)

    normalizerX = Normalizer().fit(trainX, -1)
    trainX = normalizerX.scale(trainX)
    testX = normalizerX.scale(testX)

    model = Sequential()
    model.add(Dense(1, activation='sigmoid', input_dim=trainX.shape[-1]))

    model.compile(optimizer='rmsprop', loss='binary_crossentropy')
    history = model.fit(trainX, trainY, validation_data=(testX, testY), nb_epoch=2)

    # Predict.
    outputIndex1 = dataset.create_or_get_column('crackPrediction')
    outputIndex2 = dataset.create_or_get_column('crackPredictionBinary')

    # Don't use all the frames for now.
    for f in range(0, frameNumber):
        print("Predicting crack for frame {}".format(f))

        features = dataset.h5Data[f, ...][..., featureIndices]
        featuresFlat = features.reshape((-1, featureNumber))

        predictionFlat = model.predict(normalizerX.scale(featuresFlat))
        prediction = predictionFlat.reshape((frameSize + (1,)))

        dataset.h5Data[f, ..., outputIndex1] = prediction[..., 0]
        dataset.h5Data[f, ..., outputIndex2] = prediction[..., 0] > 0.5
