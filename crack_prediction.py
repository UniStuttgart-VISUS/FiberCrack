import numpy as np
import matplotlib.pyplot as plt
from keras.layers import Dense
from keras.models import Sequential
from sklearn.model_selection import train_test_split

import FiberCrack.Dataset as Dataset
from PythonExtras.Normalizer import Normalizer
from PythonExtras.numpy_extras import NumpyDynArray


__all__ = ['append_crack_prediction']


def append_crack_prediction(dataset: 'Dataset'):

    frameNumber = dataset.get_frame_number()
    frameSize = dataset.get_frame_size()
    header = dataset.get_header()
    featureNames = ['cameraImageEntropy', 'cameraImageVar', 'camera']
    featureIndices = [header.index(name) for name in featureNames]
    featureNumber = len(featureNames)

    # patchSize = (1, 1) #todo

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

    # Don't use all the data for now. #todo
    # dataX = dataX[::10, ...]
    # dataY = dataY[::10, ...]

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

    # Don't use all the frames for now. #todo
    for f in range(0, frameNumber):
        print("Predicting crack for frame {}".format(f))

        features = dataset.h5Data[f, ...][..., featureIndices]
        featuresFlat = features.reshape((-1, featureNumber))

        predictionFlat = model.predict(normalizerX.scale(featuresFlat))
        prediction = predictionFlat.reshape((frameSize + (1,)))

        dataset.h5Data[f, ..., outputIndex1] = prediction[..., 0]
        dataset.h5Data[f, ..., outputIndex2] = prediction[..., 0] > 0.5


    print(history.history['loss'])
    print(history.history['val_loss'])

    # plt.figure()
    # plt.plot(history.history['loss'], label='Train')
    # plt.plot(history.history['val_loss'], label='Test')

    # plt.show()