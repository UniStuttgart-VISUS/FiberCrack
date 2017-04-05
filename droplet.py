import os

import numpy as np
from os import path
import time
import math

import keras
from keras.layers import Convolution3D, Dropout, MaxPooling3D, Dense, Flatten
from keras.constraints import maxnorm

from FiberCrack.Normalizer import Normalizer


# todo not tested
def downsample_volume(data, outputSize):
    inputSize = data.shape

    def gauss(x, m, s):
        invSqrt2Pi = 0.3989422804014327
        a = (x - m) / s
        return invSqrt2Pi / s * math.exp(-0.5 * a * a)

    def round_up_to_odd(value):
        rounded = math.ceil(value)
        return rounded if rounded % 2 == 1 else rounded + 1

    kernelSize = (round_up_to_odd(inputSize[0] / float(outputSize[0])),
                  round_up_to_odd(inputSize[1] / float(outputSize[1])),
                  round_up_to_odd(inputSize[2] / float(outputSize[2])))

    kernelRadius = (max((kernelSize[0] - 1) / 2, 1),
                    max((kernelSize[1] - 1) / 2, 1),
                    max((kernelSize[2] - 1) / 2, 1))
    
    sampleMax = 255.0

    output = np.empty(outputSize)
    for z in range(0, outputSize[0]):
        for y in range(0, outputSize[1]):
            for x in range(0, outputSize[2]):
                samplePos = (z / float(outputSize[0]) * inputSize[0],
                             y / float(outputSize[1]) * inputSize[1],
                             x / float(outputSize[2]) * inputSize[2])

                minInput = (math.ceil(samplePos[0] - kernelRadius[0]),
                            math.ceil(samplePos[1] - kernelRadius[1]),
                            math.ceil(samplePos[2] - kernelRadius[2]))
                maxInput = (math.floor(samplePos[0] + kernelRadius[0]),
                            math.floor(samplePos[1] + kernelRadius[1]),
                            math.floor(samplePos[2] + kernelRadius[2]))

                result = 0.0
                weightSum = 0.0
                for zi in range(minInput[0], maxInput[1] + 1):
                    for yi in range(minInput[1], maxInput[1] + 1):
                        for xi in range(minInput[2], maxInput[2] + 1):
                            isOutbound = zi < 0 or yi < 0 or xi < 0 or \
                                         zi > inputSize[0] - 1 or yi > inputSize[1] - 1 or xi > inputSize[2] - 1
                            if isOutbound:
                                continue

                            sample = data[zi, yi, xi]
                            weight = (gauss(samplePos[0] - zi, 0, kernelSize[0] * 3),
                                      gauss(samplePos[1] - yi, 0, kernelSize[1] * 3),
                                      gauss(samplePos[1] - xi, 0, kernelSize[2] * 3))
                            result += (sample / sampleMax) * weight
                            weightSum += weight

                result /= weightSum
                output[z, y, x] = int(result * sampleMax)

    return output

def load_volume(dir, width, height, depth):

    filenames = [f for f in os.listdir(dir)]
    data = np.empty((len(filenames), depth, height, width), dtype=np.uint8)

    preloadedDataPath = path.join(dir, 'data.npy')
    if not path.isfile(preloadedDataPath):
        print("Reading data from {}".format(dir))
        for i, filename in enumerate(filenames):
            filepath = path.join(dir, filename)
            frameData = np.array([])
            with open(filepath, 'rb') as file:
                frameData = np.fromfile(file, np.uint8, width * height * depth)

            frameData = frameData.reshape((depth, height, width))
            data[i, ...] = frameData


        np.save(preloadedDataPath, data)
    else:
        print("Reading preloaded data from {}".format(preloadedDataPath))
        data = np.load(preloadedDataPath)

    return data

def train_net(XTrain, yTrain, XVal, yVal, patchSize):
    timeStart = time.time()
    model = keras.models.Sequential()

    model.add(Dense(128, input_dim=XTrain.shape[-1], activation="relu", W_constraint=maxnorm(3)))
    model.add(Dense(64, activation="relu", W_constraint=maxnorm(3)))
    model.add(Dense(16, activation="relu", W_constraint=maxnorm(3)))
    model.add(Dense(1))

    epochNum = 15
    learningRate = 0.01
    batchSize = 32

    optimizer = keras.optimizers.RMSprop(lr=learningRate)
    model.compile(loss='mse', optimizer=optimizer)
    print("Finished compiling the net in {:.3f} s.".format(time.time() - timeStart))

    history = model.fit(XTrain, yTrain, validation_data=(XVal, yVal), nb_epoch=epochNum, batch_size=batchSize,
                        verbose=0)

    print("Finished training a net in {:.3f} s.)".format(time.time() - timeStart))
    print("History: {}".format(history.history))
    print("Training loss: {}".format(history.history['loss']))
    print("Val      loss: {}".format(history.history['val_loss']))

    return model, history

def predict_frame(data, targetFrame, timeWindow):
    patchSize = (2, 8, 8, 8)
    # Whether should collapse all spatial and temporal dimensions and use a 1D vector representation.
    flattenInput = True

    data = data[targetFrame - timeWindow:targetFrame + 1, :, :, :]

    patchNumber = (
        (data.shape[0] - patchSize[0] + 1 - 1),  # We want to predict the next frame, so we can't use the last frame
        (data.shape[1] - patchSize[1] + 1),
        (data.shape[2] - patchSize[2] + 1),
        (data.shape[3] - patchSize[3] + 1))
    patchNumberFlat = patchNumber[0] * patchNumber[1] * patchNumber[2] * patchNumber[3]
    netDataX = np.empty((patchNumberFlat, patchSize[0], patchSize[1], patchSize[2], patchSize[3]), dtype=np.uint8)
    netDataY = np.empty((patchNumberFlat, 1), dtype=np.uint8)

    for t in range(0, patchNumber[0]):
        for z in range(0, patchNumber[1]):
            for y in range(0, patchNumber[2]):
                for x in range(0, patchNumber[3]):
                    patchIndex = t * patchNumber[1] * patchNumber[2] * patchNumber[3] + \
                                 z * patchNumber[2] * patchNumber[3] + y * patchNumber[3] + x
                    netDataX[patchIndex, :, :, :, :] = data[t:t + patchSize[0], z:z + patchSize[1], y:y + patchSize[2], x:x + patchSize[3]]
                    netDataY[patchIndex, 0] = data[t + patchSize[0],
                                                   z + int(patchSize[1] / 2),
                                                   y + int(patchSize[2] / 2),
                                                   x + int(patchSize[3] / 2)]

    if flattenInput:
        netDataX = netDataX.reshape((patchNumberFlat, -1))
        netDataY = netDataY.reshape((patchNumberFlat, -1))


    # Filter out empty patches.
    nonzeroPatchIndices = [i for i in range(0, patchNumberFlat) if np.count_nonzero(netDataX[i, ...] != 0)]
    startTestIndexSparse = (patchNumber[0] - 1) * patchNumber[1] * patchNumber[2] * patchNumber[3]
    nonzeroTestPatchIndices = [i for i in range(startTestIndexSparse, patchNumberFlat) if np.count_nonzero(netDataX[i, ...] != 0)]
    startTestIndex = len(nonzeroPatchIndices) - len(nonzeroTestPatchIndices)

    print("Nonzero patches: {}/{}".format(len(nonzeroPatchIndices), patchNumberFlat))
    netDataX = netDataX[nonzeroPatchIndices, ...]
    netDataY = netDataY[nonzeroPatchIndices, ...]


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

    predictionRaw = model.predict(XTest)
    testScore = model.evaluate(XTest, yTest, verbose=0)
    print("Test score: {}".format(testScore))

    # We only make prediction for nonempty patches, thus we first need to arrange the patches in space again.
    predictionSparse = np.empty((patchNumber[1] * patchNumber[2] * patchNumber[3]))
    indicesRelativeToTimestep = [i - startTestIndexSparse for i in nonzeroTestPatchIndices]
    predictionSparse[indicesRelativeToTimestep] = predictionRaw
    predictionSparse = predictionSparse.reshape((patchNumber[1], patchNumber[2], patchNumber[3]))

    # Due to patch size, we cannot make a prediction near the volume edges, pad it.
    predictionVolume = np.empty((data.shape[1], data.shape[2], data.shape[3]))
    patchRZ = int(patchSize[1] / 2)
    patchRY = int(patchSize[2] / 2)
    patchRX = int(patchSize[3] / 2)
    predictionVolume[patchRZ:data.shape[1] - patchSize[1] + patchRZ + 1,
                     patchRY:data.shape[2] - patchSize[2] + patchRY + 1,
                     patchRX:data.shape[3] - patchSize[3] + patchRX + 1] = predictionSparse


    return predictionVolume

def main():

    timeStart = time.time()
    print("Loading the data.")
    data = load_volume('C:\drop', 256, 256, 256)
    print("Data loaded in {:.3f} s. Shape: {}".format(time.time() - timeStart, data.shape))

    smallChunk = data[0:6, ...]
    nonZero = np.count_nonzero(smallChunk)
    data = data[0:6, ...].astype(np.float)

    print(data.shape)

    prediction = predict_frame(data, 5, 3)
    print(prediction.shape)
    prediction.tofile('T:\prediction')


main()