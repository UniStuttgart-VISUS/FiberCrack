"""
 Requires crack propagation data for two datasets to be manually exported
 from the main application.
"""

import os

import numpy as np
import matplotlib.pyplot as plt
import h5py

crackDataPath = 'T:/projects/SimtechOne/out/fiber/crack-propagation'

datasetNames = ['PTFE-Epoxy.csv', 'Steel-Epoxy.csv']
datasetShifts = [[0, 0], [-25, -30]]

def main():

    datasetPaths = [os.path.join(crackDataPath, 'crack-propagation_{}.hdf'.format(n)) for n in datasetNames]

    datasetNumber = len(datasetNames)

    datasetH5Files = [h5py.File(path, 'r') for path in datasetPaths]

    strainArrays = [f['strain'][...] for f in datasetH5Files]

    strainMin = min([np.min(a) for a in strainArrays])
    strainMax = max([np.max(a) for a in strainArrays])

    frameSizes = [np.asarray(f['sigmaSkeleton'].shape[1:3]) for f in datasetH5Files]
    maxFrameSize = [max([frameSizes[i][d] for i in range(datasetNumber)]) for d in range(0, 2)]

    maxFrameSize = tuple(maxFrameSize)
    print("Frame sizes: {}".format(frameSizes))
    print("Max frame size: {}".format(maxFrameSize))

    currentIndices = [0] * datasetNumber
    for strainIndex, strainValue in enumerate(np.arange(int(strainMin), int(strainMax), 0.25)):

        for i in range(datasetNumber):
            while currentIndices[i] < strainArrays[i].shape[0] - 1 and \
                  strainArrays[i][currentIndices[i]] < strainValue:
                currentIndices[i] += 1

        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)

        # Create an empty RGB image.
        image = np.zeros(maxFrameSize + (3,), dtype=np.float)

        for i in range(datasetNumber):
            sigmaSkeleton = datasetH5Files[i]['sigmaSkeleton'][currentIndices[i], ...]
            sigmaSkeleton = np.pad(sigmaSkeleton, [(0, maxFrameSize[d] - sigmaSkeleton.shape[d]) for d in range(0, 2)],
                                   mode='constant', constant_values=0)

            sigmaSkeleton = np.roll(sigmaSkeleton, datasetShifts[i][0], axis=0)
            sigmaSkeleton = np.roll(sigmaSkeleton, datasetShifts[i][1], axis=1)

            image[sigmaSkeleton == 1] += (1.0, 0.0, 0.0) if i == 0 else (0.0, 0.0, 1.0)

        ax.imshow(image.swapaxes(0, 1), origin='lower')
        strainValues = [strainArrays[i][currentIndices[i]] for i in range(datasetNumber)]
        strainValues = ['{:.3f}'.format(s) for s in strainValues]
        frameNumbers = [datasetH5Files[i]['frameMap'][currentIndices[i]] for i in range(datasetNumber)]

        ax.text(0.1, 0.1, "Strain: {}; Frame: {}".format(strainValues, frameNumbers), color='white')

        fig.savefig(os.path.join(crackDataPath, 'crack-comparison_{}.png'.format(strainIndex)))


main()
