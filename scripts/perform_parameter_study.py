import os

import numpy as np

from FiberCrack.FiberCrackConfig import FiberCrackConfig
from FiberCrack.fiber_crack import fiber_crack_run


def main():

    rootOutDirPath = 'T:\\out\\fiber-crack\\parameter-study'
    baseConfigPath = '..\\configs\\ptfe-epoxy.json'
    parameterAxes = {
        'unmatchedPixelsMorphologyDepth': [0, 1, 2, 3, 4],
        'unmatchedPixelsObjectsThreshold': 1 / np.asarray([10, 25, 50, 75, 100]),
        'unmatchedPixelsHolesThreshold': 1 / np.asarray([1, 3, 6, 9, 12]),
        'hybridKernelMultiplier':  [0.1, 0.25, 0.5, 1.0],
        'hybridDilationDepth': [1, 3, 5]
    }

    for axisParam, valueRange in parameterAxes.items():
        config = FiberCrackConfig()
        config.read_from_file(baseConfigPath)

        config.enablePrediction = False  # We don't need the prediction, save the time.
        config.recomputeResults = True   # The _results_ caching in FiberCrack is working poorly and should be avoided.

        for i, value in enumerate(valueRange):
            config.__dict__[axisParam] = value
            outDirName = '{}_{}_{:02d}_{}'.format(config.dataConfig.metadataFilename, axisParam, i, value)
            config.outDir = os.path.join(rootOutDirPath, outDirName)
            if not os.path.exists(config.outDir):
                os.makedirs(config.outDir)

            print("Starting the run '{}'".format(outDirName))
            fiber_crack_run('export-figures-only-area', config)


main()
