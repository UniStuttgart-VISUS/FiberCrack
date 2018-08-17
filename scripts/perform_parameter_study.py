import time
import os

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from FiberCrack.FiberCrackConfig import FiberCrackConfig
from FiberCrack.fiber_crack import fiber_crack_run

import PythonExtras.common_data_tools as common_data_tools
import PythonExtras.pyplot_extras as pyplot_extras


def perform_parameter_analysis():

    rootOutDirPath = 'T:\\out\\fiber-crack\\parameter-study'
    # baseConfigPath = '..\\configs\\ptfe-epoxy.json'
    baseConfigPath = '..\\configs\\steel-epoxy.json'
    parameterAxes = {
        'unmatchedPixelsMorphologyDepth': [0, 1, 2, 3, 4],
        'unmatchedPixelsObjectsThreshold': 1 / np.asarray([10, 25, 50, 75, 100]),
        'unmatchedPixelsHolesThreshold': 1 / np.asarray([0.1, 1, 3, 6, 9, 12, 20]),
        'hybridKernelMultiplier':  [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0],
        'hybridDilationDepth': [0, 1, 3, 5, 7]
    }

    # frameToExport = 3330
    frameToExport = 1150
    runExperiments = True

    baseConfig = FiberCrackConfig()
    baseConfig.read_from_file(baseConfigPath)
    datasetName = os.path.splitext(baseConfig.dataConfig.metadataFilename)[0]
    pdfPath = os.path.join(rootOutDirPath, '{}.pdf'.format(datasetName))
    print("Plotting to {}".format(pdfPath))
    pdf = PdfPages(pdfPath)

    for axisParam, valueRange in parameterAxes.items():
        config = FiberCrackConfig()
        config.read_from_file(baseConfigPath)

        config.enablePrediction = False  # We don't need the prediction, save the time.
        config.recomputeResults = True   # The _results_ caching in FiberCrack is working poorly and should be avoided.

        crackAreas = []
        frameIndices = None
        frameNumbers = None

        figure = plt.figure(dpi=300)
        ax = figure.add_subplot(1, 1, 1)

        for i, value in enumerate(valueRange):
            config.__dict__[axisParam] = value
            outDirName = '{}_{}_{:02d}_{}'.format(config.dataConfig.metadataFilename, axisParam, i, value)
            config.outDir = os.path.join(rootOutDirPath, outDirName)
            if not os.path.exists(config.outDir):
                os.makedirs(config.outDir)

            if runExperiments:
                print("[{}] Starting the run '{}'".format(time.strftime('%d.%m.%y %H:%M:%S'), outDirName))
                # fiber_crack_run('export-figures-only-area', config)
                fiber_crack_run('export-figures', config, frame=3330)
            elif not os.path.exists(config.outDir):
                raise RuntimeError("Results dir is missing: '{}'".format(config.outDir))

            resultsSubdir = 'figures-{}'.format(config.dataConfig.metadataFilename)
            csvFilePath = os.path.join(config.outDir, resultsSubdir, 'crack-area-data.csv')

            csvData, csvHeader = common_data_tools.read_csv_data(csvFilePath)
            crackAreas.append(csvData[:, csvHeader.index('crackAreaHybridPhysical')])
            dicCrackArea = csvData[:, csvHeader.index('crackAreaUnmatchedPixelsPhysical')]

            if frameNumbers is None:
                frameIndices = csvData[:, csvHeader.index('frameIndex')]
                frameNumbers = csvData[:, csvHeader.index('frameNumber')]

            ax.plot(frameNumbers, crackAreas[-1],
                    label='{}-{:.3f}'.format(axisParam, value), lw=0.5)
            ax.plot(frameNumbers, dicCrackArea,
                    label='{}-{:.3f}-DIC'.format(axisParam, value), lw=0.5, linestyle='dashed')

            if 'crackAreaGroundTruthAverage' in csvHeader:
                groundTruth = csvData[:, csvHeader.index('crackAreaGroundTruthAverage')]
                ax.scatter(frameNumbers[groundTruth != 0], groundTruth[groundTruth != 0], marker='x')

        exportedHeader = ['frameIndex', 'frameNumber', *['{}-{}'.format(axisParam, val) for val in valueRange]]
        exportedData = np.vstack((frameIndices, frameNumbers, *crackAreas)).transpose()

        exportedCsvFilepath = os.path.join(rootOutDirPath, '{}-{}.csv'.format(datasetName, axisParam))

        # noinspection PyTypeChecker
        np.savetxt(exportedCsvFilepath, exportedData,
                   fmt='%.6f', delimiter='\t',
                   header='\t'.join(exportedHeader), comments='', encoding='utf-8')

        ax.legend()
        pdf.savefig(figure)
        plt.close(figure)

    pdf.close()


perform_parameter_analysis()
