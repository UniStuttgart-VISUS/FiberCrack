import time
import os
import itertools

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

from FiberCrack.FiberCrackConfig import FiberCrackConfig
from FiberCrack.fiber_crack import fiber_crack_run

import PythonExtras.common_data_tools as common_data_tools
import PythonExtras.pyplot_extras as pyplot_extras


def perform_parameter_analysis():

    rootOutDirPath = 'T:\\out\\fiber-crack\\parameter-study'
    paperOutDirPath = 'T:\\projects\\papers\\MontrealGeneral\\Papers\\Applied\\'

    baseConfigPath = '..\\configs\\ptfe-epoxy.json'
    frameToExport = 3330

    # baseConfigPath = '..\\configs\\steel-epoxy.json'
    # frameToExport = 1150

    parameterAxes = {
        # 'unmatchedPixelsMorphologyDepth': [0, 1, 2, 3, 4],
        ### 'unmatchedPixelsObjectsThreshold': 1 / np.asarray([10, 25, 50, 75, 100]),
        ### 'unmatchedPixelsHolesThreshold': 1 / np.asarray([0.1, 1, 3, 6, 9, 12, 20]),
        'hybridKernelMultiplier':  [0.1, 0.25, 0.4, 0.5, 0.6, 0.75, 1.0],
        # 'hybridKernelMultiplier': [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.9, 1.0],
        # 'hybridDilationDepth': [0, 1, 3, 5, 7],
        # 'entropyThreshold': [0.6, 0.8, 1.0, 1.2, 1.4, 1.6]
    }
    chosenValues = {
        'unmatchedPixelsMorphologyDepth': 3,
        'hybridKernelMultiplier': 0.4,
        'hybridDilationDepth': 3,
        'entropyThreshold': 1.0
    }

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

        crackAreasHybrid = []
        crackAreasDic = []
        frameIndices = None
        frameNumbers = None
        strainValues = None
        groundTruth = None

        figure = plt.figure(dpi=300)
        ax = figure.add_subplot(1, 1, 1)

        for i, value in enumerate(valueRange):
            config.__dict__[axisParam] = value
            outDirName = '{}_{}_{}'.format(config.dataConfig.metadataFilename, axisParam, value)
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

            csvData, csvHeader = common_data_tools.read_csv_data(csvFilePath, delimiter=',')
            crackAreasHybrid.append(csvData[:, csvHeader.index('crackAreaHybridPhysical')].copy())
            crackAreasDic.append(csvData[:, csvHeader.index('crackAreaUnmatchedPixelsPhysical')].copy())

            if frameNumbers is None:
                frameIndices = csvData[:, csvHeader.index('frameIndex')]
                frameNumbers = csvData[:, csvHeader.index('frameNumber')]
                strainValues = csvData[:, csvHeader.index('strainPercent')]

            ax.plot(strainValues, crackAreasHybrid[-1],
                    label='{}-{:.3f}'.format(axisParam, value), lw=0.5)
            ax.plot(strainValues, crackAreasDic[-1],
                    label='{}-{:.3f}-DIC'.format(axisParam, value), lw=0.5, linestyle='dashed')

            if 'crackAreaGroundTruthAverage' in csvHeader:
                groundTruth = csvData[:, csvHeader.index('crackAreaGroundTruthAverage')]

            # For Steel-Epoxy there's no ground truth images, just a small csv with scalars.
            if datasetName == 'SteelEpoxy.csv':
                groundTruthCsvPath = 'T:\\data\\montreal-full\\Experiments\\Steel-Epoxy\\Steel_measured_Ilyass.csv'
                csvData, csvHeader = common_data_tools.read_csv_data(groundTruthCsvPath, delimiter=';')
                truthSparse = csvData[:, (csvHeader.index('Strain'), csvHeader.index('Average'))]
                # Convert from the sparse representation into a dense one, fill in with zeros.
                groundTruth = np.zeros_like(strainValues)
                for strain, truthValue in truthSparse:
                    groundTruth[np.argwhere(strainValues == strain)] = truthValue

            if groundTruth is not None:
                ax.scatter(strainValues[groundTruth != 0], groundTruth[groundTruth != 0], marker='x')

        ax.legend()
        pdf.savefig(figure)
        plt.close(figure)

        exportedHeader = ['frameIndex', 'frameNumber', 'strainPercent',
                          *['{}-{:.3f}-{}'.format(axisParam, val, crackType) for crackType, val in
                            itertools.product(['hybrid', 'dic'], valueRange)]
                          ]
        # Check that the ordering is correct.
        assert exportedHeader[-2] == '{}-{:.3f}-{}'.format(axisParam, valueRange[-2], 'dic')

        arraysToStack = (frameIndices, frameNumbers, strainValues, *crackAreasHybrid, *crackAreasDic)
        if groundTruth is not None:
            arraysToStack += (np.ma.array(groundTruth, mask=(groundTruth == 0)),)
            exportedHeader += ['manualMean']

        exportedData = np.ma.vstack(arraysToStack).transpose()
        exportedCsvFilepath = os.path.join(rootOutDirPath, 'param-study-{}-{}.csv'.format(datasetName, axisParam))

        common_data_tools.write_to_csv(exportedCsvFilepath, exportedData, exportedHeader, addPaddingColumn=True)
        # Write the CSV to the paper as well.
        paperCsvFilepath = os.path.join(paperOutDirPath, 'data', 'param-study-{}-{}.csv'.format(datasetName, axisParam))
        common_data_tools.write_to_csv(paperCsvFilepath, exportedData, exportedHeader, addPaddingColumn=True)

        # Export groundTruth into a separate CSV.
        common_data_tools.write_to_csv(
            os.path.join(rootOutDirPath, 'manual-measurement-{}.csv'.format(datasetName)),
            np.vstack((frameNumbers[groundTruth != 0],
                       strainValues[groundTruth != 0],
                       groundTruth[groundTruth != 0])).transpose(),
            ['frameNumber', 'strainPercent', 'manualMean']
        )

        # Draw a rough parameter study figure.
        parameterPdf = PdfPages(os.path.join(rootOutDirPath,
                                             'param-study-{}-{}.pdf'.format(datasetName, axisParam)))
        figure = plt.figure(dpi=300, figsize=(4, 3))
        ax = figure.add_subplot(1, 1, 1)

        plotDic = axisParam == 'unmatchedPixelsMorphologyDepth'
        areasToPlot = crackAreasDic if plotDic else crackAreasHybrid
        for i, area in enumerate(areasToPlot):
            formatTemplate = 'Estimated (DIC), {}' if plotDic else 'Estimated, {}'
            chosen = chosenValues[axisParam]
            lineStyle = 'solid' if valueRange[i] == chosen else 'dashed'
            ax.plot(strainValues, area,
                    label=formatTemplate.format(valueRange[i]), lw=0.75, linestyle=lineStyle)

        ax.scatter(strainValues[groundTruth != 0], groundTruth[groundTruth != 0], marker='x', label='Measured')

        ax.set_ylim(0.0, 6.0)
        ax.set_ylabel("Crack area [mm^2]")
        ax.set_xlabel("Strain [%]")
        ax.legend()
        figure.tight_layout()

        parameterPdf.savefig(figure)
        plt.close(figure)
        parameterPdf.close()

    pdf.close()


perform_parameter_analysis()
