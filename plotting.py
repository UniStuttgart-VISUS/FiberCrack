import math
import colorsys
import csv

from typing import Callable, Tuple, List, Dict, Any, Union, Type

import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.patches
import numpy as np
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.transform
import skimage.util

from Dataset import Dataset

__all__ = ['plot_original_data_for_frame', 'plot_unmatched_cracks_for_frame',
           'plot_image_cracks_for_frame', 'plot_reference_crack_for_frame',
           'plot_crack_prediction_for_frame',
           'plot_feature_histograms_for_frame', 'plot_optic_flow_for_frame',
           'plot_crack_area_chart', 'plot_data_mapping']


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


def draw_magnified_region(axNorm, magnifiedRegion):
    xLim, yLim = axNorm.get_xlim(), axNorm.get_ylim()
    imageSize = xLim[1] - xLim[0], yLim[1] - yLim[0]
    axNorm.add_patch(mpl.patches.Rectangle((int(magnifiedRegion[0][0] * imageSize[0]),
                                            int((1 - magnifiedRegion[1][1]) * imageSize[1])),
                                           int((magnifiedRegion[0][1] - magnifiedRegion[0][0]) * imageSize[0]),
                                           int((magnifiedRegion[1][1] - magnifiedRegion[1][0]) * imageSize[1]),
                                           linewidth=5, edgecolor='r', facecolor='none', zorder=1000))


def plot_contour_overlay(axes, backgroundImage, binaryImage, lineWidthFactor: float = 1.0):
    """
    Plots contours over an image background.
    :return:
    """
    axes.imshow(backgroundImage.transpose(), origin='lower', cmap='gray')
    entropyContours = skimage.measure.find_contours(binaryImage.transpose(), 0.5)
    lineWidth = axes.get_window_extent().width / 500 * lineWidthFactor
    for n, contour in enumerate(entropyContours):
        axes.plot(contour[:, 1], contour[:, 0], linewidth=lineWidth, color='white')

    return axes


def plot_original_data_for_frame(axisBuilder: Callable[[str], plt.Axes], frameData: np.ndarray, header):
    """
    Plots raw data for a given frame.
    :param axisBuilder:
    :param frameData:
    :param header:
    :return:
    """
    if 'W' in header:
        imageData0 = frameData[:, :, header.index('W')]
        axisBuilder('W').imshow(imageData0.transpose(), origin='lower', cmap='gray')
    axisBuilder('optic-flow').imshow(color_map_hsv(frameData[..., header.index('u')],
                                     frameData[..., header.index('v')], maxNorm=50.0)
                                     .swapaxes(0, 1), origin='lower')
    # print("Sigma-plot")
    axisBuilder('sigma').imshow(np.invert(frameData[:, :, header.index('sigma')].transpose() > 0), origin='lower', cmap='gray')
    # print("Camera image")
    cameraImage = frameData[:, :, header.index('camera')]
    axisBuilder('camera-image').imshow(cameraImage.transpose(), origin='lower', cmap='gray')
    if 'crackGroundTruth' in header:
        crackGroundTruth = frameData[:, :, header.index('crackGroundTruth')]
        plot_contour_overlay(axisBuilder('crack-ground-truth'), cameraImage, crackGroundTruth)


def plot_unmatched_cracks_for_frame(axisBuilder: Callable[[str], plt.Axes], frameData: np.ndarray, header, magnifiedRegion):
    """
    Plots cracks detected from unmatched pixels, i.e. pixels
    that haven't been 'matched to', for a given frame.
    :param axisBuilder:
    :param frameData:
    :param header:
    :return: plot labels
    """
    matchedPixels = frameData[..., header.index('matched')]
    cameraImageData = frameData[..., header.index('camera')]

    axisBuilder('matched-pixels').imshow(np.invert(matchedPixels.transpose() > 0), origin='lower', cmap='gray')

    matchedPixelsGaussThres = frameData[..., header.index('matchedPixelsGaussThres')]
    axisBuilder('matched-pixels-gauss-thres').imshow(np.invert(matchedPixelsGaussThres.transpose() > 0), origin='lower', cmap='gray')

    matchedPixelsObjectsRemoved = frameData[..., header.index('matchedPixelsObjectsRemoved')]
    axisBuilder('matched-pixels-objects-removed')\
        .imshow(np.invert(matchedPixelsObjectsRemoved.transpose() > 0), origin='lower', cmap='gray')

    matchedPixelsHolesRemoved = frameData[..., header.index('matchedPixelsHolesRemoved')]
    axisBuilder('matched-pixels-holes-removed')\
        .imshow(np.invert(matchedPixelsHolesRemoved.transpose() > 0), origin='lower', cmap='gray')

    plot_contour_overlay(axisBuilder('matched-pixels-holes-removed-crack'),
                         cameraImageData, matchedPixelsHolesRemoved)

    matchedPixelsCrack = frameData[..., header.index('matchedPixelsCrack')]
    axisBuilder('matched-pixels-crack-raw').imshow(matchedPixelsCrack.transpose(), origin='lower', cmap='gray')
    plot_contour_overlay(axisBuilder('matched-pixels-crack'), cameraImageData, matchedPixelsCrack)
    ax = plot_contour_overlay(axisBuilder('matched-pixels-crack-thin'), cameraImageData, matchedPixelsCrack, lineWidthFactor=0.25)
    draw_magnified_region(ax, magnifiedRegion)


def plot_image_cracks_for_frame(axisBuilder: Callable[[str], plt.Axes], frameData: np.ndarray, header, magnifiedRegion):
    """
    Plot crack detected with image-based texture-feature techniques.
    :return: plot names
    """
    cameraImageData = frameData[..., header.index('camera')]

    # Variance-based camera image crack extraction.
    cameraImageVar = frameData[..., header.index('cameraImageVar')]
    varianceFiltered = frameData[..., header.index('cameraImageVarFiltered')]

    axisBuilder('image-variance').imshow(cameraImageVar.transpose(), origin='lower', cmap='gray')
    plot_contour_overlay(axisBuilder('image-variance-crack'), cameraImageData, varianceFiltered)

    # Entropy-based camera image crack extraction.
    cameraImageEntropy = frameData[..., header.index('cameraImageEntropy')]
    cameraImageEntropyBinary = frameData[..., header.index('cameraImageEntropyBinary')]
    entropyFiltered = frameData[..., header.index('cameraImageEntropyFiltered')]

    axisBuilder('image-entropy').imshow(cameraImageEntropy.transpose(), origin='lower', cmap='gray')
    axisBuilder('image-entropy-binary').imshow(cameraImageEntropyBinary.transpose(), origin='lower', cmap='gray')
    axisBuilder('image-entropy-crack-raw').imshow(entropyFiltered.transpose(), origin='lower', cmap='gray')
    plot_contour_overlay(axisBuilder('image-entropy-crack'), cameraImageData, entropyFiltered)

    # Unmatched pixels + entropy crack extraction.
    hybridUnmatchedDilated = np.invert(frameData[..., header.index('hybridUnmatchedDilated')] > 0)
    hybridEntropyBinary = frameData[..., header.index('hybridEntropyBinary')]
    hybridCracks = frameData[..., header.index('hybridCracks')]

    axisBuilder('hybrid-matched-dilated').imshow(np.invert(hybridUnmatchedDilated.transpose() > 0), origin='lower', cmap='gray')
    axisBuilder('hybrid-entropy-binary').imshow(hybridEntropyBinary.transpose(), origin='lower', cmap='gray')
    plot_contour_overlay(axisBuilder('hybrid-entropy-binary-crack'), cameraImageData, hybridEntropyBinary)
    plot_contour_overlay(axisBuilder('hybrid-crack'), cameraImageData, hybridCracks)
    ax = plot_contour_overlay(axisBuilder('hybrid-crack-thin'), cameraImageData, hybridCracks, lineWidthFactor=0.25)
    # Here we only draw a rectangle around the area that should be magnified, and do the magnification externally.
    draw_magnified_region(ax, magnifiedRegion)

    axisBuilder('hybrid-crack-raw').imshow(hybridCracks.transpose(), origin='lower', cmap='gray')


def plot_crack_prediction_for_frame(axisBuilder: Callable[[str], plt.Axes], frameData: np.ndarray, header):
    """
    Plot crack area predicted with machine learning techniques.
    :param axisBuilder:
    :param frameData:
    :param header:
    :return:
    """

    cameraImageData = frameData[..., header.index('camera')]

    prediction = frameData[..., header.index('crackPredictionSpatial')]

    axisBuilder('crack-prediction').imshow(prediction.transpose(), origin='lower', cmap='gray')
    plot_contour_overlay(axisBuilder('crack-prediction-overlay'), cameraImageData, prediction)


def plot_reference_crack_for_frame(axisBuilder: Callable[[str], plt.Axes], frameData: np.ndarray, header):
    """
    Plots the crack path in the reference frame.
    :param axisBuilder: 
    :param frameData:
    :param header:
    :return:
    """
    binarySigmaFiltered = frameData[..., header.index('sigmaFiltered')]
    binarySigmaSkeleton = frameData[..., header.index('sigmaSkeleton')]
    binarySigmaSkeletonPruned = frameData[..., header.index('sigmaSkeletonPruned')]

    axisBuilder('sigma-filtered').imshow(binarySigmaFiltered.transpose(), origin='lower', cmap='gray')
    axisBuilder('sigma-skeleton').imshow(binarySigmaSkeleton.transpose(), origin='lower', cmap='gray')
    axisBuilder('sigma-crack').imshow(binarySigmaSkeletonPruned.transpose(), origin='lower', cmap='gray')


def plot_feature_histograms_for_frame(axisBuilder: Callable[[str], plt.Axes], frameData, header):
    cameraImageVar = frameData[..., header.index('cameraImageVar')]
    cameraImageEntropy = frameData[..., header.index('cameraImageEntropy')]

    # Plot variance and entropy histograms for the image.
    varianceHist = np.histogram(cameraImageVar, bins=16, range=(0, np.max(cameraImageVar)))[0]
    entropyHist = np.histogram(cameraImageEntropy, bins=16, range=(0, np.max(cameraImageEntropy)))[0]

    varHistDomain = np.linspace(0, np.max(cameraImageVar), 16)
    entropyHistDomain = np.linspace(0, np.max(cameraImageEntropy), 16)

    varHistAx = axisBuilder('variance-histogram')
    entHistAx = axisBuilder('entropy-histogram')

    varHistAx.bar(varHistDomain * 100, varianceHist)
    entHistAx.bar(entropyHistDomain, entropyHist)

    varHistAx.axis('on')
    entHistAx.axis('on')
    varHistAx.get_yaxis().set_visible(False)
    entHistAx.get_yaxis().set_visible(False)


def plot_crack_area_chart(dataset: 'Dataset', csvOutPath: str=None):
    """
    Plots area of the crack (a scalar) detected by various methods
    against time (or more precisely, frame index).

    :param csvOutPath:
    :param dataset:
    :return:
    """

    frameMap = dataset.get_frame_map()

    result = []
    resultHeader = []

    result.append(np.arange(0, dataset.get_frame_number()).tolist())
    resultHeader.append('frameIndex')
    result.append(frameMap)
    resultHeader.append('frameNumber')
    result.append(dataset.get_metadata_column('Strain (%)'))
    resultHeader.append('strainPercent')

    fig = plt.figure()
    fig.set_size_inches(4, 3)

    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])

    ax.grid(which='both')
    ax.set_ylabel('$Crack \/ area, mm^2$')
    ax.set_xlabel('Frame')

    # Plot the estimated crack area.
    crackAreaFeatures = dataset.get_str_array_attr('crackAreaNames')
    crackAreaFeaturesShort = dataset.get_str_array_attr('crackAreaNamesShort')

    featuresToPlot = ['hybrid', 'entropy', 'unmatchedPixels', 'trackingLoss']

    for i, name in enumerate(crackAreaFeatures):
        shortName = crackAreaFeaturesShort[i]
        if shortName not in featuresToPlot:
            continue

        crackArea = dataset.get_metadata_column(name + 'Physical')
        if name != 'crackAreaGroundTruth':
            ax.plot(crackArea, label='Est. ({})'.format(shortName), linewidth=1.0)

            result.append(crackArea)
            resultHeader.append(name + 'Physical')

    # A hacky way to convert x axis labels from frame indices to frame numbers.
    locs, labels = plt.xticks()
    for i in range(len(labels)):
        frameIndex = int(locs[i])
        fullFrameNumber = str(frameMap[frameIndex]) if 0 <= frameIndex < len(frameMap) else ''
        labels[i].set_text(fullFrameNumber)
        plt.xticks(locs, labels)

    if (dataset.has_numpy_array_attr('crackAreaGroundTruth')):

        # Convert frame numbers into frame indices, i.e. align with X axis.
        def get_closest_frame_index(frameNumber):
            return np.count_nonzero(np.array(frameMap, dtype=np.int)[:-1] < frameNumber)

        groundTruth = dataset.get_numpy_array_attr('crackAreaGroundTruth')
        header = dataset.get_str_array_attr('crackAreaGroundTruthHeader')

        # Plot the ground truth.
        # ax.scatter(groundTruth[:, 0], groundTruth[:, 1], marker='x', label='Manually measured')
        ax.errorbar(groundTruth[:, header.index('frame')],
                    groundTruth[:, header.index('average')],
                    groundTruth[:, header.index('std')] * 3.0,
                    fmt='x', label='Manual (with std)')

        # Export data to CSV.

        # Ground truth data is sparse (not for all frames).
        # Expand a ground truth column into a full column, with a value for each frame.
        def sparse_data_to_full_column(data: np.ndarray):
            column = np.zeros((dataset.get_frame_number(), 1), dtype=np.float)
            column[groundTruth[:, header.index('frame')].astype(np.int)] = data[..., np.newaxis]
            return column[:, 0].tolist()

        result.append(sparse_data_to_full_column(groundTruth[:, header.index('average')]))
        resultHeader.append('crackAreaGroundTruthAverage')
        result.append(sparse_data_to_full_column(groundTruth[:, header.index('std')] * 3.0))
        resultHeader.append('crackAreaGroundTruthTripleStd')

        ax.bar([0], [0], color='g', alpha=0.5, label='Percentage error')  # Empty series, just for legend.

        ax.legend(loc=0, fontsize='x-small')
        ax.set_ylim(bottom=0)

        # Plot the percentage error on the second Y-axis.
        ax2 = ax.twinx()

        # Compute and plot crack area percentage error for each frame with ground truth available.
        crackAreaFeatureName = crackAreaFeatures[crackAreaFeaturesShort.index(featuresToPlot[0])]
        crackArea = dataset.get_metadata_column(crackAreaFeatureName + 'Physical')

        percentageError = np.zeros(groundTruth.shape[0])
        for j, frameIndex in enumerate(groundTruth[:, 0]):
            truth = groundTruth[j, 1]
            if truth == 0:
                percentageError[j] = 0
                continue

            estimated = crackArea[int(frameIndex)]
            percentageError[j] = math.fabs((truth - estimated) / truth) * 100

        ax2.bar(groundTruth[:, 0], percentageError, color='g', alpha=0.5, zorder=-10)
        ax2.yaxis.set_major_locator(ticker.MultipleLocator(10))

        ax2.set_ylabel('Error, %')
        ax2.set_ylim(bottom=0, top=100)

        result.append(sparse_data_to_full_column(percentageError))
        resultHeader.append('crackAreaPercentageError')

    if csvOutPath is not None:
        print("Writing crack area chart data to a CSV file at {}".format(csvOutPath))
        with open(csvOutPath, 'w', newline='\n', encoding='utf-8') as f:
            writer = csv.writer(f)
            writer.writerow(resultHeader)
            writer.writerows(map(list, zip(*result)))  # Transpose and write out.

    return fig


def plot_optic_flow_for_frame(ax, dataset: 'Dataset', frameIndex):
    h5Data, header, frameMap, *r = dataset.unpack_vars()
    imageShift = dataset.get_image_shift()
    frameSize = dataset.get_frame_size()
    min, max, step = dataset.get_data_image_mapping()

    imageShift = imageShift[frameIndex]

    frameFlow = h5Data[frameIndex, :, :, :][:, :, [header.index('u'), header.index('v')]]
    frameFlow[:, :, 0] = ((frameFlow[:, :, 0] - imageShift[0]) / step[0])
    frameFlow[:, :, 1] = ((frameFlow[:, :, 1] - imageShift[1]) / step[1])


    # Shift the array to transform to the current frame coordinates.
    cropMin = np.maximum(np.rint(imageShift / step).astype(np.int), 0)
    cropMax = np.minimum(np.rint(frameSize + imageShift / step).astype(np.int), frameSize)
    cropSize = cropMax - cropMin
    plottedFlow = np.zeros(frameFlow.shape)
    plottedFlow[0:cropSize[0], 0:cropSize[1], :] = frameFlow[cropMin[0]:cropMax[0], cropMin[1]:cropMax[1], :]
    plottedFlow = plottedFlow.swapaxes(0, 1)

    X, Y = np.meshgrid(np.arange(0, frameSize[0]), np.arange(0, frameSize[1]))

    ax.quiver(X, Y, plottedFlow[:, :, 0], plottedFlow[:, :, 1], angles='uv', scale_units='xy', scale=1,
              width=0.003, headwidth=2)


def plot_data_mapping(dataset: 'Dataset'):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    mappingText = 'Data to camera image mapping\n'
    mappingText += 'Data size: {} Image size: {}\n'.format(
        list(dataset.get_frame_size()), dataset.get_attr('cameraImageSize'))
    mappingText += 'Physical size: {}\n'.format(dataset.get_attr('physicalFrameSize'))
    mappingText += 'Min: {} Max: {} Step: {}\n'.format(*dataset.get_data_image_mapping())

    ax.text(0.1, 0.1, mappingText)

    return fig

