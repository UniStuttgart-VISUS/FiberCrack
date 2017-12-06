import math
import colorsys
import csv

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.transform
import skimage.util

from FiberCrack.Dataset import Dataset

__all__ = ['plot_original_data_for_frame', 'plot_unmatched_cracks_for_frame',
           'plot_image_cracks_for_frame', 'plot_reference_crack_for_frame',
           'plot_crack_prediction_for_frame',
           'plot_feature_histograms_for_frame', 'plot_optic_flow_for_frame'
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


def plot_contour_overlay(axes, backgroundImage, binaryImage):
    """
    Plots contours over an image background.
    :param axes:
    :param backgroundImage:
    :param binaryImage:
    :return:
    """
    axes.imshow(backgroundImage.transpose(), origin='lower', cmap='gray')
    entropyContours = skimage.measure.find_contours(binaryImage.transpose(), 0.5)
    lineWidth = axes.get_window_extent().width / 500
    for n, contour in enumerate(entropyContours):
        axes.plot(contour[:, 1], contour[:, 0], linewidth=lineWidth, color='white')


def plot_original_data_for_frame(axes, frameData: np.ndarray, header):
    """
    Plots raw data for a given frame.
    :param axes:
    :param frameData:
    :param header:
    :return:
    """
    if 'W' in header:
        imageData0 = frameData[:, :, header.index('W')]
        axes[0].imshow(imageData0.transpose(), origin='lower', cmap='gray')
    axes[1].imshow(color_map_hsv(frameData[..., header.index('u')],
                                 frameData[..., header.index('v')], maxNorm=50.0)
                   .swapaxes(0, 1), origin='lower')
    # print("Sigma-plot")
    axes[2].imshow(frameData[:, :, header.index('sigma')].transpose(), origin='lower', cmap='gray')
    # print("Camera image")
    cameraImage = frameData[:, :, header.index('camera')]
    axes[3].imshow(cameraImage.transpose(), origin='lower', cmap='gray')
    if 'crackGroundTruth' in header:
        crackGroundTruth = frameData[:, :, header.index('crackGroundTruth')]
        plot_contour_overlay(axes[4], cameraImage, crackGroundTruth)

    return ['W', 'optic-flow', 'sigma', 'camera-image']


def plot_unmatched_cracks_for_frame(axes, frameData: np.ndarray, header):
    """
    Plots cracks detected from unmatched pixels, i.e. pixels
    that haven't been 'matched to', for a given frame.
    :param axes:
    :param frameData:
    :param header:
    :return: plot labels
    """
    matchedPixels = frameData[..., header.index('matched')]
    cameraImageData = frameData[..., header.index('camera')]

    axes[0].imshow(matchedPixels.transpose(), origin='lower', cmap='gray')

    matchedPixelsGaussThres = frameData[..., header.index('matchedPixelsGaussThres')]
    axes[1].imshow(matchedPixelsGaussThres.transpose(), origin='lower', cmap='gray')

    matchedPixelsCrack = frameData[..., header.index('matchedPixelsCrack')]
    axes[2].imshow(matchedPixelsCrack.transpose().astype(np.float), origin='lower', cmap='gray')

    plot_contour_overlay(axes[3], cameraImageData, matchedPixelsCrack)

    return ['matched-pixels', 'matched-pixels-gauss-thres',
            'matched-pixels-filtered', 'matched-pixels-crack']


def plot_image_cracks_for_frame(axes, frameData: np.ndarray, header):
    """
    Plot crack detected with image-based texture-feature techniques.
    :param axes:
    :param frameData:
    :param header:
    :return: plot names
    """
    cameraImageData = frameData[..., header.index('camera')]

    # Variance-based camera image crack extraction.
    cameraImageVar = frameData[..., header.index('cameraImageVar')]
    varianceFiltered = frameData[..., header.index('cameraImageVarFiltered')]

    axes[0].imshow(cameraImageVar.transpose(), origin='lower', cmap='gray')
    plot_contour_overlay(axes[1], cameraImageData, varianceFiltered)

    # Entropy-based camera image crack extraction.
    cameraImageEntropy = frameData[..., header.index('cameraImageEntropy')]
    entropyFiltered = frameData[..., header.index('cameraImageEntropyFiltered')]

    axes[2].imshow(cameraImageEntropy.transpose(), origin='lower', cmap='gray')
    plot_contour_overlay(axes[3], cameraImageData, entropyFiltered)

    # Unmatched pixels + entropy crack extraction.
    cracksFromUnmatchedAndEntropy = frameData[..., header.index('cracksFromUnmatchedAndEntropy')]
    plot_contour_overlay(axes[4], cameraImageData, cracksFromUnmatchedAndEntropy)

    # If we have enough axes, plot
    if len(axes) > 5:
        axes[5].imshow(cracksFromUnmatchedAndEntropy.transpose(), origin='lower', cmap='gray')

    return ['image-variance', 'image-variance-crack',
            'image-entropy', 'image-entropy-crack',
            'matched-pixels-entropy-crack', 'matched-pixels-entropy-crack-raw']


def plot_crack_prediction_for_frame(axes, frameData: np.ndarray, header):
    """
    Plot crack area predicted with machine learning techniques.
    :param axes:
    :param frameData:
    :param header:
    :return:
    """

    cameraImageData = frameData[..., header.index('camera')]

    prediction = frameData[..., header.index('crackPredictionSpatial')]

    axes[0].imshow(prediction.transpose(), origin='lower', cmap='gray')
    plot_contour_overlay(axes[1], cameraImageData, prediction)

    return ['crack-prediction', 'crack-prediction-overlay']


def plot_reference_crack_for_frame(axes, frameData: np.ndarray, header):
    """
    Plots the crack path in the reference frame.
    :param axes:
    :param frameData:
    :param header:
    :return:
    """
    binarySigmaFiltered = frameData[..., header.index('sigmaFiltered')]
    binarySigmaSkeleton = frameData[..., header.index('sigmaSkeleton')]
    binarySigmaSkeletonPruned = frameData[..., header.index('sigmaSkeletonPruned')]

    axes[0].imshow(binarySigmaFiltered.transpose(), origin='lower', cmap='gray')
    axes[1].imshow(binarySigmaSkeleton.transpose(), origin='lower', cmap='gray')
    axes[2].imshow(binarySigmaSkeletonPruned.transpose(), origin='lower', cmap='gray')

    return ['sigma-filtered', 'sigma-skeleton', 'sigma-crack']


def plot_feature_histograms_for_frame(axes, frameData, header):
    cameraImageVar = frameData[..., header.index('cameraImageVar')]
    cameraImageEntropy = frameData[..., header.index('cameraImageEntropy')]

    # Plot variance and entropy histograms for the image.
    varianceHist = np.histogram(cameraImageVar, bins=16, range=(0, np.max(cameraImageVar)))[0]
    entropyHist = np.histogram(cameraImageEntropy, bins=16, range=(0, np.max(cameraImageEntropy)))[0]

    varHistDomain = np.linspace(0, np.max(cameraImageVar), 16)
    entropyHistDomain = np.linspace(0, np.max(cameraImageEntropy), 16)

    axes[0].bar(varHistDomain * 100, varianceHist)
    axes[1].bar(entropyHistDomain, entropyHist)

    axes[0].axis('on')
    axes[1].axis('on')
    axes[0].get_yaxis().set_visible(False)
    axes[1].get_yaxis().set_visible(False)


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

    fig = plt.figure()
    fig.set_size_inches(4, 3)

    ax = fig.add_axes([0.15, 0.15, 0.7, 0.7])

    ax.grid(which='both')
    ax.set_ylabel('$Crack \/ area, mm^2$')
    ax.set_xlabel('Frame')

    # Plot the estimated crack area.
    crackAreaFeatures = dataset.get_str_array_attr('crackAreaNames')
    crackAreaFeaturesShort = dataset.get_str_array_attr('crackAreaNamesShort')

    featuresToPlot = ['unmatchedAndEntropy', 'trackingLoss']

    for i, name in enumerate(crackAreaFeatures):
        shortName = crackAreaFeaturesShort[i]
        if shortName not in featuresToPlot:
            continue

        crackArea = dataset.get_metadata_column(name + 'Physical')
        if name != 'crackAreaGroundTruth':
            ax.plot(crackArea, label='Estimated', linewidth=1.0)

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

        ax.legend(loc=0)
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

