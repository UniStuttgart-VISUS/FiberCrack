import math
import colorsys

import matplotlib.pyplot as plt
import numpy as np
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.transform
import skimage.util


__all__ = ['plot_original_data_for_frame', 'plot_unmatched_cracks_for_frame',
           'plot_image_cracks_for_frame', 'plot_reference_crack_for_frame',
           'plot_feature_histograms_for_frame',
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
    axes.imshow(backgroundImage.transpose(), origin='lower', cmap='gray')
    entropyContours = skimage.measure.find_contours(binaryImage.transpose(), 0.5)
    for n, contour in enumerate(entropyContours):
        axes.plot(contour[:, 1], contour[:, 0], linewidth=1, color='white')


def plot_original_data_for_frame(axes, frameData, header):
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


def plot_unmatched_cracks_for_frame(axes, frameData, header):
    """
    Plots cracks detected from unmatched pixels, i.e. pixels
    that haven't been 'matched to', for a given frame.
    :param axes:
    :param frameData:
    :param header:
    :return:
    """
    matchedPixels = frameData[..., header.index('matched')]
    cameraImageData = frameData[..., header.index('camera')]

    axes[0].imshow(matchedPixels.transpose(), origin='lower', cmap='gray')
    matchedPixelsGauss = frameData[..., header.index('matchedPixelsGauss')]

    axes[1].imshow(matchedPixelsGauss.transpose(), origin='lower', cmap='gray')
    matchedPixelsGaussThres = frameData[..., header.index('matchedPixelsGaussThres')]

    axes[2].imshow(matchedPixelsGaussThres.transpose(), origin='lower', cmap='gray')
    matchedPixelsGaussThresClean = frameData[..., header.index('matchedPixelsGaussThresClean')]

    axes[3].imshow(matchedPixelsGaussThresClean.transpose().astype(np.float), origin='lower', cmap='gray')

    plot_contour_overlay(axes[4], cameraImageData, matchedPixelsGaussThresClean)


def plot_image_cracks_for_frame(axes, frameData, header):
    """
    Plot crack detected with image-based texture-feature techniques.
    :param axes:
    :param frameData:
    :param header:
    :return:
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


def plot_reference_crack_for_frame(axes, frameData, header):
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


def plot_crack_area_chart(dataset, crackAreaGroundTruthPath=None):
    """
    Plots area of the crack (a scalar) detected by various methods
    against time (or more precisely, frame index).
    :param dataset:
    :return:
    """
    frameMap = dataset.get_frame_map()

    fig = plt.figure()
    fig.suptitle("Crack area in the current frame")
    ax = fig.add_subplot(1, 1, 1)

    ax.grid(which='both')
    ax.set_ylabel('Millimeter^2')

    # Plot the estimated crack area.
    ax.plot(dataset.get_metadata_column('crackAreaVariancePhysical'), label='Variance estimation')
    ax.plot(dataset.get_metadata_column('crackAreaEntropyPhysical'), label='Entropy  estimation')
    ax.plot(dataset.get_metadata_column('crackAreaUnmatchedAndEntropyPhysical'), label='Unmatched&Entropy  estimation')

    # A hacky way to convert x axis labels from frame indices to frame numbers.
    locs, labels = plt.xticks()
    for i in range(len(labels)):
        frameIndex = int(locs[i])
        frameNumber = str(frameMap[frameIndex]) if 0 <= frameIndex < len(frameMap) else ''
        labels[i].set_text(frameNumber)
        plt.xticks(locs, labels)

    if (crackAreaGroundTruthPath):
        # Convert frame numbers into frame indices, i.e. align with X axis.
        def get_closest_frame_index(frameNumber):
            return np.count_nonzero(np.array(frameMap, dtype=np.int)[:-1] < frameNumber)

        groundTruth = np.genfromtxt(crackAreaGroundTruthPath, delimiter=',')
        groundTruth[:, 0] = [get_closest_frame_index(frameNumber) for frameNumber in groundTruth[:, 0]]

        # Compute crack area percentage error for each frame with ground truth available.
        percentageError = np.zeros(groundTruth.shape[0])
        crackAreaEntropy = dataset.get_metadata_column('crackAreaUnmatchedAndEntropyPhysical')
        for i, frameIndex in enumerate(groundTruth[:, 0]):
            truth = groundTruth[i, 1]
            estimated = crackAreaEntropy[int(frameIndex)]
            percentageError[i] = math.fabs((truth - estimated) / truth) * 100

        # Plot the ground truth.
        ax.scatter(groundTruth[:, 0], groundTruth[:, 1], marker='x', label='Measured by hand')

        # Plot the percentage error on the second Y-axis.
        ax2 = ax.twinx()
        ax2.scatter(groundTruth[:, 0], percentageError, marker='o', s=8, c='green', label='Percentage error (entropy)')
        ax2.set_ylim(bottom=0)
        ax2.set_ylabel('Error, %')

    plt.legend()

    return fig


def plot_optic_flow(pdf, dataset, frameIndex):
    h5Data, header, frameMap, *r = dataset.unpack_vars()
    imageShift = dataset.get_image_shift()
    frameSize = dataset.get_frame_size()
    min, max, step = dataset.get_data_image_mapping()

    imageShift = imageShift[frameIndex]

    frameFlow = h5Data[frameIndex, :, :, :][:, :, [header.index('u'), header.index('v')]]
    frameFlow[:, :, 0] = np.rint((frameFlow[:, :, 0] - imageShift[0]) / step[0])
    frameFlow[:, :, 1] = np.rint((frameFlow[:, :, 1] - imageShift[1]) / step[1])


    # Shift the array to transform to the current frame coordinates.
    cropMin = np.maximum(np.rint(imageShift / step).astype(np.int), 0)
    cropMax = np.minimum(np.rint(frameSize + imageShift / step).astype(np.int), frameSize)
    cropSize = cropMax - cropMin
    plottedFlow = np.zeros(frameFlow.shape)
    plottedFlow[0:cropSize[0], 0:cropSize[1], :] = frameFlow[cropMin[0]:cropMax[0], cropMin[1]:cropMax[1], :]
    plottedFlow = plottedFlow.swapaxes(0, 1)

    X, Y = np.meshgrid(np.arange(0, frameSize[0]), np.arange(0, frameSize[1]))

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)
    ax.quiver(X, Y, plottedFlow[:, :, 0], plottedFlow[:, :, 1], angles='uv', scale_units='xy', scale=1,
              width=0.003, headwidth=2)
    # pdf.savefig(fig)
    # plt.cla()
    plt.show()


def plot_data_mapping(dataset):

    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    mappingText = 'Data to camera image mapping\n'
    mappingText += 'Data size: {} Image size: {}\n'.format(
        list(dataset.get_frame_size()), dataset.get_attr('cameraImageSize'))
    mappingText += 'Physical size: {}\n'.format(dataset.get_attr('physicalFrameSize'))
    mappingText += 'Min: {} Max: {} Step: {}\n'.format(*dataset.get_data_image_mapping())

    ax.text(0.1, 0.1, mappingText)

    return fig

