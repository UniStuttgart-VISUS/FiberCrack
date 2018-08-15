import math
import warnings

import numpy as np
import scipy.ndimage.morphology
import scipy.stats
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.transform
import skimage.util

from FiberCrack.Dataset import Dataset
import FiberCrack.image_processing as image_processing


__all__ = ['append_crack_from_tracking_loss',
           'append_crack_from_unmatched_pixels', 'append_crack_from_variance',
           'append_crack_from_entropy', 'append_crack_from_unmatched_and_entropy',
           'append_reference_frame_crack']


def _binary_erosion_repeated(data, selem, repetitions):
    for i in range(repetitions):
        data = skimage.morphology.binary_erosion(data, selem)
    return data


def _binary_dilation_repeated(data, selem, repetitions):
    for i in range(repetitions):
        data = skimage.morphology.binary_dilation(data, selem)
    return data


def append_crack_from_tracking_loss(dataset: 'Dataset'):
    """
    The most primitive crack estimation (not even close): if the pixel in reference frame
    lost tracking, there's a crack at this pixel in the current frame.
    Used only to estimate the corresponding 'crack area' and compare it against the good results.
    :param dataset:
    :return:
    """

    header = dataset.get_header()

    newColumnIndex = dataset.create_or_get_column('trackingLossCrack')
    for frameIndex in range(0, dataset.get_frame_number()):
        frameData = dataset.h5Data[frameIndex, ...]

        sigma = frameData[:, :, header.index('sigma')]

        # Pixels with sigma = -1 have lost tracking.
        dataset.h5Data[frameIndex, :, :, newColumnIndex] = (sigma < 0)


def append_crack_from_unmatched_pixels(dataset: 'Dataset', dicKernelRadius,
                                       unmatchedPixelsPadding: float,
                                       unmatchedPixelsMorphologyDepth: int,
                                       unmatchedPixelsObjectsThreshold: float,
                                       unmatchedPixelsHolesThreshold: float):
    """
    Detect the crack pixels based on which pixels haven't been 'matched to'
    from the reference frame. These pixels have 'appeared out of nowhere' and
    potentially represent the crack.

    :param dataset:
    :param dicKernelRadius:
    :param unmatchedPixelsPadding:
    :param unmatchedPixelsMorphologyDepth:
    :param unmatchedPixelsObjectsThreshold:
    :param unmatchedPixelsHolesThreshold:
    :return:
    """
    frameWidth, frameHeight = dataset.get_frame_size()
    header = dataset.get_header()

    # Prepare columns for the results.
    index1 = dataset.create_or_get_column('matchedPixelsGaussThres')
    index2 = dataset.create_or_get_column('matchedPixelsObjectsRemoved')
    index3 = dataset.create_or_get_column('matchedPixelsHolesRemoved')
    index4 = dataset.create_or_get_column('matchedPixelsCrack')

    selem = scipy.ndimage.morphology.generate_binary_structure(2, 2)
    for frameIndex in range(0, dataset.get_frame_number()):
        frameData = dataset.h5Data[frameIndex, ...]
        matchedPixels = frameData[:, :, header.index('matched')]

        # Manually remove the unmatched pixels that are pushing into the image from the sides.
        # todo It would be better to remove regions adjacent to borders using morphology.
        if unmatchedPixelsPadding > 0.0:
            paddingWidth = int(frameWidth * unmatchedPixelsPadding)
            paddingHeight = int(frameHeight * unmatchedPixelsPadding)
            matchedPixels[:paddingWidth, :] = True
            matchedPixels[-paddingWidth:, :] = True
            matchedPixels[:, :paddingHeight] = True
            matchedPixels[:, -paddingHeight:] = True

        # Gaussian smoothing.
        matchedPixelsGauss = skimage.filters.gaussian(matchedPixels, 2.0)

        # Binary thresholding.
        thresholdBinary = lambda t: lambda x: 1.0 if x >= t else 0.0
        matchedPixelsGaussThres = np.vectorize(thresholdBinary(0.5))(matchedPixelsGauss)
        # matchedPixelsGaussThres = skimage.morphology.binary_dilation(matchedPixels, selem)

        # Morphological filtering.

        # Suppress warnings from remove_small_objects/holes which occur when there's a single object/hole.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')

            maxObjectSize = int(frameWidth * frameHeight * unmatchedPixelsObjectsThreshold)
            tempResult = skimage.morphology.remove_small_objects(
                matchedPixelsGaussThres.astype(np.bool), min_size=maxObjectSize)

            cropSelector = (slice(int(frameWidth * unmatchedPixelsPadding), int(frameWidth * unmatchedPixelsPadding)),
                            slice(int(frameHeight * unmatchedPixelsPadding), int(frameHeight * unmatchedPixelsPadding)))
            holePixelNumber = np.count_nonzero(tempResult[cropSelector] == False)
            tempResult = skimage.morphology.remove_small_holes(tempResult,
                                                               min_size=holePixelNumber * unmatchedPixelsHolesThreshold)

            tempResult = _binary_erosion_repeated(tempResult, selem, unmatchedPixelsMorphologyDepth)
            tempResult = skimage.morphology.remove_small_objects(tempResult, min_size=maxObjectSize)
            tempResult = _binary_dilation_repeated(tempResult, selem, unmatchedPixelsMorphologyDepth)
            matchedPixelsObjectsRemoved = tempResult.copy()
            tempResult = _binary_dilation_repeated(tempResult, selem, unmatchedPixelsMorphologyDepth)
            tempResult = skimage.morphology.remove_small_holes(tempResult,
                                                               min_size=holePixelNumber * unmatchedPixelsHolesThreshold)
            matchedPixelsHolesRemoved = tempResult.copy()

            # Don't erode back: instead, compensate for the kernel used during DIC.
            currentDilation = unmatchedPixelsMorphologyDepth  # Because we just dilated.
            assert currentDilation <= dicKernelRadius + 1     # Make sure we didn't dilate too far clearing the noise.
            tempResult = _binary_dilation_repeated(tempResult, selem, dicKernelRadius + 1 - currentDilation)

            matchedPixelsCrack = tempResult == 0.0

        # Write the results.
        dataset.h5Data[frameIndex, :, :, index1] = matchedPixelsGaussThres
        dataset.h5Data[frameIndex, :, :, index2] = matchedPixelsObjectsRemoved
        dataset.h5Data[frameIndex, :, :, index3] = matchedPixelsHolesRemoved
        dataset.h5Data[frameIndex, :, :, index4] = matchedPixelsCrack


def append_crack_from_variance(dataset: 'Dataset', textureKernelSize, varianceThreshold=0.003):
    """
    Detect crack pixels based on local variance computed from the current camera frame.
    (Purely image-based technique.)

    :param dataset:
    :param textureKernelSize:
    :param varianceThreshold:
    :return:
    """
    print("Computing cracks from variance.")

    header = dataset.get_header()

    selem = scipy.ndimage.morphology.generate_binary_structure(2, 2)

    index1 = dataset.create_or_get_column('cameraImageVar')
    index2 = dataset.create_or_get_column('cameraImageVarFiltered')

    for frameIndex in range(0, dataset.get_frame_number()):
        frameData = dataset.h5Data[frameIndex, ...]
        cameraImage = frameData[..., header.index('camera')]

        # Compute variance.
        cameraImageVar = image_processing.image_variance_filter(cameraImage, textureKernelSize)

        # Threshold.
        varianceBinary = cameraImageVar < varianceThreshold

        # Clean up.
        varianceFiltered = varianceBinary.copy()
        for i in range(0, math.ceil(textureKernelSize / 2.0)):
            varianceFiltered = skimage.morphology.binary_dilation(varianceFiltered, selem)

        dataset.h5Data[frameIndex, ..., index1] = cameraImageVar
        dataset.h5Data[frameIndex, ..., index2] = varianceFiltered


def append_crack_from_entropy(dataset: 'Dataset', textureKernelSize, entropyThreshold=1.0):
    """
    Detect crack pixels based on local entropy computed from the current camera frame.
    (Purely image-based technique.)
    :param dataset:
    :param textureKernelSize:
    :param entropyThreshold:
    :return:
    """

    # todo a lot of repetition between the variance and the entropy implementations.
    # refactor, if more features are added.
    print("Computing cracks from entropy.")

    header = dataset.get_header()

    selem = scipy.ndimage.morphology.generate_binary_structure(2, 2)

    index1 = dataset.create_or_get_column('cameraImageEntropy')
    index2 = dataset.create_or_get_column('cameraImageEntropyBinary')
    index3 = dataset.create_or_get_column('cameraImageEntropyFiltered')

    for frameIndex in range(0, dataset.get_frame_number()):
        frameData = dataset.h5Data[frameIndex, ...]
        cameraImage = frameData[..., header.index('camera')]

        #  Compute entropy.
        cameraImageEntropy = image_processing.image_entropy_filter(cameraImage, textureKernelSize)

        # Threshold.
        entropyBinary = cameraImageEntropy < entropyThreshold

        # Clean up.
        entropyFiltered = entropyBinary.copy()
        for i in range(0, math.ceil(textureKernelSize / 2.0)):
            entropyFiltered = skimage.morphology.binary_dilation(entropyFiltered, selem)

        dataset.h5Data[frameIndex, ..., index1] = cameraImageEntropy
        dataset.h5Data[frameIndex, ..., index2] = entropyBinary
        dataset.h5Data[frameIndex, ..., index3] = entropyFiltered


def append_crack_from_unmatched_and_entropy(dataset: 'Dataset', textureKernelSize, hybridKernelMultiplier,
                                            entropyThreshold, hybridDilationDepth: int, unmatchedPixelsPadding=0.1):
    """
    Detect crack pixels based on both the local entropy filtering and
    unmatched pixels.
    A hybrid method.

    Narrowing the search region based on unmatched pixels allows to reduce
    the kernel size for the entropy filter and increase the spatial precision.

    :param dataset:
    :param textureKernelSize:
    :param hybridKernelMultiplier:
    :param entropyThreshold:
    :param hybridDilationDepth:
    :param unmatchedPixelsPadding:
    :return:
    """
    header = dataset.get_header()
    selem = scipy.ndimage.morphology.generate_binary_structure(2, 2)

    frameWidth, frameHeight = dataset.get_frame_size()
    # Use a smaller entropy kernel size, since we narrow the search area using unmatched pixels.
    entropyFilterRadius = int(textureKernelSize * hybridKernelMultiplier)

    index1 = dataset.create_or_get_column('hybridUnmatchedDilated')
    index2 = dataset.create_or_get_column('hybridEntropyBinary')
    index3 = dataset.create_or_get_column('hybridCracks')

    for frameIndex in range(0, dataset.get_frame_number()):
        frameData = dataset.h5Data[frameIndex, :, :, :]
        # Fetch the unmatched pixels (where matched pixels == 0).
        unmatchedPixels = frameData[..., header.index('matchedPixelsHolesRemoved')]  == 0

        # Manually remove the unmatched pixels that are pushing into the image from the sides.
        if unmatchedPixelsPadding > 0.0:
            paddingWidth = int(frameWidth * unmatchedPixelsPadding)
            unmatchedPixels[:paddingWidth, :] = False
            unmatchedPixels[-paddingWidth:, :] = False

        # Dilate the unmatched pixels crack, and later use it as a search filter.
        unmatchedPixels = _binary_dilation_repeated(unmatchedPixels, selem, hybridDilationDepth)
        unmatchedDilated = unmatchedPixels.copy()

        # todo reuse entropy code (copy-pasting right now), but note that we might need a different kernel size.
        imageEntropy = image_processing.image_entropy_filter(frameData[..., header.index('camera')], int(entropyFilterRadius))
        entropyBinary = imageEntropy < entropyThreshold  # type: np.ndarray

        entropyFiltered = entropyBinary.copy()
        entropyFiltered = _binary_dilation_repeated(entropyFiltered, selem, int(math.ceil(entropyFilterRadius / 2.0)))

        # Crack = low entropy near unmatched pixels.
        cracks = np.logical_and(unmatchedPixels, entropyFiltered)

        dataset.h5Data[frameIndex, ..., index1] = unmatchedDilated
        dataset.h5Data[frameIndex, ..., index2] = entropyBinary
        dataset.h5Data[frameIndex, ..., index3] = cracks


def append_reference_frame_crack(dataset: 'Dataset', dicKernelRadius, sigmaSkeletonPadding=0.1):
    """
    Compute the crack path in the reference frame based on sigma,
    i.e. based on which pixels have lost tracking and can no longer be found
    in the current frame.

    :param dataset:
    :param dicKernelRadius:
    :return:
    """
    frameSize = dataset.get_frame_size()

    index1 = dataset.create_or_get_column('sigmaFiltered')
    index2 = dataset.create_or_get_column('sigmaSkeleton')
    index3 = dataset.create_or_get_column('sigmaSkeletonPruned')

    for frameIndex in range(0, dataset.get_frame_number()):

        # Get the original sigma plot, with untrackable pixels as 'ones' (cracks).
        binarySigma = dataset.get_column_at_frame(frameIndex, 'sigma') < 0

        if sigmaSkeletonPadding > 0:
            paddingHeight = int(frameSize[1] * sigmaSkeletonPadding)
            binarySigma[:, :paddingHeight] = False
            binarySigma[:, -paddingHeight:] = False

        binarySigmaFiltered = binarySigma.copy()
        maxObjectSize = int(frameSize[0] * frameSize[1] / 1000)

        # Erode the cracks to remove smaller noise.
        selem = scipy.ndimage.morphology.generate_binary_structure(binarySigma.ndim, 2)
        for i in range(0, math.ceil(dicKernelRadius / 2)):
            binarySigmaFiltered = skimage.morphology.binary_erosion(binarySigmaFiltered, selem)

        # Remove tiny disconnected chunks of cracks as unnecessary noise.
        with warnings.catch_warnings():
            warnings.filterwarnings('ignore')
            binarySigmaFiltered = skimage.morphology.remove_small_objects(binarySigmaFiltered, maxObjectSize)

        # Compute the skeleton.
        binarySigmaSkeleton = skimage.morphology.skeletonize(binarySigmaFiltered)

        # Prune the skeleton from small branches.
        binarySigmaSkeletonPruned = image_processing.image_morphology_prune(binarySigmaSkeleton, int(frameSize[0] / 100))

        dataset.h5Data[frameIndex, ..., index1] = binarySigmaFiltered
        dataset.h5Data[frameIndex, ..., index2] = binarySigmaSkeleton
        dataset.h5Data[frameIndex, ..., index3] = binarySigmaSkeletonPruned