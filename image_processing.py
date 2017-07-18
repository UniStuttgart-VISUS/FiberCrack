import numpy as np

import scipy.ndimage.morphology
import scipy.stats
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.transform
import skimage.util


__all__ = ['image_morphology_prune', 'image_variance_filter', 'image_entropy_filter']


def image_morphology_prune(data, iter):
    """
    Apply morphological pruning, removing small branching out 'tails'
    from a 'skeleton'.
    Reference: http://homepages.inf.ed.ac.uk/rbf/HIPR2/thin.htm

    :param data:
    :param iter:
    :return:
    """

    # The structuring element for pruning.
    # Here '2' means empty cell (match anything).
    selems = [
        np.array([[0, 0, 0],
                  [0, 1, 0],
                  [0, 2, 2],])
    ]

    # Another basic selem is a mirrored version of the first one.
    selems.append(np.flip(selems[0], axis=0))
    # Append all rotations of the basic two selems.
    for i in range(0, 3):
        selems.append(np.rot90(selems[2 * i]))
        selems.append(np.rot90(selems[2 * i + 1]))

    output = data.copy()
    for i in range(0, iter):
        for selem in selems:
            removedPixels = scipy.ndimage.morphology.binary_hit_or_miss(output, selem == 1, selem == 0)
            output = np.logical_xor(output, removedPixels)

    return output


def image_variance_filter(data, windowRadius):
    """
    Apply a local variance filter to an image.
    :param data:
    :param windowRadius:
    :return:
    """
    windowLength = windowRadius * 2 + 1
    windowShape = (windowLength, windowLength)

    mean = scipy.ndimage.uniform_filter(data, windowShape)
    meanOfSquare = scipy.ndimage.uniform_filter(data ** 2, windowShape)
    return meanOfSquare - mean ** 2


def image_entropy_filter(data, windowRadius):
    """
    Apply a local entropy filter to an image.
    (Used as a detector of distribution 'uniformity'.)
    :param data:
    :param windowRadius:
    :return:
    """
    dataUint = (data * 16).astype(np.uint8)
    windowLength = windowRadius * 2 + 1
    windowMask = np.ones((windowLength, windowLength), dtype=np.bool)
    # windowMask = skimage.morphology.disk(windowRadius)

    # Important: data range and the number of histogram bins must be equal (skimage expects it so).
    histograms = skimage.filters.rank.windowed_histogram(dataUint, selem=windowMask, n_bins=16)

    return np.apply_along_axis(scipy.stats.entropy, axis=2, arr=histograms)