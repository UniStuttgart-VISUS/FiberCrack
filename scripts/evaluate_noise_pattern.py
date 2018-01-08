import os
import numpy as np
import skimage.util
import skimage.filters
import scipy.ndimage
import scipy.stats
from PIL import Image


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

def main():

    imagePaths = ["M:\\Experiments\\PTFE-Epoxy\\raw_images\\Spec048-0000_0.tif",
                  "M:\\Experiments\\Steel-ModifiedEpoxy\\raw_images\\spec010-0000_0.tif",
                  "M:\\Experiments\\Steel-Epoxy\\raw_images\\Spec054-0000_0.tif"]
    
    kernelSize = 40

    for cameraImagePath in imagePaths:
        cameraImageAvailable = os.path.isfile(cameraImagePath)
        if cameraImageAvailable:
            print("Processing '{}'".format(cameraImagePath))

            cameraImage = np.array(skimage.util.img_as_float(Image.open(cameraImagePath)))
            if cameraImage.ndim == 3:  # If multichannel image.
                cameraImage = cameraImage[..., 0]  # Use only the first channel.

            cameraImageEntropy = image_entropy_filter(cameraImage, kernelSize)
            cameraImageVariance = image_variance_filter(cameraImage, kernelSize)

            entropyMean = np.mean(cameraImageEntropy)
            entropyMedian = np.median(cameraImageEntropy)

            varianceMean = np.mean(cameraImageVariance)
            varianceMedian = np.median(cameraImageVariance)
            
            print("Entropy mean: {}".format(entropyMean))
            print("Entropy median: {}".format(entropyMedian))
            print("Variance mean: {}".format(varianceMean))
            print("Variance median: {}".format(varianceMedian))
        else:
            print("Can't open file '{}'".format(cameraImagePath))

main()