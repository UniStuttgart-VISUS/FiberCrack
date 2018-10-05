import numpy as np
import math

import scipy.signal


def masked_gaussian_filter(data, sourceMask, targetMask, sigma):
    """
    Applies standard Gaussian blur on the data.
    The source mask controls which pixels are sampled when computing the blurred pixel value.
    The target mask control for which pixels do we compute the blur.
    This can be used to 'fill-in' pixels in an image without affecting existing data.

    :param data:
    :param sourceMask:
    :param targetMask:
    :param sigma:
    :return:
    """
    size = data.shape

    assert(data.ndim == 2)
    assert(sourceMask.shape == size)
    assert(targetMask.shape == size)

    def round_up_to_odd(value):
        rounded = math.ceil(value)
        return rounded if rounded % 2 == 1 else rounded + 1

    kernelRadius = int(math.ceil(sigma * 3.0))
    kernelWidth = 2 * kernelRadius + 1

    # Precompute kernel values. No scaling constant, since we normalize anyway.
    # Can't precompute the normalization, since we cherry pick values on the fly.
    kernel = np.zeros(kernelWidth)
    kernel[kernelRadius] = 1.0   # middle
    sigmaSqr = sigma ** 2
    for i in range(1, kernelRadius + 1):
        w = math.exp(-0.5 * float(i ** 2) / sigmaSqr)
        kernel[kernelRadius + i] = w
        kernel[kernelRadius - i] = w

    def do_pass(source, target, sourceMask, targetMask, isHorizontal):
        for x in range(0, size[0]):
            for y in range(0, size[1]):
                if not targetMask[x, y]:
                    continue

                minInput = [math.ceil(x - kernelRadius),
                            math.ceil(y - kernelRadius)]
                maxInput = [math.floor(x + kernelRadius),
                            math.floor(y + kernelRadius)]

                if isHorizontal:
                    minInput[1] = maxInput[1] = y
                else:
                    minInput[0] = maxInput[0] = x


                result = 0.0
                weightSum = 0.0
                for xi in range(minInput[0], maxInput[0] + 1):
                    for yi in range(minInput[1], maxInput[1] + 1):
                        isOutbound = xi < 0 or yi < 0 or \
                                     xi > size[0] - 1 or yi > size[1] - 1
                        if isOutbound:
                            continue
                        if not sourceMask[xi, yi]:
                            continue

                        kernelIndexShift = xi - x if isHorizontal else yi - y

                        weight = kernel[kernelRadius + kernelIndexShift]
                        result += source[xi, yi] * weight
                        weightSum += weight

                if weightSum > 0.0:
                    result /= weightSum
                    target[x, y] = result

    firstPass = data.copy()
    do_pass(data, firstPass, sourceMask, targetMask, True)
    secondPass = firstPass.copy()
    do_pass(firstPass, secondPass, np.ones(size), targetMask, False)

    return secondPass


def downsample_with_smoothing_2d(data, factor) -> np.ndarray:
    assert(data.ndim == 2)
    assert(isinstance(factor, int))

    def gaussian(dist, std):
        return math.exp(- dist ** 2 / (2 * std ** 2))

    kernelRadius = int(factor) - 1
    kernelSize = kernelRadius * 2 + 1
    gaussianKernel = np.zeros(kernelSize)
    for i, x in enumerate(range(-kernelRadius, kernelRadius + 1)):
        gaussianKernel[i] = gaussian(x, kernelRadius / 3.0)

    gaussianKernel /= np.sum(gaussianKernel)

    kernels = []
    for dim in range(data.ndim):
        # kernelShape = tuple((min(kernelSize, data.shape[dim]) if dim == i else 1 for i in range(data.ndim)))
        kernelShape = tuple((kernelSize if dim == i else 1 for i in range(data.ndim)))
        kernels.append(gaussianKernel.reshape(kernelShape))

    result = data.astype(np.float).copy()  # Doesn't work correctly in-place.
    for kernel in kernels:
        result = scipy.signal.convolve(result, kernel, 'same')

    result = result[factor//2-1::factor, factor//2-1::factor]

    return result.astype(data.dtype)
