import os
from typing import Callable, Any, List, Tuple, Union

import numpy as np
import matplotlib.pyplot as plt
import cairo

import PythonExtras.volume_tools as volume_tools


def render_2d_volume_files_to_rgba(inputDatPath: str,
                                   transferFunction: Union[Callable[[float], Tuple], str],
                                   timeSlice: slice = None,
                                   timesteps: List[int] = None,
                                   printFn=None) -> np.ndarray:

    data = volume_tools.load_volume_data_from_dat(inputDatPath)
    if timeSlice:
        data = data[timeSlice, ...]
    elif timesteps:
        data = data[timesteps, ...]

    return render_2d_volume_to_rgba(data, transferFunction, printFn)


def render_2d_volume_to_rgba(data: np.ndarray,
                             transferFunction: Union[Callable[[float], Tuple], str],
                             printFn=None) -> np.ndarray:
    """

    :param data:
    :param transferFunction: TF should return values in range [0, 1] !!!
    :param printFn:
    :return:
    """

    if type(transferFunction) == str:
        transferFunction = plt.get_cmap(transferFunction)

    # Measure the dimensionality of the output.
    channelNumber = len(transferFunction(0.5))

    tfVector = np.vectorize(transferFunction)

    def render_frame(frameData):
        tupleOfArrays = tfVector(frameData)
        arrayOfChannels = np.concatenate(tuple(t[..., np.newaxis] for t in tupleOfArrays), axis=2)
        return 255 * arrayOfChannels

    if len(data.shape) == 4:
        # Render a temporal volume frame-by-frame.
        assert (data.shape[1] == 1)

        frameNumber = data.shape[0]
        images = np.empty((frameNumber, *data.shape[2:4], channelNumber), dtype=np.uint8)
        for f in range(frameNumber):
            if printFn:
                printFn("Rendering frame {}/{}.".format(f + 1, frameNumber))
            images[f, :, :, :] = render_frame(data[f, 0, :, :])

        return images

    else:
        # Render a static volume.
        if len(data.shape) == 3:
            assert (data.shape[0] == 1)
            data = data[0, ...]

        assert len(data.shape) == 2

        return np.clip(render_frame(data), 0, 255).astype(np.uint8)


def image_rgba_uint8_to_cairo(imageData: np.ndarray):

    assert imageData.dtype == np.uint8
    assert len(imageData.shape) == 3  # height x width x channels
    assert imageData.shape[2] == 4    # RGBA

    # Flip the Y axis (first axis in C-order) to point bottom-up.
    imageData = np.flip(imageData, axis=0)

    # Convert RGBA to ARGB and reinterpret as a uint32.
    # (uint32 is little-endian, so we actually need BGRA)
    imageBGR = np.flip(imageData[:, :, :3], axis=2)
    imageA = imageData[:, :, 3][..., np.newaxis]
    imageDataConverted = np.concatenate((imageBGR, imageA), axis=2).view('<u4')
    return cairo.ImageSurface.create_for_data(imageDataConverted, cairo.FORMAT_ARGB32,
                                              imageData.shape[1], imageData.shape[0])