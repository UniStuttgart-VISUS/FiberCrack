import os
import os.path as path

from PIL import Image
import numpy as np
import skimage.feature
import skimage.filters
import skimage.measure
import skimage.morphology
import skimage.transform
import skimage.util

from FiberCrack.Dataset import Dataset
from FiberCrack.data_loading import DataImportConfig


__all__ = ['append_camera_image', 'append_matched_pixels', 'zero_pixels_without_tracking',
           'append_data_image_mapping', 'append_physical_frame_size', 'compute_avg_flow']


def append_camera_image(dataset: 'Dataset', dataConfig: 'DataImportConfig'):
    """
    Appends a column containing grayscale data from the camera.
    The data is cropped from the frame according to the mapping between the data and the image.

    :param dataConfig:
    :param dataset:
    :return:
    """

    h5Data, header, frameMap, *r = dataset.unpack_vars()
    imageShift = dataset.get_image_shift()

    min, max, step = dataset.get_data_image_mapping()

    hasCameraImageMask = np.zeros((h5Data.shape[0]))

    columnIndex = dataset.create_or_get_column('camera')
    frameNumber = h5Data.shape[0]
    for f in range(0, frameNumber):
        print("Frame {}/{}".format(f, frameNumber))
        frameIndex = frameMap[f]
        cameraImagePath = path.join(dataConfig.basePath, dataConfig.imageDir,
                                    dataConfig.imageFilenameFormat.format(dataConfig.imageBaseName, frameIndex))
        cameraImageAvailable = os.path.isfile(cameraImagePath)
        if cameraImageAvailable:
            cameraImage = np.array(skimage.util.img_as_float(Image.open(cameraImagePath)))

            # Crop a rectangle from the camera image, accounting for the overall shift of the specimen.
            relMin = np.clip(min + imageShift[f, ...], [0, 0], cameraImage.shape)
            relMax = np.clip(max + imageShift[f, ...], [0, 0], cameraImage.shape)

            size = np.ceil((relMax - relMin) / step).astype(np.int)
            h5Data[f, 0:size[0], 0:size[1], columnIndex] = \
                cameraImage[relMin[1]:relMax[1]:step[1], relMin[0]:relMax[0]:step[0]].transpose()

            dataset.set_attr('cameraImageSize', cameraImage.shape)
            hasCameraImageMask[f] = True

    dataset.create_or_update_metadata_column('hasCameraImage', hasCameraImageMask)

    return dataset


def append_matched_pixels(dataset: 'Dataset', dataConfig: 'DataImportConfig'):
    """
    Computes which pixels have been 'matched to' in each frame
    from the reference frame.
    Also stores the 'back-flow', which maps pixels back into the reference frame.

    Note that the matching is computed w.r.t. to the overall image shift.
    :param dataConfig:
    :param dataset:
    :return:
    """
    h5Data, header, frameMap, *r = dataset.unpack_vars()
    imageShift = dataset.get_image_shift()
    frameSize = dataset.get_frame_size()
    min, max, step = dataset.get_data_image_mapping()

    matchedColumnIndex = dataset.create_or_get_column('matched')
    uBackColumnIndex = dataset.create_or_get_column('u_back')
    vBackColumnIndex = dataset.create_or_get_column('v_back')

    frameNumber = h5Data.shape[0]
    for f in range(0, frameNumber):
        matchedPixels = np.zeros(frameSize)
        backwardFlow = np.zeros(frameSize + (2,))
        print("Frame {}/{}".format(f, frameNumber))
        frameData = h5Data[f, :, :, :]
        frameFlow = frameData[:, :, [header.index('u'), header.index('v')]]
        frameMask = frameData[:, :, header.index('sigma')]
        for x in range(0, frameSize[0]):
            for y in range(0, frameSize[1]):
                # Don't consider pixels that lost tracking.
                if frameMask[x, y] < 0:
                    continue

                newX = x + int(round((frameFlow[x, y, 0] - imageShift[f, 0]) / step[0]))
                newY = y + int(round((frameFlow[x, y, 1] - imageShift[f, 1]) / step[1]))
                if (0 < newX < frameSize[0]) and (0 < newY < frameSize[1]):
                    matchedPixels[newX, newY] = 1.0
                    backwardFlow[newX, newY, :] = (imageShift[f, :] - frameFlow[x, y, :]) / step

        h5Data[f, ..., matchedColumnIndex] = matchedPixels
        h5Data[f, ..., [uBackColumnIndex, vBackColumnIndex]] = backwardFlow


def zero_pixels_without_tracking(dataset: 'Dataset'):
    """
    Remove the flow data for pixels that have lost tracking,
    sicne it's unreliable.
    :param dataset:
    :return:
    """
    h5Data, header, frameMap, *r = dataset.unpack_vars()
    # frameSize = dataset.get_frame_size()
    for f in range(0, h5Data.shape[0]):
        data = h5Data[f, ...]
        filter = data[..., header.index('sigma')] >= 0
        # flowIndices = [header.index('u'), header.index('v')]
        data[..., header.index('u')] *= filter
        data[..., header.index('v')] *= filter
        h5Data[f, ...] = data
        # todo is it faster to use an intermediate buffer? or should I write to hdf5 directly?


def append_data_image_mapping(dataset: 'Dataset'):
    """
    Fetch the min/max pixel coordinates of the data, in original image space (2k*2k image)
    Important, since the data only covers every ~fifth pixel of some cropped subimage of the camera image.

    :param dataset:
    :return: (min, max, step)
    """
    h5Data, header, *r = dataset.unpack_vars()

    minX = int(h5Data[0, 0, 0, header.index('x')])
    maxX = int(h5Data[0, -1, 0, header.index('x')])
    minY = int(h5Data[0, 0, 0, header.index('y')])
    maxY = int(h5Data[0, 0, -1, header.index('y')])
    stepX = round((maxX - minX) / h5Data.shape[1])
    stepY = round((maxY - minY) / h5Data.shape[2])

    dataset.set_attr('mappingMin', np.array([minX, minY]))
    dataset.set_attr('mappingMax', np.array([maxX, maxY]))
    dataset.set_attr('mappingStep', np.array([stepX, stepY]))


def append_physical_frame_size(dataset: 'Dataset'):
    """
    Find out the physical size of the specimen (more precisely, the size
    of the region of interest) and store it in the data.
    :param dataset:
    :return:
    """
    h5Data, header, *r = dataset.unpack_vars()

    # We cannot simply look at the corner pixels, since sometimes
    # the physical coordinates are reported incorrectly (at least near the edges).
    # Instead, find literally the minimum and maximum coordinates present in the data.
    xColumn = h5Data[0, :, :, header.index('X')]
    yColumn = h5Data[0, :, :, header.index('Y')]
    minX = np.min(xColumn)
    maxX = np.max(xColumn)
    minY = np.min(yColumn)
    maxY = np.max(yColumn)

    physicalDomain = np.array([[minX, minY], [maxX, maxY]])

    physicalSize = np.abs(np.array([maxX - minX, maxY - minY]))
    print('Computed physical size: {}'.format(physicalSize))
    print('Physical domain: {}'.format(physicalDomain))
    dataset.set_attr('physicalFrameSize', physicalSize)


def compute_avg_flow(dataset: 'Dataset'):
    """
    Sample the flow/shift (u,v) at a few points to determine the average shift of each frame
    relative to the base frame.

    :param dataset:
    :return:
    """

    h5Data, header, *r = dataset.unpack_vars()

    # Select points which will be sampled to determine the overall shift relative to the ref. frame
    # (This is used to align the camera image with the reference frame (i.e. data space)
    sampleX = np.linspace(0 + 20, h5Data.shape[1] - 20, 4)
    sampleY = np.linspace(0 + 20, h5Data.shape[2] - 20, 4)
    samplePointsX, samplePointsY = np.array(np.meshgrid(sampleX, sampleY)).astype(np.int)
    # Convert to an array of 2d points.
    samplePoints = np.concatenate((samplePointsX[:, :, np.newaxis], samplePointsY[:, :, np.newaxis]), 2).reshape(-1, 2)

    avgFlow = np.empty((h5Data.shape[0], 2))
    for f in range(0, h5Data.shape[0]):
        frameData = h5Data[f, ...]
        samples = np.array([frameData[tuple(p)] for p in samplePoints])
        uvSamples = samples[:, [header.index('u'), header.index('v'), header.index('sigma')]]
        uvSamples = uvSamples[uvSamples[:, 2] >= 0, :]  # Filter out points that lost tracking
        avgFlow[f, :] = np.mean(uvSamples[:, 0:2], axis=0).astype(np.int) if uvSamples.shape[0] > 0 else np.array(
            [0, 0])

    return avgFlow

