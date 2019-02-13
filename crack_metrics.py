import typing
import numpy as np

import Dataset as Dataset


__all__ = ['append_estimated_crack_area']


def upper_first(str: str) -> str:
    return str[0].upper() + str[1:]


def append_estimated_crack_area(dataset: 'Dataset'):
    """
    Computes the area of the crack extracted with each available method.

    :param dataset:
    :return:
    """

    print("Computing estimated crack areas...")

    # Original feature name -> clean area result name
    crackFeatureNames = {
        'trackingLossCrack': 'trackingLoss',
        'matchedPixelsCrack': 'unmatchedPixels',
        'cameraImageVarFiltered': 'variance',
        'cameraImageEntropyFiltered': 'entropy',
        'hybridCracks': 'hybrid',
        'crackPredictionBinary': 'predictionSimple',
        'crackPredictionSpatialBinary': 'predictionSpatial',

        'crackGroundTruth': 'groundTruth',
    }

    physicalFrameSize = dataset.get_attr('physicalFrameSize')
    physicalFrameArea = physicalFrameSize[0] * physicalFrameSize[1]
    frameNumber = dataset.get_frame_number()
    frameSize = dataset.get_frame_size()
    frameArea = frameSize[0] * frameSize[1]

    for featureName in crackFeatureNames:
        areaFeatureName = crackFeatureNames[featureName]

        # Compute area in each frame.
        crackAreaData = np.zeros((frameNumber))
        if featureName in dataset.get_header():  # Skip, if feature is not there.
            for f in range(0, frameNumber):
                crack = dataset.get_column_at_frame(f, featureName)
                crackAreaData[f] = np.count_nonzero(crack)

        # Output.
        fullAreaFeatureName = 'crackArea' + upper_first(areaFeatureName)
        dataset.create_or_update_metadata_column(fullAreaFeatureName, crackAreaData)
        dataset.create_or_update_metadata_column(fullAreaFeatureName + 'Physical',
                                                 crackAreaData / frameArea * physicalFrameArea)

    # Store column names for easy access.
    fullNames = ['crackArea' + upper_first(name) for name in crackFeatureNames.values()]
    dataset.set_str_array_attr('crackAreaNames', fullNames)
    dataset.set_str_array_attr('crackAreaNamesShort', crackFeatureNames.values())