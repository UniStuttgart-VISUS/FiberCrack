import argparse
import hashlib
import inspect
import os
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.ndimage.morphology
import scipy.stats
from matplotlib.backends.backend_pdf import PdfPages

import FiberCrack.crack_detection as crack_detection
import FiberCrack.data_augmentation as data_augmentation
import FiberCrack.data_loading as data_loading
import FiberCrack.plotting as plotting
from PythonExtras.numpy_extras import slice_along_axis
from PythonExtras.volume_tools import write_volume_to_datraw

np.random.seed(13)  # Fix the seed for reproducibility.
# import tensorflow as tf
# tf.set_random_seed(13)

############### Configuration ###############

dataConfig = data_loading.DataImportConfig()
dataConfig.preloadedDataDir = 'C:/preloaded_data'
dataConfig.preloadedDataFilename = None  # By default, decide automatically.
dataConfig.dataFormat = 'csv'
dataConfig.imageFilenameFormat = '{}-{:04d}_0.tif'

dataConfig.reloadOriginalData = False

maxFrames = 99999
recomputeResults = False

outDir = 'T:/projects/SimtechOne/out/fiber'

dicKernelSize = 55

globalParams = {
    'textureKernelMultiplier': 1.0,
    'entropyThreshold': 1.0,
    'varianceThreshold': 0.003,
    'unmatchedPixelsPadding': 0.0,
    'unmatchedAndEntropyKernelMultiplier': 0.5,
    'exportedVolumeTimestepWidth': 3,
    'exportedVolumeSkippedFrames': 5
}

# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Steel-Epoxy'
# metadataFilename = 'Steel-Epoxy.csv'
# dataDir = 'data_export_tstep3'
# imageDir = 'raw_images'
# imageBaseName = 'Spec054'
# dicKernelSize = 85

# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Steel-Epoxy'
# metadataFilename = 'Steel-Epoxy.csv'
# dataDir = 'data_export'
# imageDir = 'raw_images'
# imageBaseName = 'Spec054'
# dicKernelSize = 85
# preloadedDataFilename = 'Steel-Epoxy-low-t-res.hdf5'

# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Steel-ModifiedEpoxy'
# metadataFilename = 'Steel-ModifiedEpoxy.csv'
# dataDir = 'data_export'
# imageDir = 'raw_images'
# imageBaseName = 'Spec010'
# dicKernelSize = 55

dataConfig.basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/PTFE-Epoxy'
dataConfig.metadataFilename = 'PTFE-Epoxy.csv'
dataConfig.dataDir = 'data_export'
# dataConfig.dataDir = 'data_export_fine'
dataConfig.imageDir = 'raw_images'
dataConfig.imageBaseName = 'Spec048'
dataConfig.dicKernelSize = 81
dataConfig.crackAreaGroundTruthPath = 'spec_048_area.csv'

# # PTFE-Epoxy with fine spatial and temporal resolutions, mid-experiment.
# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Spec48'
# metadataFilename = '../PTFE-Epoxy/PTFE-Epoxy.csv'
# dataDir = 'data_grid_sparse_filtered'
# imageDir = '../PTFE-Epoxy/raw_images'
# imageBaseName = 'Spec048'
# preloadedDataFilename = 'PTFE-Epoxy-fine-mid.hdf5'
# dicKernelSize = 81

# dataFormat = 'tiff'
# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/micro_epoxy-hole'
# dataDir = ''
# imageDir = ''
# imageBaseName = 'test'
# imageFilenameFormat = '{}_{:03d}.bmp'
# metadataFilename = ''
# preloadedDataFilename = 'micro_epoxy-hole'


#############################################


def load_data():
    if dataConfig.dataFormat == 'csv':
        return data_loading.load_csv_data(dataConfig)
    elif dataConfig.dataFormat == 'tiff':
        return data_loading.load_tiff_data(dataConfig)
    else:
        raise ValueError("Unknown data format: {}".format(dataConfig.dataFormat))


def augment_data(dataset: 'Dataset'):
    """
    Extends the raw data with some pre-processing.
    Doesn't compute any 'results', but rather information that can help compute the results.

    :param dataset:
    :return:
    """
    header = dataset.get_header()
    metaheader = dataset.get_metaheader()

    if 'imageShiftX' in metaheader and 'camera' in header and 'matched' in header:
        print("Data already augmented, skipping.")
        return dataset

    # For now we require that all the data is present, or none of it.
    assert('imageShiftX' not in metaheader)
    assert('camera' not in header)
    assert('matched' not in header)

    # Add the data to image mapping to the dataset.
    data_augmentation.append_data_image_mapping(dataset)

    # Add the physical dimensions of the data (in millimeters).
    data_augmentation.append_physical_frame_size(dataset)

    # Add the image shift to the metadata.
    imageShift = data_augmentation.compute_avg_flow(dataset)

    dataset.create_or_update_metadata_column('imageShiftX', imageShift[..., 0])
    dataset.create_or_update_metadata_column('imageShiftY', imageShift[..., 1])

    print("Adding the camera image...")
    data_augmentation.append_camera_image(dataset, dataConfig)
    print("Adding the matched pixels...")
    data_augmentation.append_matched_pixels(dataset, dataConfig)

    print("Zeroing the pixels that lost tracking.")
    data_augmentation.zero_pixels_without_tracking(dataset)

    return dataset


def apply_function_if_code_changed(dataset: 'Dataset', function):
    """
    Calls a function that computes and writes data to the dataset.
    Stores the hash of the function's source code as metadata.
    If the function has not changed, it isn't applied to the data.

    :param dataset:
    :param function:
    :return:
    """
    # Get a string containing the full function source.
    sourceLines = inspect.getsourcelines(function)
    functionSource = ''.join(sourceLines[0])
    functionName = function.__name__

    callSignature = inspect.signature(function)
    callArguments = {}
    for callParameter in callSignature.parameters:
        if callParameter in globalParams:
            callArguments[callParameter] = globalParams[callParameter]

    callArgumentsString = ''.join([key + str(callArguments[key]) for key in sorted(callArguments)])

    attrName = '_functionHash_' + functionName
    currentHash = hashlib.sha1((functionSource + callArgumentsString).encode('utf-8')).hexdigest()

    if 'cameraImageVar' in dataset.get_header() and not recomputeResults:
        oldHash = dataset.get_attr(attrName) if dataset.has_attr(attrName) else None

        if currentHash == oldHash:
            print("Function {} has not changed, skipping.".format(functionName))
            return

    print("Applying function {} to the dataset.".format(functionName))

    callArguments['dataset'] = dataset
    function(**callArguments)

    dataset.set_attr(attrName, currentHash)


def compute_and_append_results(dataset: 'Dataset'):
    # Compute derived parameters.
    mappingMin, mappingMax, mappingStep = dataset.get_data_image_mapping()
    dicKernelRadius = int((dicKernelSize - 1) / 2 / mappingStep[0])
    globalParams['dicKernelRadius'] = dicKernelRadius
    globalParams['textureKernelSize'] = int(dicKernelRadius * globalParams['textureKernelMultiplier'])

    apply_function_if_code_changed(dataset, crack_detection.append_crack_from_unmatched_pixels)

    apply_function_if_code_changed(dataset, crack_detection.append_crack_from_variance)
    apply_function_if_code_changed(dataset, crack_detection.append_crack_from_entropy)

    apply_function_if_code_changed(dataset, crack_detection.append_crack_from_unmatched_and_entropy)
    apply_function_if_code_changed(dataset, crack_detection.append_reference_frame_crack)


def plot_data(dataset: 'Dataset'):
    h5Data, header, frameMap, *r = dataset.unpack_vars()

    # Prepare for plotting
    pdfPath = os.path.join(outDir, 'fiber-crack.pdf')
    print("Plotting to {}".format(pdfPath))
    pdf = PdfPages(pdfPath)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    # Prepare a figure with subfigures.
    fig = plt.figure()
    axes = []
    for f in range(0, 20):
        axes.append(fig.add_subplot(4, 5, f + 1))
        axes[f].axis('off')

    fig.subplots_adjust(hspace=0.025, wspace=0.025)

    # Draw the frame plots.
    for f in range(0, dataset.get_frame_number()):
        timeStart = time.time()
        frameIndex = frameMap[f]
        print("Plotting frame {}".format(frameIndex))
        fig.suptitle("Frame {}".format(frameIndex))

        frameData = h5Data[f, :, :, :]

        plotting.plot_original_data_for_frame(axes[0:5], frameData, header)
        plotting.plot_unmatched_cracks_for_frame(axes[5:10], frameData, header)
        plotting.plot_image_cracks_for_frame(axes[10:15], frameData, header)
        plotting.plot_reference_crack_for_frame(axes[15:18], frameData, header)
        plotting.plot_feature_histograms_for_frame(axes[18:20], frameData, header)

        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        for a in axes:
            a.clear()
            a.axis('off')

        print("Rendered in {:.2f} s.".format(time.time() - timeStart))

    # Crack area figure.
    fig = plotting.plot_crack_area_chart(dataset, os.path.join(dataConfig.basePath, dataConfig.crackAreaGroundTruthPath))
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    # Print the data-to-camera mapping.
    fig = plotting.plot_data_mapping(dataset)
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    pdf.close()


def export_crack_volume(dataset: 'Dataset'):
    """
    Build a volume by concatenating crack areas from each frame,
    save to the disk in datraw format.
    :param dataset:
    :return:
    """
    frameSize = dataset.get_frame_size()

    framesToSkip = globalParams['exportedVolumeSkippedFrames']
    frameWidth = globalParams['exportedVolumeTimestepWidth']
    frameNumber = dataset.get_frame_number() - framesToSkip
    # The volume should be exported in Z,Y,X C-order.
    volume = np.empty((frameNumber * frameWidth, frameSize[1], frameSize[0]), dtype=np.uint8)

    for f in range(0, frameNumber):
        crackArea = dataset.get_column_at_frame(f, 'cracksFromUnmatchedAndEntropy')
        crackAreaUint8 = np.zeros_like(crackArea, dtype=np.uint8)
        crackAreaUint8[crackArea == 1.0] = 255

        volumeSlabSelector = slice_along_axis(slice(f * frameWidth, f * frameWidth + frameWidth), 0, volume.ndim)
        volume[volumeSlabSelector] = crackAreaUint8.transpose()

    volume = scipy.ndimage.filters.gaussian_filter(volume, 0.5)

    write_volume_to_datraw(volume, os.path.join(outDir, 'crack-volume.raw'))


def main():

    # Parse the arguments.
    parser = argparse.ArgumentParser('Fiber crack.')
    parser.add_argument('-c', '--command', default='plot', choices=['plot', 'export-crack-volume'])
    args = parser.parse_args()

    timeStart = time.time()
    print("Loading the data.")
    dataset = load_data()
    print("Data loaded in {:.3f} s. Shape: {} Columns: {}".format(time.time() - timeStart, dataset.h5Data.shape, dataset.get_header()))

    timeStart = time.time()
    print("Augmenting the data.")
    augment_data(dataset)
    print("Data augmented in {:.3f} s.".format(time.time() - timeStart))

    timeStart = time.time()
    compute_and_append_results(dataset)
    print("Results computed and appended in {:.3f} s.".format(time.time() - timeStart))

    timeStart = time.time()
    print("Executing command: {}".format(args.command))
    commandMap = {
        'plot': lambda: plot_data(dataset),
        'export-crack-volume': lambda: export_crack_volume(dataset),
    }
    commandMap[args.command]()
    print("Command executed in {:.3f} s.".format(time.time() - timeStart))

    # timeStart = time.time()
    # print("Making a prediction.")
    # predict(dataset: 'Dataset')
    # print("Prediction finished in {:.3f} s.".format(time.time() - timeStart))

    # https://github.com/tensorflow/tensorflow/issues/3388
    # keras.backend.clear_session()


main()
