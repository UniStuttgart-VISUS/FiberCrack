import argparse
import hashlib
import inspect
import os
import time
from typing import Callable, List

import matplotlib.pyplot as plt
import matplotlib as mpl
import numpy as np
import h5py
import scipy.ndimage.morphology
import scipy.stats
from matplotlib.backends.backend_pdf import PdfPages

import FiberCrack.crack_detection as crack_detection
import FiberCrack.crack_prediction as crack_prediction
import FiberCrack.crack_metrics as crack_metrics
import FiberCrack.data_augmentation as data_augmentation
import FiberCrack.data_loading as data_loading
import FiberCrack.plotting as plotting
from FiberCrack.Dataset import Dataset

import FiberCrack.image_processing as image_processing

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
    'unmatchedPixelsPadding': 0.1,
    'sigmaSkeletonPadding': 0.1,
    'unmatchedAndEntropyKernelMultiplier': 0.5,
    'exportedVolumeTimestepWidth': 3,
    'exportedVolumeGradientWidth': 3,
    'exportedVolumeSkippedFrames': 5,

    'allTextureKernelMultipliers': [2.0, 1.5, 1.0, 0.5, 0.25],
    'textureFilters': ['entropy', 'variance']
}

# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Steel-Epoxy'
# metadataFilename = 'Steel-Epoxy.csv'
# dataDir = 'data_export_tstep3'
# imageDir = 'raw_images'
# imageBaseName = 'Spec054'
# dicKernelSize = 85

# Steel-Epoxy dataset. We are comparing crack progression between this one and PTFE-Epoxy.
# dataConfig.basePath = '//visus/visusstore/share/Mehr Daten/Rissausbreitung/Montreal/Experiments/Steel-Epoxy'
# dataConfig.metadataFilename = 'Steel-Epoxy.csv'
# dataConfig.dataDir = 'data_export'
# dataConfig.imageDir = 'raw_images'
# dataConfig.imageBaseName = 'Spec054'
# dataConfig.dicKernelSize = 85
# dataConfig.preloadedDataFilename = 'Steel-Epoxy-low-t-res.hdf5'

# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Steel-ModifiedEpoxy'
# metadataFilename = 'Steel-ModifiedEpoxy.csv'
# dataDir = 'data_export'
# imageDir = 'raw_images'
# imageBaseName = 'Spec010'
# dicKernelSize = 55

# The cleanest dataset: PTFE with epoxy.
dataConfig.basePath = '//visus/visusstore/share/Mehr Daten/Rissausbreitung/Montreal/Experiments/PTFE-Epoxy'
dataConfig.metadataFilename = 'PTFE-Epoxy.csv'
dataConfig.dataDir = 'data_export'
# dataConfig.dataDir = 'data_export_fine'
dataConfig.imageDir = 'raw_images'
dataConfig.groundTruthDir = 'ground_truth'
dataConfig.imageBaseName = 'Spec048'
dataConfig.dicKernelSize = 81
dataConfig.crackAreaGroundTruthPath = 'spec_048_area.csv'

# Older, different experiments.
# dataConfig.basePath = '//visus/visusstore/share/Mehr Daten/Rissausbreitung/Montreal/Experiments/20151125-Spec012'
# dataConfig.metadataFilename = '20152411-Spec012.csv'
# dataConfig.dataDir = 'export'
# # dataConfig.dataDir = 'data_export_fine'
# dataConfig.imageDir = ''
# dataConfig.imageBaseName = '20152411-Spec012'
# dataConfig.dicKernelSize = 81
# dataConfig.crackAreaGroundTruthPath = 'spec_048_area.csv'

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

    print("Adding the camera images...")
    data_augmentation.append_camera_image(dataset, dataConfig)
    print("Adding the crack ground truth images...")
    data_augmentation.append_ground_truth_image(dataset, dataConfig)
    print("Adding the matched pixels...")
    data_augmentation.append_matched_pixels(dataset, dataConfig)

    print("Adding crack area ground truth...")
    data_augmentation.append_crack_area_ground_truth(dataset, dataConfig)

    print("Zeroing the pixels that lost tracking.")
    data_augmentation.zero_pixels_without_tracking(dataset)

    return dataset


def apply_function_if_code_changed(dataset: 'Dataset', function: Callable[..., None]):
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
    textureKernelMultipliers = globalParams['allTextureKernelMultipliers']
    globalParams['dicKernelRadius'] = dicKernelRadius
    globalParams['textureKernelSize'] = int(dicKernelRadius * globalParams['textureKernelMultiplier'])
    globalParams['allTextureKernelSizes'] = [int(dicKernelRadius * mult) for mult in textureKernelMultipliers]

    apply_function_if_code_changed(dataset, data_augmentation.append_texture_features)

    apply_function_if_code_changed(dataset, crack_detection.append_crack_from_unmatched_pixels)

    apply_function_if_code_changed(dataset, crack_detection.append_crack_from_variance)
    apply_function_if_code_changed(dataset, crack_detection.append_crack_from_entropy)

    apply_function_if_code_changed(dataset, crack_detection.append_crack_from_unmatched_and_entropy)
    apply_function_if_code_changed(dataset, crack_detection.append_reference_frame_crack)

    apply_function_if_code_changed(dataset, crack_prediction.append_crack_prediction_simple)
    apply_function_if_code_changed(dataset, crack_prediction.append_crack_prediction_spatial)

    # todo this runs always, because there are too many dependencies.
    crack_metrics.append_estimated_crack_area(dataset)


def plot_frame_data_figures(dataset: 'Dataset', targetFrame=None):

    figureNumber = 19
    figures = [plt.figure(dpi=300) for i in range(figureNumber)]
    axes = [fig.add_subplot(1, 1, 1) for fig in figures]

    for frame in range(dataset.get_frame_number()):

        # If the target frame is specified, skip the other frames.
        if targetFrame is not None and dataset.get_frame_map()[frame] != targetFrame:
            continue

        frameData = dataset.h5Data[frame, ...]
        frameLabel = dataset.get_frame_map()[frame]

        # Plot the data.
        labels1 = plotting.plot_original_data_for_frame(axes[0:5], frameData, dataset.get_header())
        labels2 = plotting.plot_unmatched_cracks_for_frame(axes[5:10], frameData, dataset.get_header())
        labels3 = plotting.plot_image_cracks_for_frame(axes[10:16], frameData, dataset.get_header())
        labels4 = plotting.plot_reference_crack_for_frame(axes[16:19], frameData, dataset.get_header())

        # Assign the labels to the axes.
        labels = [''] * figureNumber
        labels[0:len(labels1)] = labels1
        labels[5:len(labels2)] = labels2
        labels[10:len(labels3)] = labels3
        labels[16:len(labels4)] = labels4

        figuresDir = os.path.join(outDir, 'figures-{}'.format(dataConfig.metadataFilename))
        if not os.path.exists(figuresDir):
            os.makedirs(figuresDir)

        for figure, ax, label in zip(figures, axes, labels):
            # Skip unused axes.
            if label == '':
                continue

            # Configure the axes to cover the whole figure and render to an image file.
            ax.axis('off')
            ax.set_frame_on(False)
            ax.set_xticks([])
            ax.set_yticks([])
            plt.axis('off')
            figure.savefig(os.path.join(figuresDir, '{}-{}'.format(frameLabel, label)), bbox_inches='tight', pad_inches=0)

        for ax in axes:
            ax.clear()


def plot_crack_area_figures(dataset: 'Dataset'):
    figuresDir = os.path.join(outDir, 'figures-{}'.format(dataConfig.metadataFilename))
    if not os.path.exists(figuresDir):
        os.makedirs(figuresDir)

    fig = plotting.plot_crack_area_chart(dataset, csvOutPath=os.path.join(figuresDir, 'crack-area-data.csv'))
    fig.savefig(os.path.join(figuresDir, 'crack-area'), dpi=300)


def plot_figures(dataset: 'Dataset', frame=None):
    plot_crack_area_figures(dataset)
    plot_frame_data_figures(dataset, frame)


def plot_to_pdf(dataset: 'Dataset', plotFrameFunction: Callable[[List, np.ndarray, List[str]], None]):
    h5Data, header, frameMap, *r = dataset.unpack_vars()

    # Prepare for plotting
    pdfPath = os.path.join(outDir, 'fiber-crack.pdf')
    print("Plotting to {}".format(pdfPath))
    pdf = PdfPages(pdfPath)

    # Prepare a figure with subfigures.
    fig = plt.figure(dpi=300)
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

        # The actual plotting is done by the provided function.
        plotFrameFunction(axes, frameData, header)

        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        for a in axes:
            a.clear()
            a.axis('off')

        print("Rendered in {:.2f} s.".format(time.time() - timeStart))

    # Crack area figure.
    fig = plotting.plot_crack_area_chart(dataset)
    fig.suptitle('Crack area in the current frame')
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    # Print the data-to-camera mapping.
    fig = plotting.plot_data_mapping(dataset)
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    pdf.close()


def plot_crack_extraction_view(axes, frameData, header):
    plotting.plot_original_data_for_frame(axes[0:5], frameData, header)
    plotting.plot_unmatched_cracks_for_frame(axes[5:10], frameData, header)
    plotting.plot_image_cracks_for_frame(axes[10:15], frameData, header)
    plotting.plot_reference_crack_for_frame(axes[15:18], frameData, header)
    plotting.plot_feature_histograms_for_frame(axes[18:20], frameData, header)
    # plotting.plot_crack_prediction_for_frame(axes[18:20], frameData, header)


def plot_crack_prediction_view(axes, frameData, header):
    plotting.plot_image_cracks_for_frame(axes[0:5], frameData, header)
    plotting.plot_crack_prediction_for_frame(axes[5:10], frameData, header)


def plot_optic_flow(dataset: 'Dataset'):
    frameIndex = dataset.get_frame_map().index(1800)

    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)

    plotting.plot_optic_flow_for_frame(ax, dataset, frameIndex)

    plt.show()


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
    # The volume should be exported in Z,Y,X,C with C-order.
    volume = np.empty((frameNumber * frameWidth, frameSize[1], frameSize[0], 4), dtype=np.uint8)

    crackAreaMeasured = dataset.get_metadata_column('crackAreaUnmatchedAndEntropy')
    firstNonEmptyFrame = next(i for i, area in enumerate(crackAreaMeasured.tolist()) if area > 0)

    strain = dataset.get_metadata_column('Strain (%)')
    minStrain = strain[firstNonEmptyFrame]
    maxStrain = np.max(strain)

    colorMap = plt.get_cmap('plasma')

    for f in range(0, frameNumber):
        crackArea = dataset.get_column_at_frame(f, 'cracksFromUnmatchedAndEntropy')
        crackAreaUint8 = np.zeros(crackArea.shape[0:2] + (4,), dtype=np.uint8)

        # Can be negative, since we map color not to the full strain range.
        t = max(0.0, (strain[f] - minStrain) / (maxStrain - minStrain))

        crackAreaUint8[crackArea == 1.0] = np.asarray(colorMap(t)) * 255

        volumeSlabSelector = slice_along_axis(slice(f * frameWidth, f * frameWidth + frameWidth), 0, volume.ndim)
        volume[volumeSlabSelector] = crackAreaUint8.swapaxes(0, 1)

    write_volume_to_datraw(np.flip(volume, 0), os.path.join(outDir, 'crack-volume.raw'))

    # Export the color mapping legend.
    fig = plt.figure(figsize=(4, 1))
    ax = fig.add_axes([0.05, 0.5, 0.9, 0.25])

    norm = mpl.colors.Normalize(vmin=minStrain, vmax=maxStrain)
    colorBar = mpl.colorbar.ColorbarBase(ax, cmap=colorMap, norm=norm, orientation='horizontal')
    colorBar.set_label('Strain (%)')

    fig.savefig(os.path.join(outDir, 'crack-volume-legend.png'))


def export_crack_propagation(dataset: 'Dataset'):
    """
    Export data required for crack propagation comparison into an HDF file.\
    :param dataset:
    :return:
    """

    # Create the output dir.
    crackDataDir = os.path.join(outDir, 'crack-propagation')
    if not os.path.exists(crackDataDir):
        os.makedirs(crackDataDir)

    # Remove the old file, if exists.
    crackDataFilePath = os.path.join(crackDataDir, 'crack-propagation_{}.hdf'.format(dataConfig.metadataFilename))
    if os.path.exists(crackDataFilePath):
        os.remove(crackDataFilePath)

    print("Exporting crack propagation data to {}.".format(crackDataFilePath))

    h5File = h5py.File(crackDataFilePath, 'w')

    header = dataset.get_header()
    crack = dataset.h5Data[..., header.index('sigmaSkeleton')]  # type: np.ndarray
    frameMap = dataset.get_frame_map()
    strain = dataset.get_metadata_column('Strain (%)')

    h5File.create_dataset('sigmaSkeleton', data=crack)
    h5File.create_dataset('frameMap', data=frameMap)
    h5File.create_dataset('strain', data=strain)
    
    h5File.close()


def main():

    # Define which commands are possible.
    commandMap = {
        'plot': lambda: plot_to_pdf(dataset, plot_crack_extraction_view),
        'export-crack-volume': lambda: export_crack_volume(dataset),
        'optic-flow': lambda: plot_optic_flow(dataset),
        'export-figures': lambda: plot_figures(dataset, frame),
        'export-crack-propagation': lambda: export_crack_propagation(dataset),
        'plot-prediction': lambda: plot_to_pdf(dataset, plot_crack_prediction_view)
    }

    # Parse the arguments.
    parser = argparse.ArgumentParser('Fiber crack.')
    parser.add_argument('-c', '--command', default='plot', choices=commandMap.keys())
    parser.add_argument('-f', '--frame', default=None, type=int)
    args = parser.parse_args()

    timeStart = time.time()
    print("Loading the data.")
    dataset = load_data()
    print("Data loaded in {:.3f} s. Shape: {} Columns: {}".format(time.time() - timeStart, dataset.h5Data.shape, dataset.get_header()))
    print("Data attributes: {}".format(dataset.get_all_attrs()))

    timeStart = time.time()
    print("Augmenting the data.")
    augment_data(dataset)
    print("Data augmented in {:.3f} s.".format(time.time() - timeStart))

    timeStart = time.time()
    compute_and_append_results(dataset)
    print("Results computed and appended in {:.3f} s.".format(time.time() - timeStart))

    timeStart = time.time()
    print("Executing command: {}".format(args.command))
    frame = args.frame if 'frame' in args.__dict__ else None

    commandMap[args.command]()
    print("Command executed in {:.3f} s.".format(time.time() - timeStart))

    # timeStart = time.time()
    # print("Making a prediction.")
    # predict(dataset: 'Dataset')
    # print("Prediction finished in {:.3f} s.".format(time.time() - timeStart))

    # https://github.com/tensorflow/tensorflow/issues/3388
    # keras.backend.clear_session()


main()
