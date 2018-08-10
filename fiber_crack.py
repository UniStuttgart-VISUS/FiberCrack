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
from matplotlib.backends.backend_pdf import PdfPages

import FiberCrack.crack_detection as crack_detection
import FiberCrack.crack_prediction as crack_prediction
import FiberCrack.crack_metrics as crack_metrics
import FiberCrack.data_augmentation as data_augmentation
import FiberCrack.data_loading as data_loading
import FiberCrack.plotting as plotting
from FiberCrack.Dataset import Dataset
from FiberCrack.FiberCrackConfig import FiberCrackConfig

from PythonExtras.numpy_extras import slice_along_axis
from PythonExtras.volume_tools import write_volume_to_datraw, write_volume_sequence

np.random.seed(13)  # Fix the seed for reproducibility.
# import tensorflow as tf
# tf.set_random_seed(13)

############### Configuration ###############


# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Steel-Epoxy'
# metadataFilename = 'Steel-Epoxy.csv'
# dataDir = 'data_export_tstep3'
# imageDir = 'raw_images'
# imageBaseName = 'Spec054'
# dicKernelSize = 85

# Steel-Epoxy dataset. We are comparing crack progression between this one and PTFE-Epoxy.
# -- Moved to steel-epoxy.json

# basePath = '//visus/visusstore/share/Daten/Sonstige/Montreal/Experiments/Steel-ModifiedEpoxy'
# metadataFilename = 'Steel-ModifiedEpoxy.csv'
# dataDir = 'data_export'
# imageDir = 'raw_images'
# imageBaseName = 'Spec010'
# dicKernelSize = 55

# The cleanest dataset: PTFE with epoxy.
# -- Moved to ptfe-epoxy.json

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


def load_data(config: FiberCrackConfig):
    if config.dataConfig.dataFormat == 'csv':
        return data_loading.load_csv_data(config.dataConfig)
    elif config.dataConfig.dataFormat == 'tiff':
        return data_loading.load_tiff_data(config.dataConfig)
    else:
        raise ValueError("Unknown data format: {}".format(config.dataConfig.dataFormat))


def augment_data(dataset: 'Dataset', config: FiberCrackConfig):
    """
    Extends the raw data with some pre-processing.
    Doesn't compute any 'results', but rather information that can help compute the results.

    :param dataset:
    :param config:
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
    data_augmentation.append_camera_image(dataset, config.dataConfig)
    print("Adding the crack ground truth images...")
    data_augmentation.append_ground_truth_image(dataset, config.dataConfig)
    print("Adding the matched pixels...")
    data_augmentation.append_matched_pixels(dataset, config.dataConfig)

    print("Adding crack area ground truth...")
    data_augmentation.append_crack_area_ground_truth(dataset, config.dataConfig)

    print("Zeroing the pixels that lost tracking.")
    data_augmentation.zero_pixels_without_tracking(dataset)

    return dataset


def apply_function_if_code_changed(dataset: 'Dataset', config: FiberCrackConfig, func: Callable[..., None]):
    """
    Calls a function that computes and writes data to the dataset.
    Stores the hash of the function's source code as metadata.
    If the function has not changed, it isn't applied to the data.

    :param dataset:
    :param config:
    :param func:
    :return:
    """
    # Get a string containing the full function source.
    sourceLines = inspect.getsourcelines(func)
    functionSource = ''.join(sourceLines[0])
    functionName = func.__name__

    callSignature = inspect.signature(func)
    callArguments = {}
    for callParameter in callSignature.parameters:
        if callParameter in config.__dict__:
            callArguments[callParameter] = config.__dict__[callParameter]

    callArgumentsString = ''.join([key + str(callArguments[key]) for key in sorted(callArguments)])

    attrName = '_functionHash_' + functionName
    currentHash = hashlib.sha1((functionSource + callArgumentsString).encode('utf-8')).hexdigest()

    if 'cameraImageVar' in dataset.get_header() and not config.recomputeResults:
        oldHash = dataset.get_attr(attrName) if dataset.has_attr(attrName) else None

        if currentHash == oldHash:
            print("Function {} has not changed, skipping.".format(functionName))
            return

    print("Applying function {} to the dataset.".format(functionName))

    callArguments['dataset'] = dataset
    func(**callArguments)

    dataset.set_attr(attrName, currentHash)


def compute_and_append_results(dataset: 'Dataset', config: FiberCrackConfig):
    # Compute derived parameters.
    mappingMin, mappingMax, mappingStep = dataset.get_data_image_mapping()
    dicKernelRadius = int((config.dataConfig.dicKernelSize - 1) / 2 / mappingStep[0])
    textureKernelMultipliers = config.__dict__['allTextureKernelMultipliers']
    config.__dict__['dicKernelRadius']       = dicKernelRadius
    config.__dict__['textureKernelSize']     = int(dicKernelRadius * config.__dict__['textureKernelMultiplier'])
    config.__dict__['allTextureKernelSizes'] = [int(dicKernelRadius * mult) for mult in textureKernelMultipliers]

    apply_function_if_code_changed(dataset, config, data_augmentation.append_texture_features)

    apply_function_if_code_changed(dataset, config, crack_detection.append_crack_from_tracking_loss)
    apply_function_if_code_changed(dataset, config, crack_detection.append_crack_from_unmatched_pixels)

    apply_function_if_code_changed(dataset, config, crack_detection.append_crack_from_variance)
    apply_function_if_code_changed(dataset, config, crack_detection.append_crack_from_entropy)

    apply_function_if_code_changed(dataset, config, crack_detection.append_crack_from_unmatched_and_entropy)
    apply_function_if_code_changed(dataset, config, crack_detection.append_reference_frame_crack)

    if config.enablePrediction:
        apply_function_if_code_changed(dataset, config, crack_prediction.append_crack_prediction_simple)
        apply_function_if_code_changed(dataset, config, crack_prediction.append_crack_prediction_spatial)

    # todo this runs always, because there are too many dependencies.
    crack_metrics.append_estimated_crack_area(dataset)


def plot_frame_data_figures(dataset: 'Dataset', config: FiberCrackConfig, targetFrame=None):

    # figures = [plt.figure(dpi=300) for i in range(figureNumber)]
    # axes = [fig.add_subplot(1, 1, 1) for fig in figures]
    figures = []
    axes = []
    labels = []

    def axis_builder(label: str) -> plt.Axes:
        fig = plt.figure(dpi=300)
        ax = fig.add_subplot(1, 1, 1)
        figures.append(fig)
        axes.append(ax)
        labels.append(label)
        return ax

    for frame in range(dataset.get_frame_number()):

        # If the target frame is specified, skip the other frames.
        if targetFrame is not None and dataset.get_frame_map()[frame] != targetFrame:
            continue

        frameData = dataset.h5Data[frame, ...]
        frameLabel = dataset.get_frame_map()[frame]

        # Plot the data.
        plotting.plot_original_data_for_frame(axis_builder, frameData, dataset.get_header())
        plotting.plot_unmatched_cracks_for_frame(axis_builder, frameData, dataset.get_header())
        plotting.plot_image_cracks_for_frame(axis_builder, frameData, dataset.get_header())
        plotting.plot_reference_crack_for_frame(axis_builder, frameData, dataset.get_header())

        # Assign the labels to the axes.
        # todo does this work?
        # labels = [''] * figureNumber
        # labels[0:len(labels1)] = labels1
        # labels[5:len(labels2)] = labels2
        # labels[12:len(labels3)] = labels3
        # labels[20:len(labels4)] = labels4

        figuresDir = os.path.join(config.outDir, 'figures-{}'.format(config.dataConfig.metadataFilename))
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

    # Cleanup, since pyplot doesn't do it automatically.
    for fig in figures:
        plt.close(fig)


def plot_crack_area_figures(dataset: 'Dataset', config: FiberCrackConfig):
    figuresDir = os.path.join(config.outDir, 'figures-{}'.format(config.dataConfig.metadataFilename))
    if not os.path.exists(figuresDir):
        os.makedirs(figuresDir)

    fig = plotting.plot_crack_area_chart(dataset, csvOutPath=os.path.join(figuresDir, 'crack-area-data.csv'))
    fig.savefig(os.path.join(figuresDir, 'crack-area'), dpi=300)


def plot_figures(dataset: 'Dataset', config: FiberCrackConfig, frame=None):
    plot_crack_area_figures(dataset, config)
    plot_frame_data_figures(dataset, config, frame)


def plot_to_pdf(dataset: 'Dataset', config: FiberCrackConfig,
                plotFrameFunction: Callable[[Callable[[str], plt.Axes], np.ndarray, List[str]], None]):
    h5Data, header, frameMap, *r = dataset.unpack_vars()

    # Prepare for plotting
    pdfPath = os.path.join(config.outDir, 'fiber-crack.pdf')
    print("Plotting to {}".format(pdfPath))
    pdf = PdfPages(pdfPath)

    # Prepare a figure with subfigures.
    fig = plt.figure(dpi=300)
    axes = []

    def axis_builder(label: str) -> plt.Axes:
        if len(axes) >= 6 * 6:
            raise RuntimeError("PDF layout doesn't have enough subplots.")

        ax = fig.add_subplot(6, 6, len(axes) + 1)
        axes.append(ax)

        ax.axis('off')
        return ax

    fig.subplots_adjust(hspace=0.025, wspace=0.025)

    # Draw the frame plots.
    for f in range(0, dataset.get_frame_number()):
        timeStart = time.time()
        frameIndex = frameMap[f]
        print("Plotting frame {}".format(frameIndex))
        fig.suptitle("Frame {}".format(frameIndex))

        frameData = h5Data[f, :, :, :]

        # The actual plotting is done by the provided function.
        plotFrameFunction(axis_builder, frameData, header)

        pdf.savefig(fig, bbox_inches='tight', dpi=300)
        for a in axes:
            a.clear()
            a.axis('off')
        axes = []

        print("Rendered in {:.2f} s.".format(time.time() - timeStart))

    # Crack area figure.
    fig = plotting.plot_crack_area_chart(dataset)
    fig.suptitle('Crack area in the current frame')
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    # Print the data-to-camera mapping.
    fig = plotting.plot_data_mapping(dataset)
    pdf.savefig(fig, bbox_inches='tight', dpi=300)

    pdf.close()
    print("Finished plotting to {}".format(pdfPath))


def plot_crack_extraction_view(axisBuilder: Callable[[str], plt.Axes], frameData, header):
    plotting.plot_original_data_for_frame(axisBuilder, frameData, header)
    plotting.plot_unmatched_cracks_for_frame(axisBuilder, frameData, header)
    plotting.plot_image_cracks_for_frame(axisBuilder, frameData, header)
    plotting.plot_reference_crack_for_frame(axisBuilder, frameData, header)
    plotting.plot_feature_histograms_for_frame(axisBuilder, frameData, header)
    # plotting.plot_crack_prediction_for_frame(axisBuilder, frameData, header)


def plot_crack_prediction_view(axisBuilder: Callable[[str], plt.Axes], frameData, header):
    plotting.plot_image_cracks_for_frame(axisBuilder, frameData, header)
    plotting.plot_crack_prediction_for_frame(axisBuilder, frameData, header)


def plot_optic_flow(dataset: 'Dataset'):
    frameIndex = dataset.get_frame_map().index(1800)

    figure = plt.figure()
    ax = figure.add_subplot(1, 1, 1)

    plotting.plot_optic_flow_for_frame(ax, dataset, frameIndex)

    plt.show()


def export_crack_volume(dataset: 'Dataset', config: FiberCrackConfig):
    """
    Build a volume by concatenating crack areas from each frame,
    save to the disk in datraw format.
    :param dataset:
    :param config:
    :return:
    """
    frameSize = dataset.get_frame_size()

    framesToSkip = config.exportedVolumeSkippedFrames
    frameWidth = config.exportedVolumeTimestepWidth
    frameNumber = dataset.get_frame_number() - framesToSkip
    # The volume should be exported in Z,Y,X,C with C-order.
    volume = np.zeros((frameNumber * frameWidth, frameSize[1], frameSize[0], 4), dtype=np.uint8)

    # crackAreaMeasured = dataset.get_metadata_column('crackAreaHybrid')
    # firstNonEmptyFrame = next(i for i, area in enumerate(crackAreaMeasured.tolist())
    #                           if area > 0 and dataset.get_metadata_val(i, 'hasCameraImage'))

    strain = dataset.get_metadata_column('Strain (%)')
    minStrain = config.exportedVolumeStrainMin
    maxStrain = config.exportedVolumeStrainMax

    colorMap = plt.get_cmap('plasma')

    for f in range(0, frameNumber):
        if not dataset.get_metadata_val(f, 'hasCameraImage'):
            continue

        crackArea = dataset.get_column_at_frame(f, 'hybridCracks')
        crackAreaUint8 = np.zeros(crackArea.shape[0:2] + (4,), dtype=np.uint8)

        # Can be negative, since we map color not to the full strain range.
        t = max(0.0, (strain[f] - minStrain) / (maxStrain - minStrain))

        crackAreaUint8[crackArea == 1.0] = np.asarray(colorMap(t)) * 255

        volumeSlabSelector = slice_along_axis(slice(f * frameWidth, f * frameWidth + frameWidth), 0, volume.ndim)
        volume[volumeSlabSelector] = crackAreaUint8.swapaxes(0, 1)

    write_volume_to_datraw(np.flip(volume, 0), os.path.join(config.outDir, 'crack-volume.raw'))

    # Export the color mapping legend.
    fig = plt.figure(figsize=(4, 1))
    ax = fig.add_axes([0.05, 0.5, 0.9, 0.25])

    norm = mpl.colors.Normalize(vmin=minStrain, vmax=maxStrain)
    colorBar = mpl.colorbar.ColorbarBase(ax, cmap=colorMap, norm=norm, orientation='horizontal')
    colorBar.set_label('Strain (%)')

    fig.savefig(os.path.join(config.outDir, 'crack-volume-legend.png'))


def export_displacement_volume(dataset: 'Dataset', config: FiberCrackConfig):
    frameSize = dataset.get_frame_size()

    frameNumber = dataset.get_frame_number()
    # The volume should be exported in Z,Y,X with C-order.
    volume = np.zeros((frameNumber, 1, frameSize[1], frameSize[0]), dtype=np.float)

    for f in range(0, frameNumber):
        displacementX = dataset.get_column_at_frame(f, 'u')
        displacementY = dataset.get_column_at_frame(f, 'v')

        volume[f, 0, ...] = np.transpose(np.sqrt(np.square(displacementX) + np.square(displacementY)))

    maxValue = np.max(volume)
    meanValue = np.mean(volume)
    print("Max displacement value (mapped to 255): {}".format(maxValue))
    print("Mean displacement value: {}".format(meanValue))

    # Manually set the mapping range to make it consistent across different datasets.
    mappingMax = 700
    if mappingMax < maxValue:
        raise RuntimeError("Dataset values are getting clipped when mapping to volume values. Max value: {}"
                           .format(maxValue))

    for f in range(0, frameNumber):
        sigma = dataset.get_column_at_frame(f, 'sigma')
        volume[f, 0, ...] = volume[f, 0, ...] / mappingMax * 127 + 127
        for y in range(volume.shape[2]):
            volume[f, 0, y, :][sigma[:, y] < 0] = 0  # Set to zero areas with no tracking.

    volumeDir = os.path.join(config.outDir, os.path.basename(config.dataConfig.metadataFilename))
    write_volume_sequence(volumeDir, volume, clip=(0, 255), dtype=np.uint8)


def export_crack_propagation(dataset: 'Dataset', config: FiberCrackConfig):
    """
    Export data required for crack propagation comparison into an HDF file.\
    :param dataset:
    :param config:
    :return:
    """

    # Create the output dir.
    crackDataDir = os.path.join(config.outDir, 'crack-propagation')
    if not os.path.exists(crackDataDir):
        os.makedirs(crackDataDir)

    # Remove the old file, if exists.
    crackDataFilePath = os.path.join(crackDataDir, 'crack-propagation_{}.hdf'.format(config.dataConfig.metadataFilename))
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


def fiber_crack_run(command: str, config: FiberCrackConfig, frame: int = None):
    # Define which commands are possible.
    commandMap = {
        'plot': lambda: plot_to_pdf(dataset, config, plot_crack_extraction_view),
        'export-crack-volume': lambda: export_crack_volume(dataset, config),
        'export-displacement-volume': lambda: export_displacement_volume(dataset, config),
        'optic-flow': lambda: plot_optic_flow(dataset),
        'export-figures': lambda: plot_figures(dataset, config, frame),
        'export-figures-only-area': lambda: plot_crack_area_figures(dataset, config),
        'export-crack-propagation': lambda: export_crack_propagation(dataset, config),
        'plot-prediction': lambda: plot_to_pdf(dataset, config, plot_crack_prediction_view)
    }
    
    timeStart = time.time()
    print("Loading the data.")
    dataset = load_data(config)
    print("Data loaded in {:.3f} s. Shape: {} Columns: {}".format(time.time() - timeStart, dataset.h5Data.shape,
                                                                  dataset.get_header()))
    print("Data attributes: {}".format(dataset.get_all_attrs()))
    timeStart = time.time()
    print("Augmenting the data.")
    augment_data(dataset, config)
    print("Data augmented in {:.3f} s.".format(time.time() - timeStart))
    timeStart = time.time()
    compute_and_append_results(dataset, config)
    print("Results computed and appended in {:.3f} s.".format(time.time() - timeStart))
    timeStart = time.time()
    print("Executing command: {}".format(command))
    commandMap[command]()
    print("Command executed in {:.3f} s.".format(time.time() - timeStart))
    # timeStart = time.time()
    # print("Making a prediction.")
    # predict(dataset: 'Dataset')
    # print("Prediction finished in {:.3f} s.".format(time.time() - timeStart))
    # https://github.com/tensorflow/tensorflow/issues/3388
    # keras.backend.clear_session()


def main():
    commandList = [
        'plot',
        'export-crack-volume',
        'export-displacement-volume',
        'optic-flow',
        'export-figures',
        'export-figures-only-area',
        'export-crack-propagation',
        'plot-prediction',
    ]
    
    # Parse the arguments.
    parser = argparse.ArgumentParser('Fiber crack.')
    parser.add_argument('-c', '--command', default='plot', choices=commandList)
    parser.add_argument('-g', '--config', required=True, type=str)
    parser.add_argument('-f', '--frame', default=None, type=int)
    args = parser.parse_args()

    configPath = args.config
    config = FiberCrackConfig()
    config.read_from_file(configPath)

    # todo: Seems that I had a bug, always using the same wrong DIC kernel size value.
    #       For consistency, I'm keeping it this way for now, but in the future, I should remove this
    #       and adjust the 'hybridKernelMultiplier' to compensate for the change.
    #       Also, possibly, need to remove the '+1' in the DIC compensation iterations.
    config.dataConfig.dicKernelSize = 55

    fiber_crack_run(args.command,
                    config,
                    args.frame if 'frame' in args.__dict__ else None)


if __name__ == '__main__':
    main()

