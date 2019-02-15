import json


import data_loading as data_loading


class FiberCrackConfig:
    
    def __init__(self):

        self.description = "Default config description."
        
        self.dataConfig = data_loading.DataImportConfig()
        # Path to a temporary storage, where loaded data will be stored as a single HDF array for future use.
        # This makes running the tools much faster than loading the data every time from scratch.
        self.dataConfig.preloadedDataDir = ''
        self.dataConfig.preloadedDataFilename = None  # By default, decide automatically.
        self.dataConfig.dataFormat = 'csv'
        self.dataConfig.imageFilenameFormat = '{}-{:04d}_0.tif'

        self.outDir = ''

        # Force a reload of the data, instead of using the preloaded data location.
        self.dataConfig.reloadOriginalData = False
        # Force results to be recomputed, instead of using the preloaded data location.
        self.recomputeResults = False

        # Default kernel size (typically overriden for each dataset).
        self.dataConfig.dicKernelSize = 80

        # Try out multiple kernel sizes for potential visual comparison of the results.
        self.allTextureKernelMultipliers = [1.5, 1.0, 0.5]
        self.textureFilters = ['entropy', 'variance']
        # Texture kernel size that will be used for image-based crack extraction.
        self.textureKernelMultiplier = 0.8
        # Thresholds for the texture filters.
        self.entropyThreshold = 1.0 
        self.varianceThreshold = 0.003 

        # Unmatched pixels (DIC) are padded because DIC always fails at the borders of ROI.
        self.unmatchedPixelsPadding = 0.15
        self.sigmaSkeletonPadding = 0.15

        self.unmatchedPixelsMorphologyDepth = 2   # How many dilations/erosions are used before removing objects/holes.
        self.unmatchedPixelsObjectsThreshold = 1 / 50   # Fraction of the image area.
        self.unmatchedPixelsHolesThreshold = 1 / 6   # Fraction of the zero-valued image area (scales with the crack).

        # Texture kernel size that will be used for hybrid crack extraction.
        self.hybridKernelMultiplier = 0.4
        # How many dilations are applied to expand the search range.
        self.hybridDilationDepth = 3

        # Volume export configuration: which timesteps to use, how to map strain to volume density.
        self.exportedVolumeTimestepWidth = 3
        self.exportedVolumeGradientWidth = 3
        self.exportedVolumeSkippedFrames = 5
        self.exportedVolumeStrainMin = 4.0
        self.exportedVolumeStrainMax = 7.5

        # Magnified figures settings.
        self.magnifiedFigureNames = ['hybrid-crack-thin', 'matched-pixels-crack-thin']
        self.magnifiedRegion = ((0.5, 0.7), (0.3, 0.55))
        self.magnifiedRatio = 2

    def read_from_file(self, path: str):
        print("Reading FiberCrack config at '{}'".format(path))
        with open(path, 'r') as file:
            config = json.load(file)

        for key, value in config.items():
            if key in self.__dict__ and key != 'dataConfig':
                self.__dict__[key] = value
            elif key == 'dataConfig':
                self.dataConfig.load_from_dict(config['dataConfig'])
            else:
                raise RuntimeError("Unknown FiberCrack config parameter: '{}'".format(key))
