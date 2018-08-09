import json


import FiberCrack.data_loading as data_loading


class FiberCrackConfig:
    
    def __init__(self):

        self.description = "Default config description."
        
        self.dataConfig = data_loading.DataImportConfig()
        self.dataConfig.preloadedDataDir = 'C:/preloaded_data'
        self.dataConfig.preloadedDataFilename = None  # By default, decide automatically.
        self.dataConfig.dataFormat = 'csv'
        self.dataConfig.imageFilenameFormat = '{}-{:04d}_0.tif'

        self.dataConfig.reloadOriginalData = False

        self.maxFrames = 99999
        self.recomputeResults = False

        self.outDir = 'T:/out/fiber-crack'

        self.dataConfig.dicKernelSize = 55

        self.textureKernelMultiplier = 1.0
        self.entropyThreshold = 1.0 
        self.varianceThreshold = 0.003 

        self.unmatchedPixelsPadding = 0.15 
        self.unmatchedPixelsMorphologyDepth = 2   # How many dilations/erosions are used before removing objects/holes.
        self.unmatchedPixelsObjectsThreshold = 1 / 50   # Fraction of the image area.
        self.unmatchedPixelsHolesThreshold = 1 / 6   # Fraction of the zero-valued image area (sclaes with the crack).

        self.hybridKernelMultiplier = 0.5 
        self.hybridDilationDepth = 3   # How many dilations are applied to expand the search range.

        self.sigmaSkeletonPadding = 0.15 
        self.exportedVolumeTimestepWidth = 3 
        self.exportedVolumeGradientWidth = 3 
        self.exportedVolumeSkippedFrames = 5 
        self.exportedVolumeStrainMin = 4.0 
        self.exportedVolumeStrainMax = 7.5 

        self.allTextureKernelMultipliers = [2.0, 1.5, 1.0, 0.5, 0.25] 
        self.textureFilters = ['entropy', 'variance']

        self.enablePrediction = True

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
