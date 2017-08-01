import h5py
import numpy as np


__all__ = ['Dataset']


class Dataset:

    @staticmethod
    def get_data_feature_number():
        """
        Specifies how many columns should be created when allocating an hdf5 array.
        :return:
        """
        # When allocating a continuous hdf5 file we need to know the final size
        # of the data in advance.

        augmentedFeatureNumber = 4
        resultsFeatureNumber = 30  # Approximately, leave some empty.

        return augmentedFeatureNumber + resultsFeatureNumber

    def __init__(self, h5File):
        self.h5Data = h5File['data']
        self.h5Header = h5File['header']
        self.h5FrameMap = h5File['frameMap']
        self.h5Metadata = h5File['metadata']
        self.h5Metaheader = h5File['metaheader']
        self.h5ArrayAttributes = h5File['arrayAttributes']

    def unpack_vars(self):
        header = self.get_header()
        frameMap = self.h5FrameMap[:].tolist()
        metadata = self.h5Metadata[:].tolist()
        metaheader = self.get_metaheader()
        return self.h5Data, header, frameMap, metadata, metaheader

    def create_or_get_column(self, newColumnName):
        """
        Appends a new empty column to the data. If the column already exists, returns its index.

        Note: we do not insert the data directly, because typically we don't want to
        copy it into a single huge array. The size would be to big.
        :param newColumnName:
        :return:
        """

        # Check if the column already exists.
        header = self.get_header()
        if newColumnName in header:
            return header.index(newColumnName)

        # Make sure we still have preallocated space available.
        assert(self.h5Data.shape[-1] > self.h5Header.shape[0])

        self.h5Header.resize((self.h5Header.shape[0] + 1,))
        self.h5Header[-1] = newColumnName.encode('ascii')

        newColumnIndex = self.h5Header.shape[0] - 1
        return newColumnIndex

    def create_or_update_metadata_column(self, newColumnName, newColumnData):

        assert(newColumnData.shape[0] == self.h5Metadata.shape[0])

        # Check if the column already exists.
        metaheader = self.get_metaheader()
        if newColumnName not in metaheader:
            self.h5Metadata.resize(self.h5Metadata.shape[1] + 1, axis=1)
            self.h5Metaheader.resize(self.h5Metaheader.shape[0] + 1, axis=0)
            self.h5Metadata[:, -1] = newColumnData
            self.h5Metaheader[-1] = newColumnName.encode('ascii')

            return self.h5Metaheader.shape[0] - 1
        else:
            self.h5Metadata[:, metaheader.index(newColumnName)] = newColumnData
            return metaheader.index(newColumnName)

    def get_column_at_frame(self, frame, columnName) -> np.ndarray:
        return self.h5Data[frame, ..., self.get_header().index(columnName)]

    def get_data_image_mapping(self):
        return (self.get_attr('mappingMin'), self.get_attr('mappingMax'), self.get_attr('mappingStep'))

    def get_image_shift(self):
        metaheader = self.get_metaheader()
        indices = [metaheader.index('imageShiftX'), metaheader.index('imageShiftY')]
        return self.h5Metadata[:, indices].astype(np.int)

    def get_frame_number(self):
        return self.h5Data.shape[0]

    def get_frame_size(self):
        return tuple(self.h5Data.shape[1:3])

    def get_metadata_val(self, frame, columnName):
        return self.h5Metadata[frame, self.get_metaheader().index(columnName)]

    def get_metadata_column(self, columnName):
        return self.h5Metadata[:, self.get_metaheader().index(columnName)]

    def set_attr(self, attrName, attrValue):
        self.h5Data.attrs[attrName] = attrValue

    def set_str_array_attr(self, attrName, array):
        """
        Pack a small string array into a string attribute.
        More convenient than creating separate hdf5 arrays.
        :param attrName:
        :param array:
        :return:
        """
        self.set_attr(attrName, '#!$'.join(array))

    def set_numpy_array_attr(self, attrName, array):
        """
        Store an Numpy array as an 'attribute'.
        The array is actually stored as a separate hdf5 dataset.

        :param attrName:
        :param array:
        :return:
        """
        if attrName in self.h5ArrayAttributes:
            storage = self.h5ArrayAttributes[attrName]
            if storage.shape != array.shape:
                del self.h5ArrayAttributes[attrName]
                self.h5ArrayAttributes.create_dataset(attrName, data=array)
            else:
                storage[...] = array
        else:
            self.h5ArrayAttributes.create_dataset(attrName, data=array)

    def get_attr(self, attrName):
        return self.h5Data.attrs[attrName]

    def get_str_array_attr(self, attrName):
        return self.get_attr(attrName).split('#!$')

    def get_numpy_array_attr(self, attrName):
        return self.h5ArrayAttributes[attrName][...]

    def get_all_attrs(self):
        return {name: self.h5Data.attrs[name] for name in self.h5Data.attrs}

    def has_attr(self, attrName):
        return attrName in self.h5Data.attrs

    def has_numpy_array_attr(self, attrName):
        return attrName in self.h5ArrayAttributes

    def get_header(self):
        return self.h5Header[:].astype(np.str).tolist()

    def get_metaheader(self):
        return self.h5Metaheader[:].astype(np.str).tolist()

    def get_frame_map(self):
        return self.h5FrameMap[:].tolist()