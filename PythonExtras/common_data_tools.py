import io
import csv

from typing import List, Dict, Union

import numpy as np
from PIL import Image


def read_csv_data(filepath, encoding='utf-8', delimiter='\t'):
    with io.open(filepath, newline='', encoding=encoding) as file:
        csvReader = csv.reader(file, delimiter=delimiter, quoting=csv.QUOTE_NONE)
        header = [s.strip(' "') for s in csvReader.__next__()]

    data = np.genfromtxt(filepath, delimiter=delimiter, skip_header=1)

    return data, header


def read_csv_header(filepath, encoding='utf-8'):
    with io.open(filepath, newline='', encoding=encoding) as file:
        csvReader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONE)
        return [s.strip(' "') for s in csvReader.__next__()]


def write_to_csv(outputPath: str, data: Union[np.ndarray, np.ma.MaskedArray],
                 header: List[str], addPaddingColumn: bool = False,
                 delimiter='\t', fmt='{:.6f}'):
    """
    Writes a Numpy array to CSV. Handles both vanilla and masked arrays.
    Has an option of adding a dummy padding column, which helps with import into LaTex.
    :return:
    """

    if addPaddingColumn:
        paddingColumn = np.tile(-1, (data.shape[0], 1))
        data = np.ma.hstack((data, paddingColumn))
        header = header + ['padding']

    with open(outputPath, 'w', newline='') as f:
        writer = csv.writer(f, delimiter=delimiter)  # type: csv.
        writer.writerow(header)
        for i in range(data.shape[0]):
            dataRow = data[i, :]
            if isinstance(data, np.ma.MaskedArray):
                stringRow = [fmt.format(val) if not dataRow.mask[j] else '' for j, val in enumerate(dataRow)]
            else:
                stringRow = [fmt.format(val) for j, val in enumerate(dataRow)]
            writer.writerow(stringRow)


def read_tiff_data(filepath):
    tiffFile = Image.open(filepath, 'r')

    fullData = np.empty((tiffFile.n_frames, tiffFile.size[1], tiffFile.size[0]), dtype=np.float)
    for f in range(0, tiffFile.n_frames):
        tiffFile.seek(f)
        fullData[f, ...] = np.array(tiffFile)

    return fullData.swapaxes(1, 2)