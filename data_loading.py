import io
import csv
import numpy as np

def readDataFromCsv(filename, encoding='utf-8'):
    with io.open(filename, newline='', encoding=encoding) as file:
        csvReader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONE)
        header = [s.strip(' "') for s in csvReader.__next__()]

    data = np.genfromtxt(filename, delimiter=',', skip_header=1)

    return data, header


def readHeaderFromCsv(filename, encoding='utf-8'):
    with io.open(filename, newline='', encoding=encoding) as file:
        csvReader = csv.reader(file, delimiter=',', quoting=csv.QUOTE_NONE)
        return [s.strip(' "') for s in csvReader.__next__()]