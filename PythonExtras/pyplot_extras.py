import math
import matplotlib.pyplot as plt


def get_subplot_grid_size(subplotNumber):
    gridSizeLin = int(math.ceil(math.sqrt(subplotNumber)))
    width = gridSizeLin
    height = int(math.ceil(subplotNumber / max(width, 1)))

    return width, height