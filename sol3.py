import numpy as np
from scipy import ndimage
# builds gaussian filter
def build_filter(filter_size):
    filter = np.ones(2)
    for i in np.arange(filter_size-2):
        filter = np.convolve(filter, np.ones(2))
    filter /= np.sum(filter)
    return filter

def reduce(im, filter_size):
    filter = build_filter(filter_size)




print(build_filter(5))