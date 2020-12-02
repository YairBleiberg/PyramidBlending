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
    # first blur and take every 2nd pixel along x axis
    filter = np.expand_dims(build_filter(filter_size), axis=0)
    im = ndimage.filters.convolve(im, filter, mode='reflect')
    im = im[:, ::2]
    # then do it along y axis
    im = ndimage.filters.convolve(im, filter.transpose, mode='reflect')
    im = im[::2, :]
    return im

print(build_filter(5))