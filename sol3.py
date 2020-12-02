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
    kernel = np.expand_dims(build_filter(filter_size), axis=0)
    im = ndimage.filters.convolve(im, kernel, mode='mirror')
    im = im[:, ::2]
    # then do it along y axis
    im = ndimage.filters.convolve(im, kernel.transpose(), mode='mirror')
    im = im[::2, :]
    return im


def expand(im, filter_size):
    N, M = im.shape
    # first do along x axis
    kernel = np.expand_dims(build_filter(filter_size), axis=0)
    im = np.insert(im, np.arange(1, M), 0,  axis=1)
    im = np.append(im, np.zeros((N,1)), axis=1)
    # twice the kernel in order to maintain same average pixel brightness when expanding
    im = ndimage.filters.convolve(im, 2*kernel, mode='mirror')
    # now, along y axis
    im = np.insert(im, np.arange(1, N), 0, axis=0)
    im = np.append(im, np.zeros((1, 2*M)), axis=0)
    im = ndimage.filters.convolve(im, 2*kernel.transpose(), mode='mirror')
    return im