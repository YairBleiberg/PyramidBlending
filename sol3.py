import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
# builds gaussian filter
def build_filter(filter_size):
    if filter_size == 1:
        return np.ones(1)
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


def build_gaussian_pyramid(im, max_levels, filter_size):
    pyr = [im]
    for i in np.arange(max_levels-1):
        if np.min(im.shape) < 32:
            break
        im = reduce(im, filter_size)
        pyr.append(im)
    return [pyr, build_filter(filter_size)]


def build_laplacian_pyramid(im, max_levels, filter_size):
    pyr, filter_vec = build_gaussian_pyramid(im, max_levels, filter_size)
    for i in range(len(pyr)-1):
        pyr[i] = pyr[i] - expand(pyr[i+1], filter_size)
    return [pyr, filter_vec]


def laplacian_to_image(lpyr, filter_vec, coeff):
    # we reverse the pyramid to start the sum from the bottom
    lpyr.reverse()
    coeff.reverse()
    img = coeff[0]*lpyr[0]
    for i in np.arange(1, len(lpyr)):
        img = expand(img, filter_vec.size) + coeff[i]*lpyr[i]
    return img


def render_pyramid(pyr, levels):
    res = (pyr[0]-np.amin(pyr[0]))/(np.amax(pyr[0])-np.amin(pyr[0]))
    N, M = pyr[0].shape
    for i in np.arange(1, levels):
        normalized_level = (pyr[i]-np.amin(pyr[i]))/(np.amax(pyr[i])-np.amin(pyr[i]))
        add_level = np.pad(normalized_level, ((0, N-pyr[i].shape[0]), (0, 0)))
        res = np.append(res, add_level, axis=1)
    return res


def display_pyramid(pyr, levels):
    res = render_pyramid(pyr, levels)
    plt.imshow(res)
    plt.show()
def pyramid_blending(im1, im2, mask, max_levels, filter_size_im, filter_size_mask):
        