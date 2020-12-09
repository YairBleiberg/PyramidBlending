import numpy as np
from scipy import ndimage
import matplotlib.pyplot as plt
# builds gaussian filter
def build_filter(filter_size):
    if filter_size == 1:
        return np.ones((1,1))
    filter = np.ones(2)
    for i in np.arange(filter_size-2):
        filter = np.convolve(filter, np.ones(2))
    filter /= np.sum(filter)
    return np.expand_dims(filter, axis=0)


def reduce(im, filter_size):
    # first blur and take every 2nd pixel along x axis
    kernel = build_filter(filter_size)
    im = ndimage.filters.convolve(im, kernel, mode='mirror')
    im = im[:, ::2]
    # then do it along y axis
    im = ndimage.filters.convolve(im, kernel.transpose(), mode='mirror')
    im = im[::2, :]
    return im


def expand(im, filter_size):
    N, M = im.shape
    # first do along x axis
    kernel = build_filter(filter_size)
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
        L1, filter_vec = build_laplacian_pyramid(im1, max_levels, filter_size_im)
        L2, filter_vec = build_laplacian_pyramid(im2, max_levels, filter_size_im)
        Gm, filter_vec = build_gaussian_pyramid(np.float64(mask), max_levels, filter_size_mask)
        Lout = []
        for i in range(len(L1)):
            Lout.append(L1[i]*Gm[i] + L2[i]*(1-Gm[i]))
        im_blend = laplacian_to_image(Lout, build_filter(filter_size_im), np.ones(len(Lout)).tolist())
        return np.clip(im_blend, 0, 1)


import imageio
from skimage.color import rgb2gray
Z = 256


def read_image(filename, representation):
    im = imageio.imread(filename)
    if representation == 2:
        return (im.astype(np.float64))/(Z-1)
    else:
        if len(im.shape) >= 3:
            return (rgb2gray(im).astype(np.float64))
        else:
            return (im.astype(np.float64))/(Z-1)


import os
def relpath(filename):
    return os.path.join(os.path.dirname(__file__), filename)

def blending_example1():
    genie = read_image(relpath("externals/genie.jpg"), 2)
    vaping_cloud = read_image(relpath("externals/cloud.jpg"), 2)
    mask = np.rint(read_image(relpath("externals/genie_mask.jpg"), 1)).astype(bool)
    blended = np.zeros(genie.shape)
    for i in np.arange(3):
        blended[:,:,i] = pyramid_blending(genie[:,:, i], vaping_cloud[:,:,i], mask, 10, 3, 3)
    plt.subplot(2,2,1)
    plt.imshow(genie)
    plt.subplot(2,2,2)
    plt.imshow(vaping_cloud)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(blended)
    plt.show()
    return genie, vaping_cloud, mask, blended


def blending_example2():
    johnny = read_image(relpath("externals/heresjohnny.jpg"), 2)
    hair = read_image(relpath("externals/hair.jpg"), 2)
    mask = np.rint(read_image(relpath("externals/hair_mask.jpg"), 1)).astype(bool)
    blended = np.zeros(hair.shape)
    for i in np.arange(3):
        blended[:,:,i] = pyramid_blending(hair[:,:, i], johnny[:,:,i], mask, 10, 3, 3)
    plt.subplot(2,2,1)
    plt.imshow(hair)
    plt.subplot(2,2,2)
    plt.imshow(johnny)
    plt.subplot(2, 2, 3)
    plt.imshow(mask, cmap='gray')
    plt.subplot(2, 2, 4)
    plt.imshow(blended)
    plt.show()
    return hair, johnny, mask, blended






