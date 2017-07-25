# -*- coding: utf-8 -*-
"""
The binarization module contains the binarization routine used to segment phase contrast images of 
mycelium networks into foreground and background.
"""

import numpy as np

try:
    import numexpr
except ImportError:
    numexpr = None

from skimage.transform import integral_image
from skimage.feature import shape_index


def mean_and_std(image, window_size=15):
    """
    Helper function returning mean and average images sped up using integral images / summed area tables.
    
    :param image: Input image 
    :param window_size: Window size
    :return: tuple (mean, std)
    """
    enlarged = np.zeros((image.shape[0] + 2 * window_size, image.shape[1] + 2 * window_size), np.double)

    enlarged[window_size:-window_size, window_size:-window_size] = image
    enlarged[0:window_size] = enlarged[window_size + 1, :]
    enlarged[-window_size:] = enlarged[-window_size - 1, :]

    for n in range(window_size):
        enlarged[:, n] = enlarged[:, window_size]
        enlarged[:, -n] = enlarged[:, -window_size - 1]

    ints, ints_sq = integral_image(enlarged), integral_image(enlarged**2)

    def calculate_sums(mat):
        a = mat[:-2 * window_size, :-2 * window_size]
        b = mat[2 * window_size:, 2 * window_size:]
        c = mat[:-2 * window_size, 2 * window_size:]
        d = mat[2 * window_size:, :-2 * window_size]
        if numexpr:
            return numexpr.evaluate("(a + b) - (c + d)").astype(np.float32)
        else:
            return (a + b) - (c + d)

    sums = calculate_sums(ints)
    sums_squared = calculate_sums(ints_sq)

    area = (2.0 * window_size + 1) ** 2

    mean = sums / area

    if numexpr:
        std = numexpr.evaluate("sqrt(sums_squared / area - mean ** 2)")
    else:
        std = np.sqrt(sums_squared / area - mean ** 2)

    return mean, std


# I had highly optimized versions of sub functions of these, mainly
# using cv2, however to keep installation easy, I've replaced them by slower scikit-image etc. functions

def experimental_thresholding(image, window_size=15, gaussian_sigma=3.0, shift=0.2, target=-0.5, quotient=1.2):
    """
    A novel thresholding method basing upon the shape index as defined by [Koenderink1992]_, and [Bataineh2011]_
    automatic adaptive thresholding. The method is due to be explained in detail in the future.
    
    .. [Koenderink1992] Koenderink and van Doorn (1992) Image Vision Comput.
       `10.1016/0262-8856(92)90076-F <https://dx.doi.org/10.1016/0262-8856(92)90076-F>`_
    .. [Bataineh2011] Bataineh et al. (2011) Pattern Recognit. Lett.
       `10.1016/j.patrec.2011.08.001 <https://dx.doi.org/10.1016/j.patrec.2011.08.001>`_
      
    
    :param image: Input image
    :param window_size: Window size
    :param gaussian_sigma: Sigma of the Gaussian used for smoothing
    :param shift: Shift parameter
    :param target: Target shape index parameter
    :param quotient: Quotient parameter
    :return: 
    """
    # novel method based upon shape index and Bataineh thresholding

    means, stddev = mean_and_std(image, window_size)

    with np.errstate(invalid='ignore'):
        sim = shape_index(image, gaussian_sigma)

    stddev_min, stddev_max = stddev.min(), stddev.max()
    stddev_delta = stddev_max - stddev_min
    image_mean = image.mean()

    if numexpr:
        return numexpr.evaluate(
            "image < "
            "(exp((-(sim - target)**2)/quotient) + shift)*"
            "means*"
            "((image_mean + stddev) * (stddev + ((stddev - stddev_min) / stddev_delta)))/(means**2 - stddev)"
        )
    else:
        return image < \
               (np.exp((-(sim - target)**2)/quotient) + shift) * \
               means * \
               ((image_mean + stddev) * (stddev + ((stddev - stddev_min) / stddev_delta)))/(means**2 - stddev)

