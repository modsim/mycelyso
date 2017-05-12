# -*- coding: utf-8 -*-
"""
documentation
"""

import numpy as np

try:
    import numexpr
except ImportError:
    numexpr = None

from skimage.transform import integral_image
from skimage.feature import shape_index


def mean_and_std(image, window_size=15):
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

def experimental_thresholding(image, window_size=15, gaussian_radius=3.0, shift=0.2, target=-0.5, quotient=1.2):
    # novel method based upon shape index and Bataineh thresholding

    means, stddev = mean_and_std(image, window_size)
    sim = shape_index(image, gaussian_radius)

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

