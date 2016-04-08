# -*- coding: utf-8 -*-
"""
documentation
"""

import numpy
import numexpr

from cv2 import GaussianBlur
from pilyso.processing.thresholding import means_and_stddev
from pilyso.processing.hessian import shapeindexmap


# noinspection PyUnusedLocal
def experimental_thresholding(image, window_size=15, gaussian_radius=3.0, shift=0.2, target=-0.5, quotient=1.2):
    #novel method based upon shape index and Bataineh thresholding

    float_image = image.astype(numpy.float32)
    means, stddev = means_and_stddev(image, window_size)
    sim = shapeindexmap(GaussianBlur(float_image, (-1, -1), gaussian_radius))
    sim[~numpy.isfinite(sim)] = 0.0

    stddev_min, stddev_max = stddev.min(), stddev.max()
    stddev_delta = stddev_max - stddev_min
    image_mean = image.mean()

    return numexpr.evaluate(
        "image < "
        "(exp((-(sim - target)**2)/quotient) + shift)*"
        "means*"
        "((image_mean + stddev) * (stddev + ((stddev - stddev_min) / stddev_delta)))/(means**2 - stddev)"
    )
