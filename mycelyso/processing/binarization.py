# -*- coding: utf-8 -*-
"""
The binarization module contains the binarization routine used to segment phase contrast images of 
mycelium networks into foreground and background.
"""

import warnings
import numpy as np

try:
    import numexpr
except ImportError:
    numexpr = None

from skimage.transform import integral_image
from skimage.feature import shape_index
from skimage.filters import rank


def mean_and_std(image, window_size=15):
    """
    Helper function returning mean and average images sped up using integral images / summed area tables.

    :param image: Input image
    :param window_size: Window size
    :return: tuple (mean, std)
    """

    enlarged = np.pad(image.astype(np.float64), window_size, 'reflect')

    def integral_image_and_squared(image):
        cv2 = None  # could be sped up using OpenCV
        if cv2:
            sums_, sqsums_ = cv2.integral2(image)
            return sums_[1:, 1:], sqsums_[1:, 1:]
        else:
            return integral_image(image), integral_image(image ** 2)

    ints, ints_sq = integral_image_and_squared(enlarged)

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


def normalize(image):
    """
    Normalizes an image to the range 0-1

    :param image:
    :return:
    """

    image = image.astype(np.float32)
    image -= image.min()
    image /= image.max()

    return image


# noinspection PyUnusedLocal
def sauvola(image, window_size=15, k=0.5, r=128, return_threshold=False, **kwargs):
    """
    Thresholding method as developed by [Sauvola1997]_.

    .. [Sauvola1997] Sauvola et al. (1997) Proc. Doc. Anal. Recog.
       DOI: `10.1109/ICDAR.1997.619831 <https://dx.doi.org/10.1109/ICDAR.1997.619831>`_

    :param image: Input image
    :param window_size: Window size
    :param k: k value
    :param r: r value
    :param return_threshold: Whether to return a binarization, or the actual threshold values
    :param kwargs: For compatibility
    :return:
    """

    mean, std = mean_and_std(image, window_size)

    if numexpr:
        threshold = numexpr.evaluate("mean * (1.0 + k * ((std / r) - 1.0))")
    else:
        threshold = mean * (1.0 + k * ((std / r) - 1.0))

    if return_threshold:
        return threshold
    else:
        return image < threshold


# noinspection PyUnusedLocal
def phansalkar(image, window_size=15, k=0.25, r=0.5, p=2.0, q=10.0, return_threshold=False, **kwargs):
    """
    Thresholding method as developed by [Phansalkar2011]_.

    .. [Phansalkar2011] Phansalkar et al. (2011) Proc. ICCSP
       DOI: `10.1109/ICCSP.2011.5739305 <https://dx.doi.org/10.1109/ICCSP.2011.5739305>`_

    :param image: Input image
    :param window_size: Window size
    :param k: k value
    :param r: r value
    :param p: p value
    :param q: q value
    :param return_threshold: Whether to return a binarization, or the actual threshold values
    :param kwargs: For compatibility
    :return:
    """

    original_image = image.copy()
    image = normalize(image)
    mean, std = mean_and_std(image, window_size)

    if numexpr:
        threshold = numexpr.evaluate("mean * (1.0 + p * exp(-q * mean) + k * ((std / r) - 1.0))")
    else:
        threshold = mean * (1.0 + p * np.exp(-q * mean) + k * ((std / r) - 1.0))

    if return_threshold:
        image_range = original_image.max()-original_image.min()
        return threshold * image_range + original_image.min()
    else:
        return image < threshold


# noinspection PyUnusedLocal,PyPep8Naming
def wolf(image, mask=None, window_size=15, a=0.5, return_threshold=False, **kwargs):
    """
    Thresholding method as developed by [Wolf2004]_.

    .. [Wolf2004] Wolf & Jolion (2004) Form. Pattern Anal. & App.
       DOI: `10.1007/s10044-003-0197-7 <https://dx.doi.org/10.1007/s10044-003-0197-7>`_

    :param image: Input image
    :param mask: Possible mask denoting a ROI
    :param window_size: Window size
    :param a: a value
    :param return_threshold: Whether to return a binarization, or the actual threshold values
    :param kwargs: For compatibility
    :return:
    """

    mean, std = mean_and_std(image, window_size)

    if mask is not None:
        M = image[mask].min()
        R = std[mask].max()
    else:
        M = image.min()
        R = std.max()

    if numexpr:
        threshold = numexpr.evaluate("(1-a) * mean + a*M + a*(std/R)*(mean - M)")
    else:
        threshold = (1-a) * mean + a*M + a*(std/R)*(mean - M)

    if return_threshold:
        return threshold
    else:
        return image < threshold


# noinspection PyUnusedLocal,PyUnusedLocal,PyPep8Naming
def feng(image, mask=None,
         window_size=15, window_size2=30,
         a1=0.12, gamma=2, k1=0.25, k2=0.04,
         return_threshold=False, **kwargs):
    """
    Thresholding method as developed by [Feng2004]_.

    .. [Feng2004] Fend & Tan (2004) IEICE Electronics Express
       DOI: `10.1587/elex.1.501 <https://dx.doi.org/10.1587/elex.1.501>`_

    :param image: Input image
    :param mask: Possible mask denoting a ROI
    :param window_size: Window size
    :param window_size2: Second window size
    :param a1: a1 value
    :param gamma: gamma value
    :param k1: k1 value
    :param k2: k2 value
    :param return_threshold: Whether to return a binarization, or the actual threshold values
    :param kwargs: For compatibility
    :return:
    """

    mean, std = mean_and_std(image, window_size)

    mean2, std2 = mean_and_std(image, window_size2)

    M = rank.minimum(image, np.ones((window_size, window_size)))
    # what exactly do they mean? maximum of window?
    Rs = rank.maximum(std2.astype(np.uint8), np.ones((window_size2, window_size2)))

    if numexpr:
        threshold = numexpr.evaluate("""(
        (1 - a1) * mean + (k1 * (std / Rs) ** gamma) * (std/Rs) * (mean - M) + (k2 * (std / Rs) ** gamma) * M
        )""")
    else:
        a2 = k1 * (std / Rs) ** gamma
        a3 = k2 * (std / Rs) ** gamma
        threshold = (1 - a1) * mean + a2 * (std/Rs) * (mean - M) + a3 * M

    if return_threshold:
        return threshold
    else:
        return image < threshold


# noinspection PyUnusedLocal
def nick(image, window_size=15, k=-0.1, return_threshold=False, **kwargs):
    """
    Thresholding method as developed by [Khurshid2009]_.

    .. [Khurshid2009] Khurshid et al. (2009) Proc. SPIE
       DOI: `10.1117/12.805827 <https://dx.doi.org/10.1117/12.805827>`_

    :param image: Input image
    :param window_size: Window size
    :param k: k value
    :param return_threshold: Whether to return a binarization, or the actual threshold values
    :param kwargs: For compatibility
    :return:
    """
    mean, std = mean_and_std(image, window_size)

    if numexpr:
        image_size = image.size
        image_sq_sum = (image**2).sum()
        threshold = numexpr.evaluate("mean + k * sqrt((image_sq_sum - mean**2)/image_size)")
    else:
        threshold = mean + k * np.sqrt(((image**2).sum() - mean**2)/image.size)

    if return_threshold:
        return threshold
    else:
        return image < threshold


# noinspection PyUnusedLocal
def bataineh(image, mask=None, window_size=15, return_threshold=False, **kwargs):
    """
    Thresholding method as developed by [Bataineh2011a]_.

    .. [Bataineh2011a] Bataineh et al. (2011) Pattern Recognit. Lett.
       DOI: `10.1016/j.patrec.2011.08.001 <https://dx.doi.org/10.1016/j.patrec.2011.08.001>`_

    :param image: Input image
    :param mask: Possible mask denoting a ROI
    :param window_size: Window size
    :param return_threshold: Whether to return a binarization, or the actual threshold values
    :param kwargs: For compatibility
    :return:
    """

    mean, std = mean_and_std(image, window_size)
    if mask is not None:
        adaptive_stddev = (std - std[mask].min()) / (std[mask].max() - std[mask].min())
        image_mean = image[mask].mean()
    else:
        adaptive_stddev = (std - std.min()) / (std.max() - std.min())
        image_mean = image.mean()

    if numexpr:
        threshold = numexpr.evaluate("mean - ((mean**2.0 - std) / ((image_mean + std) * (std + adaptive_stddev)))")
    else:
        threshold = mean - ((mean**2.0 - std) / ((image_mean + std) * (std + adaptive_stddev)))

    if return_threshold:
        return threshold
    else:
        return image < threshold


# I had highly optimized versions of sub functions of these, mainly
# using cv2, however to keep installation easy, I've replaced them by slower scikit-image etc. functions

# noinspection PyUnusedLocal
def experimental_thresholding(image, mask=None, window_size=15,
                              gaussian_sigma=3.0, shift=0.2, target=-0.5, quotient=1.2,
                              return_threshold=False, **kwargs):
    """
    A novel thresholding method basing upon the shape index as defined by [Koenderink1992]_, and [Bataineh2011]_
    automatic adaptive thresholding. The method is due to be explained in detail in the future.
    
    .. [Koenderink1992] Koenderink and van Doorn (1992) Image Vision Comput.
       DOI: `10.1016/0262-8856(92)90076-F <https://dx.doi.org/10.1016/0262-8856(92)90076-F>`_
    .. [Bataineh2011] Bataineh et al. (2011) Pattern Recognit. Lett.
       DOI: `10.1016/j.patrec.2011.08.001 <https://dx.doi.org/10.1016/j.patrec.2011.08.001>`_
      
    
    :param image: Input image
    :param mask: Possible mask denoting a ROI
    :param window_size: Window size
    :param gaussian_sigma: Sigma of the Gaussian used for smoothing
    :param shift: Shift parameter
    :param target: Target shape index parameter
    :param quotient: Quotient parameter
    :param return_threshold: Whether to return a binarization, or the actual threshold values
    :param kwargs: For compatibility
    :return: 
    """
    # novel method based upon shape index and Bataineh thresholding

    means, stddev = mean_and_std(image, window_size)

    with np.errstate(invalid='ignore'):
        sim = shape_index(image, gaussian_sigma)

    if mask is not None:
        adaptive_stddev = (stddev - stddev[mask].min()) / (stddev[mask].max() - stddev[mask].min())
        image_mean = image[mask].mean()
    else:
        adaptive_stddev = (stddev - stddev.min()) / (stddev.max() - stddev.min())
        image_mean = image.mean()

    if numexpr:
        threshold = numexpr.evaluate(
            "(exp((-(sim - target)**2)/quotient) + shift)*"
            "means*"
            "((image_mean + stddev) * (stddev + adaptive_stddev))/(means**2 - stddev)"
        )
    else:
        threshold = (
            (np.exp((-(sim - target)**2)/quotient) + shift) *
            means *
            ((image_mean + stddev) * (stddev + adaptive_stddev))/(means**2 - stddev)
        )

    if return_threshold:
        return threshold
    else:
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            return image < threshold
