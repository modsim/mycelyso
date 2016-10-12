# -*- coding: utf-8 -*-
"""
documentation
"""

import numpy

try:
    import numexpr
except ImportError:
    numexpr = None

from skimage.transform import integral_image
try:
    from skimage.feature import shape_index
except ImportError:
    from skimage.feature.corner import hessian_matrix, hessian_matrix_eigvals
    import numpy as np
    # backport from skimage 0.13dev
    # TODO keep in mind they plan to change the hessian matrix ordering!

    def shape_index(image, sigma=1, mode='constant', cval=0):
        """Compute the shape index.
        The shape index, as defined by Koenderink & van Doorn [1]_, is a
        single valued measure of local curvature, assuming the image as a 3D plane
        with intensities representing heights.
        It is derived from the eigen values of the Hessian, and its
        value ranges from -1 to 1 (and is undefined (=NaN) in *flat* regions),
        with following ranges representing following shapes:
        .. table:: Ranges of the shape index and corresponding shapes.
          ===================  =============
          Interval (s in ...)  Shape
          ===================  =============
          [  -1, -7/8)         Spherical cup
          [-7/8, -5/8)         Through
          [-5/8, -3/8)         Rut
          [-3/8, -1/8)         Saddle rut
          [-1/8, +1/8)         Saddle
          [+1/8, +3/8)         Saddle ridge
          [+3/8, +5/8)         Ridge
          [+5/8, +7/8)         Dome
          [+7/8,   +1]         Spherical cap
          ===================  =============
        Parameters
        ----------
        image : ndarray
            Input image.
        sigma : float, optional
            Standard deviation used for the Gaussian kernel, which is used for
            smoothing the input data before Hessian eigen value calculation.
        mode : {'constant', 'reflect', 'wrap', 'nearest', 'mirror'}, optional
            How to handle values outside the image borders.
        cval : float, optional
            Used in conjunction with mode 'constant', the value outside
            the image boundaries.
        Returns
        -------
        s : ndarray
            Shape index
        References
        ----------
        .. [1] Koenderink, J. J. & van Doorn, A. J.,
               "Surface shape and curvature scales",
               Image and Vision Computing, 1992, 10, 557-564.
               DOI:10.1016/0262-8856(92)90076-F
        Examples
        --------
        >>> from skimage.feature import shape_index
        >>> square = np.zeros((5, 5))
        >>> square[2, 2] = 4
        >>> s = shape_index(square, sigma=0.1)
        >>> s
        array([[ nan,  nan, -0.5,  nan,  nan],
               [ nan, -0. ,  nan, -0. ,  nan],
               [-0.5,  nan, -1. ,  nan, -0.5],
               [ nan, -0. ,  nan, -0. ,  nan],
               [ nan,  nan, -0.5,  nan,  nan]])
        """

        Hxx, Hxy, Hyy = hessian_matrix(image, sigma=sigma, mode=mode, cval=cval)
        l1, l2 = hessian_matrix_eigvals(Hxx, Hxy, Hyy)

        return (2.0 / np.pi) * np.arctan((l2 + l1) / (l2 - l1))


def mean_and_std(image, window_size=15):
    enlarged = numpy.zeros((image.shape[0] + 2 * window_size, image.shape[1] + 2 * window_size), numpy.double)

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
            return numexpr.evaluate("(a + b) - (c + d)").astype(numpy.float32)
        else:
            return (a + b) - (c + d)

    sums = calculate_sums(ints)
    sums_squared = calculate_sums(ints_sq)

    area = (2.0 * window_size + 1) ** 2

    mean = sums / area

    if numexpr:
        std = numexpr.evaluate("sqrt(sums_squared / area - mean ** 2)")
    else:
        std = numpy.sqrt(sums_squared / area - mean ** 2)

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
               (numpy.exp((-(sim - target)**2)/quotient) + shift) * \
               means * \
               ((image_mean + stddev) * (stddev + ((stddev - stddev_min) / stddev_delta)))/(means**2 - stddev)

