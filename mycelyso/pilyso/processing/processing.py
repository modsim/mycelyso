# -*- coding: utf-8 -*-
"""
documentation
"""


import numpy

import cv2

from skimage.draw import circle

def draw_circles(image, coordinates, radius=1, color=1):
    circle_image = numpy.zeros((radius * 2, radius * 2), dtype=image.dtype)
    rr, cc = circle(radius-1, radius-1, radius)

    circle_image[rr, cc] = color

    for y, x in coordinates:
        ylo, yhi = max(0, y - radius), min(y + radius, image.shape[0])
        xlo, xhi = max(0, x - radius), min(x + radius, image.shape[1])

        image[ylo:yhi, xlo:xhi] += circle_image[:yhi-ylo, :xhi-xlo]

    return image


def _label_image_skimage(image, neighbors=8):
    return skimage.measure.label(image, connectivity=2 if neighbors == 8 else 1, background=0) + 1

def _label_image_cv2(image, neighbors=8):
    return cv2.connectedComponents(image.astype(numpy.uint8), connectivity=neighbors)[1]

label_image = _label_image_cv2




def draw_circles_only_if_join(image, coordinates, radius=1, color=1):
    labeled = label_image(image)
    circle_image = numpy.zeros((radius * 2, radius * 2), dtype=image.dtype)
    rr, cc = circle(radius-1, radius-1, radius)

    circle_image[rr, cc] = color

    binary_circle = circle_image.astype(bool)

    for y, x in coordinates:
        ylo, yhi = max(0, y - radius), min(y + radius, image.shape[0])
        xlo, xhi = max(0, x - radius), min(x + radius, image.shape[1])

        uniques = numpy.unique((labeled[ylo:yhi, xlo:xhi] * binary_circle[:yhi-ylo, :xhi-xlo]).ravel())
        if uniques.shape[0] > 2:
            image[ylo:yhi, xlo:xhi] += circle_image[:yhi-ylo, :xhi-xlo]

    return image


def fill_holes_smaller_than(image, size=100):
    image = image.astype(bool)
    labeled = label_image(~image)
    bins = numpy.bincount(labeled.ravel())

    pairs = numpy.c_[numpy.linspace(0, bins.shape[0] - 1, bins.shape[0]).astype(int), bins]

    bad = pairs[pairs[:, 1] < size][:, 0]

    return image | numpy.in1d(labeled.ravel(), bad).reshape(image.shape)


from skimage.measure import regionprops
def only_centerpoints(image):
    labeled = label_image(image)

    result = numpy.zeros_like(image)

    for region in regionprops(labeled):
        y, x = region.centroid
        result[y, x] = 1

    return result

def blur_gaussian(image, sigma=1.0):
    return cv2.GaussianBlur(image, ksize=(-1, -1), sigmaX=sigma)

def blur_box(image, width_x=1, width_y=None):
    if width_y is None:
        width_y = width_x
    return cv2.blur(image, (width_x, width_y))

def blur_and_threshold(image, sigma=1.0, threshold=0.5):
    image = (image * 1.0).astype(numpy.float32)
    image = blur_gaussian(image, sigma)
    return image > threshold

def merge(p, r=15):
    from cv2 import filter2D

    p = only_centerpoints(p)
    p = filter2D(p.astype(numpy.uint8), -1, numpy.ones((r,r)))
    p[p > 0] = 1
    p = only_centerpoints(p)

    return p


def qimshow(image, cmap='gray'):
    import matplotlib.pyplot
    fig = matplotlib.pyplot.figure()
    ax = fig.add_subplot(111)
    range = ax.imshow(image, cmap=cmap, interpolation='none')

    def _format_coords(x, y):
        try:
            y, x = int(y + 0.5), int(x + 0.5)
            if y < 0 or x < 0:
                raise IndexError
            value = image[y, x]
        except IndexError:
            value = float('nan')
        return 'x=%d y=%d value=%1.4f' % (x, y, value,)
    ax.format_coord = _format_coords
    matplotlib.pyplot.colorbar(range)
    matplotlib.pyplot.show()




def grow_points(i):
    from cv2 import filter2D
    return filter2D(i, -1, numpy.ones((15, 15)))

def _numpy_get_integral_image_and_squared(image):
    integral_image = lambda mat: mat.cumsum(axis=1).cumsum(axis=0)
    return integral_image(image), integral_image(image**2)

def _cv2_get_integral_image_and_squared(image):
    ints, intss = cv2.integral2(image, sdepth=cv2.CV_64F)
    return ints[1:, 1:], intss[1:, 1:]

get_integral_image_and_squared = _cv2_get_integral_image_and_squared




