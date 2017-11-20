# -*- coding: utf-8 -*-
"""
The steps file contains various generally reusable image processing steps.
"""

import hashlib
from base64 import b64encode

import numpy as np
import scipy.ndimage as ndi
from ..imagestack.imagestack import Dimensions
from ..application.application import Meta

from mfisp_boxdetection import find_box
from molyso.generic.rotation import find_rotation, rotate_image
from molyso.generic.registration import translation_2x1d, shift_image
from ..pipeline.executor import Skip, Collected
from ..misc.h5writer import CompressedObject


class Delete(object):
    pass


class Compress(object):
    pass


def set_result(**kwargs):
    def _inner(result):
        for k, v in kwargs.items():
            if v is Delete:
                del result[k]
            elif v is Compress:
                result[k] = CompressedObject(result[k])
            else:
                result[k] = v
        return result

    return _inner


# noinspection PyUnusedLocal
def image_source(ims, meta, image=None):
    return ims.view(Dimensions.Position, Dimensions.Time)[meta.pos, meta.t]


# noinspection PyUnusedLocal
def calculate_image_sha256_hash(image, image_sha256_hash=None):
    hasher = hashlib.sha256()
    hasher.update(image.tobytes())
    hash_value = b64encode(hasher.digest()).decode()
    return hash_value


def image_to_ndarray(image):
    return np.array(image)


# noinspection PyUnusedLocal
def pull_metadata_from_image(image, timepoint=None, position=None, calibration=None):
    return image, image.meta.time, image.meta.position, image.meta.calibration


_substract_start_frame_start_images = {}


# noinspection PyUnusedLocal
def substract_start_frame(meta, ims, reference_timepoint, image, subtracted_image=None):
    gaussian_blur_radius = 15.0

    if meta.pos not in _substract_start_frame_start_images:
        reference = image_source(ims, Meta(t=reference_timepoint, pos=meta.pos))
        blurred = ndi.gaussian_filter(reference, gaussian_blur_radius)
        _substract_start_frame_start_images[meta.pos] = blurred
    else:
        blurred = _substract_start_frame_start_images[meta.pos]

    image = image.astype(np.float32)

    image /= blurred

    image -= image.min()
    image /= image.max()

    return image


_box_cache_fft_cache = {}
_box_cache_boxes = {}
_box_cache_angles = {}


def _box_detection_get_parameters(ims, timepoint, pos):
    reference = image_source(ims, Meta(t=timepoint, pos=pos))
    angle = find_rotation(reference)
    reference = rotate_image(reference, angle)

    shift, fft_a = translation_2x1d(image_a=reference, image_b=reference, return_a=True)

    try:

        reference = reference.astype(np.float32)

        cleaned_reference = reference - ndi.uniform_filter(reference, 25)  # "box" blur
        cleaned_reference[cleaned_reference < 0] = 0
        cleaned_reference /= cleaned_reference.max()

        # qimshow(cleaned_reference)

        crop = find_box(cleaned_reference, throw=True, subsample=2)
        t, b, l, r = crop
        cleaned_reference[t, :] = 1
        cleaned_reference[b, :] = 1
        cleaned_reference[:, l] = 1
        cleaned_reference[:, r] = 1

    except RuntimeError:
        # noinspection PyCompatibility
        raise Skip(Meta(pos=pos, t=Collected)) from None

    return angle, fft_a, crop


# noinspection PyUnusedLocal
def box_detection(ims, image, meta, reference_timepoint, shift=None, crop=None, angle=None):
        # probably implement a voting scheme?

        if meta.pos not in _box_cache_boxes:
            _box_cache_angles[meta.pos], _box_cache_fft_cache[meta.pos], _box_cache_boxes[meta.pos] = \
                _box_detection_get_parameters(ims, reference_timepoint, meta.pos)

        angle = _box_cache_angles[meta.pos]
        image = rotate_image(image, angle)

        shift, = translation_2x1d(image_a=None, image_b=image, ffts_a=_box_cache_fft_cache[meta.pos])
        crop = _box_cache_boxes[meta.pos]

        return shift, crop, angle


def create_boxcrop_from_subtracted_image(subtracted_image, shift, angle, crop, result):
    result.shift_x = shift[0]
    result.shift_y = shift[1]

    result.crop_t, result.crop_b, result.crop_l, result.crop_r = crop
    t, b, l, r = crop

    subtracted_image = rotate_image(subtracted_image, angle)
    subtracted_image = shift_image(subtracted_image, shift, background='blank')

    box = subtracted_image[t:b, l:r]

    result.image = box

    return result


def rescale_image_to_uint8(image):
    image = image.astype(np.float32)
    image -= image.min()
    image /= image.max()

    image *= 255.0

    return image.astype(np.uint8)
