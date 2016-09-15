# -*- coding: utf-8 -*-
"""
documentation
"""


def add_result(**kwargs):
    def _inner(result):
        result.update(kwargs)
        return result


def remove_result(**kwargs):
    def _inner(result):
        for k, v in kwargs.items():
            del result[k]
        return result
    return _inner


class Delete(object):
    pass


def set_result(**kwargs):
    def _inner(result):
        for k, v in kwargs.items():
            if v is Delete:
                del result[k]
            else:
                result[k] = v
        return result
    return _inner


def image_source(ims, meta, image=None):
    return ims.view(Dimensions.Position, Dimensions.Time)[meta.pos, meta.t]


def image_to_ndarray(image):
    return numpy.array(image)


def pull_metadata_from_image(image, timepoint=None, position=None, calibration=None):
    return image, image.meta.time, image.meta.position, image.meta.calibration


######


from ..imagestack.imagestack import Dimensions
from ..application.application import Meta

from boxdetection import find_box
from molyso.generic.rotation import find_rotation, rotate_image
from molyso.generic.registration import translation_2x1d, shift_image
from skimage import morphology
from ..processing.processing import blur_gaussian
import numpy


def skeletonize(binary, skeleton=None):
    return morphology.skeletonize(binary)


class substract_start_frame(object):

    gaussian_blur_radius = 15.0

    _start_images = {}

    def __call__(self, meta, ims, reference_timepoint, image, subtracted_image=None):
        if meta.pos not in self._start_images:
            reference = image_source(ims, Meta(t=reference_timepoint, pos=meta.pos)) # TODO proper sub pipeline
            #reference = self.embedded_pipeline([ImageSource], Meta(t=reference_timepoint, pos=meta.pos)).image
            blurred = blur_gaussian(reference, self.gaussian_blur_radius)
            self._start_images[meta.pos] = blurred
        else:
            blurred = self._start_images[meta.pos]

        image = image.astype(numpy.float32)

        image /= blurred

        image -= image.min()
        image /= image.max()

        return image


from ..processing.processing import blur_box
from ..pipeline.executor import Skip, Collected


class box_detector_cropper(object):

    _fft_cache = {}
    _boxes = {}
    _angles = {}

    def get_parameters(self, ims, timepoint, pos):
        reference = image_source(ims, Meta(t=timepoint, pos=pos))  # TODO proper sub pipeline
#        reference = self.embedded_pipeline([ImageSource], Meta(t=timepoint, pos=pos)).image
        angle = find_rotation(reference)
        reference = rotate_image(reference, angle)

        shift, fft_a = translation_2x1d(image_a=reference, image_b=reference, return_a=True)

        try:

            reference = reference.astype(numpy.float32)

            cleaned_reference = reference - blur_box(reference, 25)
            cleaned_reference[cleaned_reference < 0] = 0
            cleaned_reference /= cleaned_reference.max()

            #qimshow(cleaned_reference)

            crop = find_box(cleaned_reference, throw=True, subsample=2)
            t, b, l, r = crop
            cleaned_reference[t, :] = 1
            cleaned_reference[b, :] = 1
            cleaned_reference[:, l] = 1
            cleaned_reference[:, r] = 1

            #qimshow(cleaned_reference)

        except RuntimeError:
            raise Skip(Meta(pos=pos, t=Collected)) from None

        return angle, fft_a, crop

    def __call__(self, ims,  image, meta, reference_timepoint, shift=None, crop=None, angle=None):
        # probably implement a voting scheme?

        if meta.pos not in self._boxes:
            self._angles[meta.pos], self._fft_cache[meta.pos], self._boxes[meta.pos] =\
                self.get_parameters(ims, reference_timepoint, meta.pos)

        angle = self._angles[meta.pos]
        image = rotate_image(image, angle)

        shift, = translation_2x1d(image_a=None, image_b=image, ffts_a=self._fft_cache[meta.pos])
        crop = self._boxes[meta.pos]

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
    image = image.astype(numpy.float32)
    image -= image.min()
    image /= image.max()

    image *= 255.0

    return image.astype(numpy.uint8)

