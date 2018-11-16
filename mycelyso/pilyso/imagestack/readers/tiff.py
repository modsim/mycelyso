# -*- coding: utf-8 -*-
"""
The TIFF reader module implements an ImageStack able to open TIFF image stacks, by using the tifffile module.
"""

from ..imagestack import ImageStack, Dimensions

import warnings

with warnings.catch_warnings():
    warnings.simplefilter('ignore')
    from tifffile import TiffFile


class TiffImageStack(ImageStack):
    extensions = ('.tif', '.tiff',)

    priority = 1000

    def open(self, location, **kwargs):
        self.tiff = TiffFile(location.path, movie=True)

        self.set_dimensions_and_sizes([Dimensions.Time], [len(self.tiff.pages)])

    # noinspection PyProtectedMember
    def notify_fork(self):
        self.tiff._fh.close()
        self.tiff._fh.open()

    def get_data(self, what):
        return self.tiff.pages[what[Dimensions.Time]].asarray()

    def get_meta(self, what):
        try:
            calibration = float(self.parameters['calibration'])
        except KeyError:
            try:
                tags = self.tiff.pages[0].tags
                assert tags['ResolutionUnit'].value == 1  # 1 == µm?
                x_resolution, y_resolution = tags['XResolution'].value, tags['YResolution'].value
                assert x_resolution == y_resolution
                calibration = x_resolution[1] / x_resolution[0]
            except (KeyError, AssertionError):
                calibration = 1.0

        try:
            interval = float(self.parameters['interval'])
        except KeyError:
            interval = 1.0

        position = self.__class__.Position(x=0.0, y=0.0, z=0.0)
        meta = self.__class__.Metadata(
            time=interval * what[Dimensions.Time],
            position=position,
            calibration=calibration
        )
        return meta

