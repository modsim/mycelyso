# -*- coding: utf-8 -*-
"""
The ND2 reader module implements an ImageStack able to open Nikon ND2 images, using the nd2file module.
"""

from ..imagestack import ImageStack, Dimensions

from nd2file import ND2MultiDim


class ND2ImageStack(ImageStack):
    extensions = ('.nd2',)

    priority = 500

    def open(self, location, **kwargs):
        self.nd2 = ND2MultiDim(location.path)

        self.set_dimensions_and_sizes(
            [Dimensions.Time, Dimensions.PositionXY, Dimensions.PositionZ, Dimensions.Channel],
            [int(self.nd2.timepointcount), int(self.nd2.multipointcount), 1, int(self.nd2.channels)]
        )

    def get_data(self, what):

        image = self.nd2.image(multipoint=what[Dimensions.PositionXY], timepoint=what[Dimensions.Time])

        if len(image.shape) == 3:
            try:
                channel = what[Dimensions.Channel]
            except KeyError:
                channel = 0

            image = image[:, :, channel]

        return image

    def get_meta(self, what):

        nan = float('nan')

        pos_meta = {'x': nan, 'y': nan, 'z': nan}

        try:
            pos_meta.update(self.nd2.multipoints[what[Dimensions.PositionXY]])
        except KeyError:
            pass

        position = self.__class__.Position(x=float(pos_meta['x']), y=float(pos_meta['y']), z=float(pos_meta['z']))

        meta = self.__class__.Metadata(
            time=float(
                self.nd2.get_time(
                    self.nd2.calc_num(multipoint=what[Dimensions.PositionXY], timepoint=what[Dimensions.Time])
                )
            ),
            position=position,
            calibration=self.nd2.calibration)

        return meta

