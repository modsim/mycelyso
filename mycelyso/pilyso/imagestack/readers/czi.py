"""
The CZI reader module implements an ImageStack able to open CZI (Carl Zeiss Imaging) image stacks,
by using the czifile module.
"""

from __future__ import division, unicode_literals, print_function

from ..imagestack import ImageStack, Dimensions

from xml.etree import cElementTree as ElementTree

import warnings

from czifile import CziFile


def _is_dimension_oi(d):
    return d in {'T', 'C', 'Z', 'S'}


def _dim_shape_to_dict(dim, shape):
    if not isinstance(dim, str):
        dim = dim.decode()
    return dict(zip(list(dim), shape))


def _get_subblock_position(subblock):
    return _dim_shape_to_dict(subblock.axes, subblock.start)


def _get_subblock_identifier(subblock):
    return _normalize(_get_subblock_position(subblock))


def _get_image_from_subblock(subblock):
    return subblock.data_segment().data().reshape([s for s in subblock.stored_shape if s != 1])


def _normalize(d):
    return tuple([(k, v,) for k, v in sorted(d.items()) if _is_dimension_oi(k)])


def _only_existing_dim(reference, test):
    test = test.copy()
    for k in list(test.keys()):
        if k not in reference:
            del test[k]
    return test


class CziImageStack(ImageStack):
    extensions = ('.czi',)

    priority = 500

    def open(self, location, **kwargs):

        with warnings.catch_warnings():
            warnings.simplefilter('ignore')
            self.czi = CziFile(location.path)

        self.frames = {
            _get_subblock_identifier(subblock): subblock for subblock in self.czi.filtered_subblock_directory
            }

        self.size = _dim_shape_to_dict(self.czi.axes, self.czi.shape)

        self.metadata = self.czi.metadata(raw=False)

        scaling = self.metadata['ImageDocument']['Metadata']['Scaling']['Items']['Distance']

        # /ImageDocument
        calibration_x = float(
            next(iter([scale for scale in scaling if scale['Id'] == 'X']))['Value']
        ) * 1E6

        calibration_y = float(
            next(iter([scale for scale in scaling if scale['Id'] == 'Y']))['Value']
        ) * 1E6

        assert calibration_x == calibration_y

        self.calibration = calibration_x

        timestamps = None

        for entry in self.czi.attachment_directory:
            if entry.name == 'TimeStamps':
                timestamps = entry.data_segment().data()
                break

        self.timestamps = timestamps

        positions = []

        for scene in sorted(self.metadata['ImageDocument'
                            ]['Metadata']['Information']['Image']['Dimensions']['S']['Scenes']['Scene'],
                            key=lambda scene_: int(scene_['Index'])):
            x, y = scene['CenterPosition'].split(',')
            positions.append((float(x), float(y)))

        self.positions = positions

        self.set_dimensions_and_sizes(
            [Dimensions.Time, Dimensions.PositionXY, Dimensions.PositionZ, Dimensions.Channel],
            [self.size.get('T', 1), self.size.get('S', 1), 1, self.size.get('C', 1)]
        )

    # noinspection PyProtectedMember
    def notify_fork(self):
        self.czi._fh.close()
        self.czi._fh.open()

    def get_data(self, what):
        channel = what.get(Dimensions.Channel, 0)
        position = what.get(Dimensions.PositionXY, 0)
        time = what.get(Dimensions.Time, 0)

        return _get_image_from_subblock(self.frames[_normalize(
            _only_existing_dim(self.size, dict(C=channel, S=position, T=time))
        )])

    def get_meta(self, what):
        try:
            time = float(self.timestamps[what[Dimensions.Time]])
        except TypeError:
            time = what[Dimensions.Time]

        try:
            position = (
                self.positions[what[Dimensions.PositionXY]][0],
                self.positions[what[Dimensions.PositionXY]][1],
                0.0,
            )
        except KeyError:
            position = (float('nan'), float('nan'), 0.0)

        meta = self.__class__.Metadata(
            time=time,
            position=position,
            calibration=self.calibration)

        return meta

