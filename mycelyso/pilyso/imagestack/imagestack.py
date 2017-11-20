# coding=utf-8

import numpy as np
import os
from os import getpid
from collections import namedtuple
from itertools import product

# noinspection PyCompatibility
from urllib.parse import urlparse, parse_qs


class Dimensions(object):

    @classmethod
    def all(cls):
        return cls.Dimension.__subclasses__()

    @classmethod
    def all_by_char(cls):
        return {d.char: d for d in cls.all()}

    @classmethod
    def by_char(cls, char):
        return cls.all_by_char()[char]

    # noinspection PyClassHasNoInit
    class Dimension(object):
        char = 'd'

    # noinspection PyClassHasNoInit
    class Time(Dimension):
        char = 't'

    # noinspection PyClassHasNoInit
    class PositionXY(Dimension):
        char = 'r'  # region

    Position = PositionXY

    # noinspection PyClassHasNoInit
    class PositionZ(Dimension):
        char = 'z'

    # noinspection PyClassHasNoInit
    class Channel(Dimension):
        char = 'c'

    # noinspection PyClassHasNoInit
    class Width(Dimension):
        char = 'w'

    # noinspection PyClassHasNoInit
    class Height(Dimension):
        char = 'h'


class UnsupportedImageError(NotImplementedError):
    pass


class ImageStackAPI(object):
    def get_data(self, what):
        pass

    def get_meta(self, what):
        return {}

    def init(self):
        pass

    def open(self, location, **kwargs):
        pass

    def notify_fork(self):
        pass


class ImageStackFilter(object):
    def filter(self, image):
        return image


class Image(np.ndarray):
    def __new__(cls, input_array, meta=None):
        obj = np.asarray(input_array).view(cls)
        obj.meta = meta
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return

        self.meta = getattr(obj, 'meta', None)

    def __array_wrap__(self, out_arr, context=None):
        # noinspection PyArgumentList
        return np.ndarray.__array_wrap__(self, out_arr, context)


class FloatFilter(ImageStackFilter):
    def filter(self, image):
        return image.astype(np.float32, copy=False)


class MinMaxFilter(ImageStackFilter):
    def filter(self, image):
        float_image = image.astype(np.float32, copy=True)
        float_image -= float_image.min()
        float_image /= float_image.max()
        return float_image


class UnwrapFilter(ImageStackFilter):
    def filter(self, image):
        return image.view(np.ndarray)


Metadata = namedtuple('Metadata', ['time', 'position', 'calibration'])
Position = namedtuple('Position', ['x', 'y', 'z'])


class ImageStack(ImageStackAPI):

    Metadata = Metadata
    Position = Position

    priority = 0
    schemes = ('file',)
    extensions = ('',)

    class ImageStackAccessBridge(object):
        def __init__(self, parent, purpose):
            self.parent = parent
            self.purpose = purpose

        def __getitem__(self, item):
            return self.parent.bridged_access(self.purpose, item)

    # noinspection PyProtectedMember
    class ImageStackView(object):
        @property
        def order(self):
            if self._order:
                return self._order
            else:
                return self.parent._all_dimensions[:]

        @order.setter
        def order(self, order):
            self._order = order

        def __init__(self, parent, pinned_indices=None, order=None, filters=None):
            self.parent = parent

            if pinned_indices:
                self.pinned_indices = pinned_indices
            else:
                self.pinned_indices = {}

            self.next_pinned_indices = None

            self._order = order

            self.meta = ImageStack.ImageStackAccessBridge(self, 'meta')
            self.data = ImageStack.ImageStackAccessBridge(self, 'data')

            if filters:
                self.filters = filters
            else:
                self.filters = []

        def copy(self):
            return self.__class__(self.parent, self.pinned_indices.copy(), self.order, self.filters)

        def filter(self, *what):
            self_copy = self.copy()
            if isinstance(what[0], tuple) or isinstance(what[0], list):
                what = what[0]

            for w in what:
                self_copy.filters.append(w)

            return self_copy

        def pin_indices(self, item):
            self_copy = self.copy()

            self_copy.begin()

            if isinstance(item, tuple):
                for n, i in enumerate(item):
                    self_copy.pin_index(n, i)
            elif isinstance(item, slice):
                self_copy.pin_index(0, item)
            elif isinstance(item, int):
                self_copy.pin_index(0, item)
            else:
                raise IndexError('...')

            self_copy.commit()

            return self_copy

        def __getitem__(self, item):
            self_copy = self.pin_indices(item)

            if self_copy.is_fixed():
                filters = []
                for a_filter in self.filters:
                    if issubclass(a_filter, ImageStackFilter):
                        filter_instance = a_filter()
                        filters.append(filter_instance.filter)
                    else:
                        filters.append(a_filter)

                return self.parent.perform_get_data(self_copy.collect(), filters)
            else:
                return self_copy

        def collect(self):
            return {
                k: v * self.parent.get_subsampling(k)
                for k, v in self.pinned_indices.items() if k in self.parent._all_dimensions
            }

        def is_fixed(self):
            for item in self.pinned_indices.values():
                if type(item) != int:
                    return False
            return True

        def pin_index(self, index, to_what):
            n = 0
            item = None

            for item in self.order:
                try:
                    v = self.pinned_indices[item]
                except KeyError:
                    v = self.next_pinned_indices[item] = self.pinned_indices[item] = False

                if v is False or isinstance(v, range):
                    if index == n:
                        break
                    n += 1

            try:
                size = self.parent.size[item]
            except KeyError:
                size = 1

            to_what_raw = to_what

            if isinstance(to_what, slice):
                start, stop, step = to_what.start, to_what.stop, to_what.step
                if start is None:
                    start = 0
                if stop is None:
                    stop = size
                if step is None:
                    step = 1

                to_what = range(start, stop, step)

            if self.next_pinned_indices[item] is False:
                self.next_pinned_indices[item] = to_what
            else:
                self.next_pinned_indices[item] = self.next_pinned_indices[item][to_what_raw]

        def commit(self):
            self.pinned_indices = self.next_pinned_indices

        def begin(self):
            self.next_pinned_indices = self.pinned_indices.copy()

        def bridged_access(self, entry_point, what):
            if entry_point == 'meta':
                self_copy = self.pin_indices(what)

                if self_copy.is_fixed():
                    return self.parent.perform_get_meta(self_copy.collect())
                else:
                    return self_copy

            elif entry_point == 'data':
                return self.__getitem__(what)

        def view(self, *new_order):
            if isinstance(new_order[0], list):
                new_order = new_order[0]

            self_copy = self.copy()
            self_copy.order = list(new_order)
            return self_copy

        @property
        def dimensions(self):
            return [
                item
                for item in self.order
                if item not in self.pinned_indices or type(self.pinned_indices[item]) != int
            ]
            # return self.order

        @property
        def size(self):
            result = {}
            for k in self.order:
                try:
                    result[k] = self.parent._sizes[self.parent._all_dimensions.index(k)]
                except ValueError:
                    result[k] = 1
            return result

        @property
        def sizes(self):
            sizes = self.size
            return [sizes[k] for k in self.dimensions]

        def __len__(self):
            result = 1
            for size in self.sizes:
                result *= size
            return result

        @property
        def every_index(self):
            for tup in product(*(range(s) for s in self.sizes)):
                yield tup

        @property
        def every_dict(self):
            dim = self.dimensions
            for tup in self.every_index:
                yield dict(zip(dim, tup))

        @property
        def every_index_image(self):
            for tup in self.every_index:
                yield tup, self.__getitem__(tup)

        @property
        def every_dict_image(self):
            dim = self.dimensions
            for tup in self.every_index:
                yield dict(zip(dim, tup)), self.__getitem__(tup)

    def __new__(cls, *args, **kwargs):
        if cls != ImageStack:
            return object.__new__(cls)

        what = args[0]
        to_open = urlparse(what)

        if os.name == 'nt':
            # windows peculiarity
            drive_prefix = "%s:" % (to_open.scheme,)
            if len(to_open.scheme) == 1 and os.path.isdir(drive_prefix):
                # noinspection PyProtectedMember
                to_open = to_open._replace(scheme='')._replace(path=drive_prefix + to_open.path)

        if to_open.scheme == '':
            # noinspection PyProtectedMember
            to_open = to_open._replace(scheme='file')

        def recursive_subclasses(class_, collector):
            collector.add(class_)
            for inner_class_ in class_.__subclasses__():
                recursive_subclasses(inner_class_, collector)
            return collector

        subclasses = recursive_subclasses(cls, set()) - {ImageStack}

        hits = list(sorted((
            class_ for class_ in subclasses
            if
            sum(to_open.scheme == scheme for scheme in class_.schemes) > 0 and
            sum(to_open.path.endswith(ext) for ext in class_.extensions) > 0),
            key=lambda c: c.priority))

        parameters = {k: (v if len(v) > 1 else v[0]) for k, v in parse_qs(to_open.query).items()}

        for hit in hits:
            try:
                result = hit()
                result.uri = what
                result.parameters = parameters
                result.parse_parameters_before()
                result.open(to_open, **parameters)
                result.parse_parameters_after()
                return result
            except UnsupportedImageError:
                pass

        raise RuntimeError('No suitable image plugin found.')

    __init___called = False

    # noinspection PyUnusedLocal
    def __init__(self, *args, **kwargs):
        if self.__init___called:
            return

        self.parent = self

        self.pid = getpid()

        self.__init___called = True
        self._sizes = []
        self._all_dimensions = []

        self.parameters = {}
        self.subsampling = {}

        self.root_view = self.__class__.ImageStackView(self)

        self.init()

    def parse_parameters_before(self):
        self.subsampling = {}
        dimensions = Dimensions.all_by_char()
        for param, value in self.parameters.items():
            if param.startswith('subsample_'):
                d = param.replace('subsample_', '')
                if d in dimensions:
                    self.subsampling[dimensions[d]] = int(value)

    def parse_parameters_after(self):
        dimensions = Dimensions.all_by_char()
        for param, value in self.parameters.items():
            if param in dimensions:
                dim = dimensions[param]
                self.root_view.pinned_indices[dim] = int(value)
                self.root_view = self.root_view.view([_d for _d in self.root_view.order if _d != dim])

    def get_subsampling(self, dim):
        try:
            return self.subsampling[dim]
        except KeyError:
            return 1

    def __getitem__(self, item):
        return self.root_view.__getitem__(item)

    def __getattr__(self, item):
        if hasattr(self.root_view, item):
            return getattr(self.root_view, item)

    def __len__(self):
        return self.root_view.__len__()

    def set_dimensions_and_sizes(self, dimensions, sizes):
        self._real_dimensions = dimensions
        self._real_sizes = sizes

        self._all_dimensions = dimensions
        self._sizes = [s // self.get_subsampling(dim) for dim, s in zip(dimensions, sizes)]

    def perform_get_data(self, what, filters=None):
        self.check_fork()
        data = self.get_data(what)

        xy_subsample = [self.parent.get_subsampling(Dimensions.Width), self.parent.get_subsampling(Dimensions.Height)]
        if xy_subsample[0] != xy_subsample[1]:
            raise RuntimeError('Asymmetric subsampling for width/height currently unsupported!')

        xy_subsample = xy_subsample[0]
        if xy_subsample != 1:
            data = data[::xy_subsample, ::xy_subsample]

        data = data.view(Image)

        data.meta = self.perform_get_meta(what)

        if filters:
            for a_callable_filter in filters:
                data = a_callable_filter(data)

        return data

    def perform_get_meta(self, what):
        self.check_fork()
        meta = self.get_meta(what)

        if isinstance(meta, Metadata):
            # time_offset allows to shift all times by an offset,
            # eg. when multiple files belong to one experiment, but all start at 0 albeit consecutive
            time_offset_str = 'time_offset'
            if time_offset_str in self.parameters:
                meta = meta._replace(time=meta.time + float(self.parameters[time_offset_str]))

        return meta

    def check_fork(self):
        if self.pid != getpid():
            self.notify_fork()
        self.pid = getpid()

