# -*- coding: utf-8 -*-
"""
The h5writer submodule contains the hdf5_output function, which accepts various datatypes (binary, image, tabular),
and adds them to a (possibly pre-existing) HDF5 file.
"""

from pandas import HDFStore, DataFrame
from tables.nodes import filenode
from tables import NodeError, Filters

from os.path import isfile
from time import sleep
from datetime import datetime
from os import remove, fdopen, open as low_level_open, O_CREAT, O_EXCL, O_WRONLY, getpid
from fnmatch import fnmatch

import numpy as np
import zlib
import bz2
try:
    import lzma
except ImportError:
    lzma = None

import pickle
import time

lock_debugging = False


def wait_for_lock_and_prepare_filename(base_filename, local_timeout):
    found_filename = False
    filename = base_filename
    lock_file = None

    while not found_filename:

        lock_file = '%s.lock' % (filename,)

        begin_time = datetime.now()

        while isfile(lock_file) or local_timeout == 0:
            # print(lock_file, isfile(lock_file), local_timeout)
            sleep(1.0)
            current_time = datetime.now()
            if (current_time - begin_time).total_seconds() > local_timeout:
                newfilename = '%s_%s.h5' % (base_filename, current_time.strftime("%Y%m%d%H%M%S_%f"),)
                print(
                    "WARNING: Waited %d seconds to acquire a lock for \"%s\", "
                    "but failed. Will now try to write data to new file \"%s\"." %
                    (local_timeout, filename, newfilename,)
                )
                filename = newfilename
                lock_file = '%s.lock' % (filename,)
                break

        if not isfile(lock_file):
            found_filename = True
    return filename, lock_file


def acquire_lock(lock_file):
    if lock_debugging:
        print("Process %d acquired lock %s." % (getpid(), lock_file))
    return fdopen(low_level_open(lock_file, O_CREAT | O_EXCL | O_WRONLY), 'w')


def release_lock(lock_file):
    if lock_debugging:
        print("Process %d released lock %s." % (getpid(), lock_file))
    try:
        remove(lock_file)
    except FileNotFoundError:
        pass


timeout = 5 * 60.0


def hdf5_node_name(s):
    s = s.\
        replace('/', '_').\
        replace(':', '_').\
        replace('\\', '_').\
        replace('.', '_').\
        replace('-', '_').\
        replace('?', '_').\
        replace('=', '_').\
        replace('&', '_').\
        replace(';', '_')

    if s[0:1] == '_':
        s = s[1:]

    return s


class CompressedObject(object):
    __slots__ = 'data', 'compression'

    debug = False
    default_compression = 'bz2'

    compressions = {
        'zlib': (lambda _: zlib.compress(_, 6), lambda _: zlib.decompress(_)),
        'bz2': (lambda _: bz2.compress(_, 9), lambda _: bz2.decompress(_)),
        'lzma': (lambda _: lzma.compress(_, check=lzma.CHECK_NONE), lambda _: lzma.decompress(_)),
    }

    def __init__(self, data, compression=None):
        before = time.time()
        if compression is None:
            compression = self.default_compression

        serialized = pickle.dumps(data, pickle.HIGHEST_PROTOCOL)
        before_len = len(serialized)

        assert compression in self.compressions

        compress, uncompress = self.compressions[compression]

        serialized = compress(serialized)

        self.compression = compression
        self.data = serialized

        del serialized
        del data

        after_len = len(self.data)
        after = time.time()

        if self.debug:
            print(
                "... serialized and compressed object using %s from %d to %d (%.3f) took %.3fs" % (
                    compression, before_len, after_len, after_len/before_len, (after-before)
                )
            )

    def get(self):
        compress, uncompress = self.compressions[self.compression]
        return pickle.loads(uncompress(self.data))


def return_or_uncompress(something):
    if isinstance(something, CompressedObject):
        return something.get()
    else:
        return something


def hdf5_output(_filename, immediate_prefix='', tabular_name='result_table'):
    def _inner_hdf5_output(meta, result):

        # noinspection PyProtectedMember
        meta_str = '_'.join(
            k + '_' + ('%09d' % v if type(v) == int else v.__name__)
            for k, v in sorted(meta._asdict().items(), key=lambda x: x[0])
        )

        prefix = '/results/'

        if immediate_prefix != '':
            prefix += immediate_prefix

        prefix += '/' + meta_str + '/'

        success = False

        local_timeout = timeout

        base_filename = _filename

        lock_file = None

        while not success:

            filename, lock_file = wait_for_lock_and_prepare_filename(base_filename, local_timeout)

            compression_type = 'zlib'
            compression_level = 6

            compression_filter = Filters(complib=compression_type, complevel=compression_level)

            try:
                # race conditions
                # open(lock_file, 'w+')
                # noinspection PyUnusedLocal
                with acquire_lock(lock_file) as lock:
                    store = HDFStore(filename, complevel=compression_level, complib=compression_type)

                    # noinspection PyProtectedMember
                    h5 = store._handle

                    # cache for palettes
                    # currently unused
                    # palette_written = {}

                    def store_image(h5path, name, data, upsample_binary=True):
                        h5path = h5path.replace('//', '/')
                        # hdf5 stores bitfields as well, but default 0,1 will be invisible on a fixed 0-255 palette ...
                        if data.dtype == bool and upsample_binary:
                            data = (data * 255).astype(np.uint8)
                        arr = h5.create_carray(h5path, name, obj=data, createparents=True, filters=compression_filter)
                        arr.attrs.CLASS = 'IMAGE'
                        arr.attrs.IMAGE_SUBCLASS = 'IMAGE_GRAYSCALE'
                        arr.attrs.IMAGE_VERSION = '1.2'

                    def store_data(h5path, name, data):
                        h5path = h5path.replace('//', '/')

                        h5path_splits = [x for x in h5path.split('/') if x != '']

                        for i in range(len(h5path_splits)):
                            try:
                                h5.create_group('/' + '/'.join(h5path_splits[:i]), h5path_splits[i])
                            except NodeError:
                                pass

                        f = filenode.new_node(h5, where=h5path, name=name, filters=compression_filter)
                        if type(data) == str:
                            data = data.encode('utf-8')
                        f.write(data)
                        f.close()

                    def store_table(name, data):
                        _frame = DataFrame(data)
                        store[name] = _frame  # .append(name, _frame, data_columns=_frame.columns)

                    image_counter = {}
                    data_counter = {}
                    table_counter = {}

                    def process_row(result_table_rows, m, row):
                        cresults = []

                        # noinspection PyProtectedMember
                        tmp = {('meta_' + mk): (mv if type(mv) == int else -1) for mk, mv in m._asdict().items()}

                        if type(result_table_rows) == list:
                            result_table_rows = {key: True for key in result_table_rows}

                        if '_plain' in result_table_rows:
                            for v in result_table_rows['_plain']:
                                result_table_rows[v] = True
                            del result_table_rows['_plain']

                        def is_wildcarded(s):
                            return '*' in s

                        for k, v in list(result_table_rows.items()):
                            if is_wildcarded(k):
                                del result_table_rows[k]

                                for row_key in row.keys():
                                    if fnmatch(row_key, k):
                                        result_table_rows[row_key] = v

                        for k, v in result_table_rows.items():
                            if v == 'table':
                                if k not in table_counter:
                                    table_counter[k] = 0

                                if k in row and len(row[k]) > 0:
                                    if type(row[k][0]) == list:
                                        # it's a list of lists
                                        # create a mapping table
                                        # point to the mapping table

                                        the_counter = table_counter[k]

                                        new_path = '/tables/_mapping_%s' % (k,)
                                        new_name = '%s_%09d' % (k, the_counter)

                                        tmp[k] = -1
                                        tmp['_mapping_%s' % k] = the_counter

                                        table_counter[k] += 1

                                        i_mapping = []

                                        for n, i_table in enumerate(row[k]):
                                            i_new_path = '/tables/_individual_%s' % (k,)
                                            i_new_name = '%s_%09d' % (k, table_counter[k])
                                            store_table(prefix + i_new_path + '/' + i_new_name, i_table)

                                            i_mapping.append({
                                                '_index': n,
                                                'individual_table': table_counter[k]
                                            })

                                            table_counter[k] += 1

                                        store_table(prefix + new_path + '/' + new_name, i_mapping)

                                        tmp[k] = table_counter[k]
                                        table_counter[k] += 1
                                    else:
                                        new_path = '/tables/%s' % (k,)
                                        new_name = '%s_%09d' % (k, table_counter[k])
                                        store_table(prefix + new_path + '/' + new_name, row[k])
                                        tmp[k] = table_counter[k]
                                        table_counter[k] += 1
                                else:
                                    tmp[k] = table_counter[k]
                                    table_counter[k] += 1
                            elif v == 'image':
                                if k not in image_counter:
                                    image_counter[k] = 0

                                if k in row:
                                    new_path = '/images/%s' % (k,)
                                    new_name = '%s_%09d' % (k, image_counter[k])
                                    store_image(prefix + new_path, new_name, return_or_uncompress(row[k]))
                                tmp[k] = image_counter[k]

                                image_counter[k] += 1
                            elif v == 'data':
                                if k not in data_counter:
                                    data_counter[k] = 0

                                if k in row:
                                    new_path = '/data/%s' % (k,)
                                    new_name = '%s_%09d' % (k, data_counter[k])
                                    store_data(prefix + new_path, new_name, return_or_uncompress(row[k]))
                                tmp[k] = data_counter[k]

                                data_counter[k] += 1
                            else:
                                if k in row:
                                    tmp[k] = row[k]
                                else:
                                    tmp[k] = float('nan')

                        cresults.append(tmp)
                        return cresults

                    if 'collected' in result:
                        collected = []
                        for m, row in result['collected'].items():
                            if tabular_name in row:
                                result_table_rows = row[tabular_name]
                                collected += process_row(result_table_rows, m, row)
                        store_table(prefix + 'result_table_collected', collected)

                    if tabular_name in result:
                        store_table(prefix + 'result_table', process_row(result[tabular_name], meta, result))

                    store.close()

                success = True
            except NodeError:
                print("NodeError Exception occurred while writing, " +
                      "apparently the file has already been used to store similar results.")
                # print("Leaving it LOCKED (remove manually!) and trying to write to another file!")
                local_timeout = 0
                release_lock(lock_file)

            except Exception as e:
                print("Exception occurred while writing results: " + repr(e))
                release_lock(lock_file)
                return
        release_lock(lock_file)

        return result

    return _inner_hdf5_output


# Storing other image data or paletted images

# from tables import Atom
# arr = store._handle.create_carray(h5path, name, Atom.from_dtype(data.dtype), data.shape, createparents=True)
# arr[:] = data[:]

# if data.dtype == bool:
    # arr.attrs.PALETTE = pal.object_id

# if False and True not in palette_written:
#
#     palette = np.zeros((256, 3,), dtype=np.uint8)
#
#     s = 256//3
#
#     for i in range(0, s):
#         palette[i+0*s, :] = [i, 0, 0]
#         palette[i+1*s, :] = [s+i, i, 0]
#         palette[i+2*s, :] = [s+i, s+i, i]
#
#     palette[0, :] = 0
#     palette[1, :] = 255
#
#     # pytables does not seem to support object references
#     # so that nice palette code is actually useless ...
#     pal = store._handle.create_array('/', 'palette', palette, createparents=True)
#     pal.attrs.CLASS = 'PALETTE'
#     pal.attrs.PAL_COLORMODEL = 'RGB'
#     pal.attrs.PAL_TYPE = 'STANDARD8'
#     pal.attrs.PAL_VERSION = '1.2'
#
#     palette_written[True] = True
# #else:
# #    pal = store._handle.get_node('/palette')
