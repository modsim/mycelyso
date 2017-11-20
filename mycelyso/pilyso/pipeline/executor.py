# -*- coding: utf-8 -*-
"""
The executor submodule contains the PipelineExecutor, which runs an image processing pipeline.
"""

import gc
import logging
try:
    import tqdm
except ImportError:
    tqdm = None
from itertools import product
from collections import OrderedDict

from copy import deepcopy


from ..misc.processpool import cpu_count, SimpleProcessPool, InProcessFakePool, WrappedException
from .pipeline import PipelineEnvironment


class Collected(object):
    pass


class Every(object):
    pass


class Skip(RuntimeWarning):
    meta = None

    def __init__(self, meta):
        self.meta = meta


class NotDispatchedYet(object):
    pass


exception_debugging = False

singleton_class_mapper_local_cache = {}


def singleton_class_mapper(class_, what, args, kwargs):
    try:
        if class_ not in singleton_class_mapper_local_cache:
            singleton_class_mapper_local_cache[class_] = class_.__new__(class_)

        result = getattr(singleton_class_mapper_local_cache[class_], what)(*args, **kwargs)
        gc.collect()

        return result
    except Exception as _:
        raise


def get_progress_bar(n):
    if tqdm:
        return iter(tqdm.tqdm(range(n)))
    else:
        return iter(range(n))


# noinspection PyMethodMayBeStatic
class PipelineExecutor(object):
    wait = 0.01

    multiprocessing = True

    def __init__(self):
        if self.multiprocessing:
            if self.multiprocessing is True:
                self.multiprocessing = cpu_count()

    def set_workload(self, meta_tuple, item_counts, processing_order):
        self.meta_tuple = meta_tuple
        self.item_counts = item_counts
        self.processing_order = processing_order

    def initialize(self, pec, *args, **kwargs):

        self.log = logging.getLogger(__name__)
        self.pec = pec

        complete_args = (self.pec, '__init__', args, kwargs,)

        self.complete_args = complete_args

        singleton_class_mapper(*complete_args)

        if self.multiprocessing:
            self.pool = SimpleProcessPool(
                processes=self.multiprocessing,
                initializer=singleton_class_mapper,
                initargs=complete_args,
                future_timeout=30.0 * 60,  # five minute timeout, only works with the self-written pool
            )
        else:
            self.pool = InProcessFakePool()

    def set_progress_total(self, l):
        self.progress_indicator = get_progress_bar(l)

    def progress_tick(self):
        try:
            next(self.progress_indicator)
        except StopIteration:
            pass

    # noinspection PyUnusedLocal
    def in_cache(self, token):
        return False

    # noinspection PyUnusedLocal
    def get_cache(self, token):
        pass

    # noinspection PyUnusedLocal
    def set_cache(self, token, result):
        if result is None:
            return

    def skip_callback(self, op, skipped):
        self.log.info("Operation %r caused Skip for %r.", op, skipped)

    def run(self):
        meta_tuple = self.meta_tuple

        if getattr(self, 'cache', False) is False:
            self.cache = False

        sort_order = [index for index, _ in sorted(enumerate(self.processing_order), key=lambda p: p[0])]

        def prepare_steps(step, replace):
            return list(meta_tuple(*t) for t in sorted(product(*[
                self.item_counts[num] if value == replace else [value] for num, value in
                enumerate(step)
                ]), key=lambda t: [t[i] for i in sort_order]))

        todo = OrderedDict()

        reverse_todo = {}
        results = {}

        mapping = {}
        reverse_mapping = {}

        steps = singleton_class_mapper(self.pec, 'get_step_keys', (), {})

        for step in steps:
            order = prepare_steps(step, Every)
            reverse_todo.update({k: step for k in order})
            for k in order:
                todo[k] = NotDispatchedYet
            deps = {t: set(prepare_steps(t, Collected)) for t in order}
            mapping.update(deps)
            for key, value in deps.items():
                for k in value:
                    if k not in reverse_mapping:
                        reverse_mapping[k] = {key}
                    else:
                        reverse_mapping[k] |= {key}

        mapping_copy = deepcopy(mapping)

        def is_concrete(t):
            for n in t:
                if n is Collected or n is Every:
                    return False
            return True

        # initial_length = len(todo)

        self.set_progress_total(len(todo))

        check = OrderedDict()

        cache_originated = set()

        invalidated = set()

        concrete_counter, non_concrete_counter = 0, 0

        while len(todo) > 0 or len(check) > 0:
            for op in list(todo.keys()):

                result = None

                if op not in invalidated:

                    parameter_dict = {'meta': op}

                    if is_concrete(op):
                        # we are talking about a definite point, that is one that is not dependent on others
                        concrete_counter += 1
                        priority = 1 * concrete_counter
                    elif len(mapping[op]) != 0:
                        continue
                    else:
                        collected = OrderedDict()
                        for fetch in sorted(mapping_copy[op], key=lambda t: [t[i] for i in sort_order]):
                            collected[fetch] = results[fetch]
                        parameter_dict[PipelineEnvironment.KEY_COLLECTED] = collected
                        non_concrete_counter += 1
                        priority = -1 * non_concrete_counter

                    token = (reverse_todo[op], op,)
                    if self.in_cache(token):
                        cache_originated.add(op)
                        raise RuntimeError('TODO')  # TODO
                        # result = self.pool.advanced_apply(
                        #     command=singleton_class_mapper,
                        #     args=(self.__class__, 'get_cache', (token,), {},),
                        #     priority=priority
                        # )

                    else:
                        result = self.pool.advanced_apply(
                            singleton_class_mapper,
                            args=(self.pec, 'dispatch', (reverse_todo[op],), parameter_dict,),
                            priority=priority
                        )

                results[op] = result

                check[op] = True

                del todo[op]

            for op in list(check.keys()):
                result = results[op]

                modified = False

                if op in invalidated:
                    if getattr(result, 'fail', False) and callable(result.fail):
                        result.fail()
                        modified = True
                else:

                    if self.wait:
                        result.wait(self.wait)

                    if result.ready():
                        try:
                            result = result.get()

                            if op not in cache_originated:
                                token = (reverse_todo[op], op,)
                                # so far, solely accessing (write) the cache from
                                # one process should mitigate locking issues
                                self.set_cache(token, result)

                        except WrappedException as ee:
                            e = ee.exception
                            if type(e) == Skip:
                                old_invalid = invalidated.copy()

                                def _add_to_invalid(what):
                                    if what not in invalidated:
                                        invalidated.add(what)
                                        if what in mapping_copy:
                                            for item in mapping_copy[what]:
                                                _add_to_invalid(item)

                                _add_to_invalid(e.meta)

                                new_invalid = invalidated - old_invalid

                                self.skip_callback(op, new_invalid)

                            else:
                                if exception_debugging:
                                    raise

                                self.log.exception("Exception occurred at op=%s: %s",
                                                   repr(reverse_todo[op]) + ' ' + repr(op), ee.message)

                            result = None

                        modified = True

                if modified:
                    results[op] = result

                    if op in reverse_mapping:
                        for affected in reverse_mapping[op]:
                            mapping[affected] -= {op}

                    del check[op]
                    self.progress_tick()

        self.progress_tick()

        self.close()

    def close(self):
        self.pool.close()
