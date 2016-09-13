# encoding: utf-8

import inspect

if getattr(inspect, 'signature'):
    def get_argnames_and_defaults(call):
        sig = inspect.signature(call)
        args = [para for para in sig.parameters]
        defaults = [para.default for para in sig.parameters.values() if para.default is not inspect._empty]
        return args, defaults

else:

    def get_argnames_and_defaults(call):
        # noinspection PyDeprecation
        argspec = inspect.getargspec(call)

        args = list(argspec.args)
        defaults = argspec.defaults if list(argspec.defaults) else []

        return args, defaults

from collections import OrderedDict


class NeatDict(dict):
    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]


KEY_COLLECTED = 'collected'
KEY_RESULT = 'result'


def reassemble_result(result, args, _call_return, ignore=None):
    if ignore is None:
        ignore = {}

    if type(_call_return) == dict and KEY_RESULT in _call_return:
        # if we get a dict back, we merge it with the ongoing result object
        result.update(_call_return)
    elif type(_call_return) == NeatDict:
        # if we get a neatdict back, we assume its the proper result object
        # and the pipeline step knew what it did ...
        # we continue with it as-is
        result = _call_return
    else:
        if type(_call_return) != tuple:
            _call_return = (_call_return,)

        for n, item in enumerate(reversed(_call_return)):
            k = args[-(n + 1)]
            if k == KEY_RESULT:
                if type(item) == dict or type(item) == NeatDict:
                    result.update(item)
            else:
                # do nothing if the parameter came from di
                if k not in ignore:
                    result[k] = item
    return result


def call(self, call, result, di=None):

    result = self.wrap_result(result)

    args, defaults = get_argnames_and_defaults(call)

    if args[0] == 'self':
        args = args[1:]

    non_default_parameters = len(args) - len(defaults)

    from_di = set()
    parameters = []

    for n, arg in enumerate(args):
        if arg == self.KEY_RESULT:
            parameters.append(result)
        elif arg in result:
            parameters.append(result[arg])
        else:
            if n >= non_default_parameters:
                parameters.append(defaults[n - non_default_parameters])
            else:
                try:
                    di_arg = self.di(arg)

                    from_di.add(arg)
                    parameters.append(di_arg)
                except KeyError:
                    # problem: pipeline step asks for a parameter we do not have
                    raise ValueError('[At %s]: Argument %r not in %r' % (repr(call), arg, result,))

    _call_return = call(*parameters)

    result = reassemble_result(result, args, _call_return, ignore=from_di)

    result = unwrap_result(result)

    return result


def wrap_result(result):

    result = NeatDict(result)

    result[KEY_RESULT] = True

    if KEY_COLLECTED in result:
        wrapped = OrderedDict()
        for k, v in result[KEY_COLLECTED].items():
            if v is None:
                #wrapped[k] = None
                pass
            else:
                wrapped[k] = NeatDict(v)
        result[self.KEY_COLLECTED] = wrapped

    return result


def unwrap_result(result):
    if KEY_COLLECTED in result and (result[KEY_COLLECTED] is not None):
        unwrapped = OrderedDict()
        for k, v in result[KEY_COLLECTED].items():
            unwrapped[k] = dict(v)
        result[KEY_COLLECTED] = unwrapped

    result = dict(result)

    if KEY_RESULT in result:
        del result[KEY_RESULT]

    return result
