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

from dotmap import DotMap

def wrap_dict(what):
    return DotMap(what)

def unwrap_dict(what):
    return what.toDict()


def call(call, result, di=None):

    args, defaults = get_argnames_and_defaults(call)

    if args[0] == 'self':
        args = args[1:]

    non_default_parameters = len(args) - len(defaults)

    from_di = set()
    parameters = []

    for n, arg in enumerate(args):
        if arg in result:
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

    ### the call

    _call_return = call(*parameters)

    ### /the call

    if type(_call_return) != tuple:
        _call_return = (_call_return,)

    for n, item in enumerate(reversed(_call_return)):
        k = args[-(n + 1)]
        # do nothing if the parameter came from di
        if k not in from_di:
            result[k] = item

    return result
