
# encoding: utf-8

from collections import OrderedDict
from inspect import isclass
import inspect

if getattr(inspect, 'signature'):
    def get_argnames_and_defaults(call):
        sig = inspect.signature(call)
        args = [para for para in sig.parameters]
        # noinspection PyProtectedMember
        defaults = [para.default for para in sig.parameters.values() if para.default is not inspect._empty]
        return args, defaults

else:
    def get_argnames_and_defaults(call):
        # noinspection PyDeprecation
        argspec = inspect.getargspec(call)

        args = list(argspec.args)
        defaults = argspec.defaults if list(argspec.defaults) else []

        return args, defaults


class NeatDict(dict):
    def __getattr__(self, item):
        return self.get(item)

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, item):
        del self[item]


class Pipeline(object):

    def __init__(self):
        self.steps = []

    def __lt__(self, other):
        return self.__le__(other)

    def __le__(self, other):
        return self.call(other)

    def __or__(self, other):
        self.add(other)
        return self

    def __ior__(self, other):
        self.add(other)
        return self

    def add(self, what):
        self.steps.append(what)

    def call(self, parameter):
        return PipelineInvocation(self, parameter)


class PipelineInvocation(object):
    def __init__(self, pipeline, parameter):
        self.pipeline = pipeline
        self.parameter = parameter


class PipelineEnvironment(object):
    debug = False

    KEY_COLLECTED = 'collected'
    KEY_RESULT = 'result'

    def __init__(self, **kwargs):
        self.external_di = {'pipeline_environment': self}
        self.external_di.update(kwargs)

    @staticmethod
    def new_pipeline():
        return Pipeline()

    def run(self, pipeline_invocation, **kwargs):

        if type(pipeline_invocation) == Pipeline:
            pipeline_invocation = pipeline_invocation < kwargs

        result = pipeline_invocation.parameter
        for raw_step in pipeline_invocation.pipeline.steps:
            step = self.wrap(raw_step)

            result = step(result)
        return result

    def debug_message(self, *args, **kwargs):
        if self.debug:
            print(args, kwargs)
        else:
            pass

    def wrap_result(self, result):

        result = NeatDict(result)

        result[self.KEY_RESULT] = True

        if self.KEY_COLLECTED in result:
            wrapped = OrderedDict()
            for k, v in result[self.KEY_COLLECTED].items():
                if v is None:
                    # wrapped[k] = None
                    pass
                else:
                    wrapped[k] = NeatDict(v)
            result[self.KEY_COLLECTED] = wrapped

        return result

    def unwrap_result(self, result):
        if self.KEY_COLLECTED in result and (result[self.KEY_COLLECTED] is not None):
            unwrapped = OrderedDict()
            for k, v in result[self.KEY_COLLECTED].items():
                unwrapped[k] = dict(v)
            result[self.KEY_COLLECTED] = unwrapped

        result = dict(result)

        if self.KEY_RESULT in result:
            del result[self.KEY_RESULT]

        return result

    def prepare_call(self, call, di=None, result=None):
        if di is None:
            di = {}

        if result is None:
            result = {}

        args, defaults = get_argnames_and_defaults(call)

        if args[0] == 'self':
            args = args[1:]

        non_default_parameters = len(args) - len(defaults)

        def _wrapped():
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
                        if arg in di:
                            from_di.add(arg)
                            parameters.append(di[arg])
                        else:
                            # problem: pipeline step asks for a parameter we do not have
                            raise ValueError('[At %s]: Argument %r not in %r' % (repr(call), arg, result,))

            _call_return = call(*parameters)
            return args, _call_return

        return _wrapped

    def reassemble_result(self, result, args, _call_return):
        if type(_call_return) == dict and self.KEY_RESULT in _call_return:
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
                k = args[-(n+1)]
                if k == self.KEY_RESULT:
                    if type(item) == dict or type(item) == NeatDict:
                        result.update(item)
                else:
                    # do nothing if the parameter came from di
                    if k not in self.external_di:
                        result[k] = item
        return result

    def wrap(self, what):
        name = ("class_" if isclass(what) else "function_") +\
            ('__LAMBDA__' if getattr(what, '__name__', False) == '<lambda>' else getattr(what, '__name__', repr(what)))

        if isclass(what):
            # we create an instance without calling the constructor, so we can setup the environment first
            instance = what.__new__(what)
            # now we call the constructor, and it has everything neatly set up already!

            init_call = self.prepare_call(instance.__init__, di=self.external_di)
            init_call()

            call = instance
        else:
            call = what

        def _wrapped(result):
            self.debug_message("Entering %s", name)
            result = self.wrap_result(result)

            _pre_wrapped_call = self.prepare_call(call, self.external_di, result)

            args, _call_return = _pre_wrapped_call()

            result = self.reassemble_result(result, args, _call_return)

            result = self.unwrap_result(result)

            self.debug_message("Leaving %s", name)
            return result

        _wrapped.__name__ = name
        _wrapped.__qualname__ = _wrapped.__name__

        return _wrapped


class PipelineExecutionContext(object):
    def add_stage(self, step, pipeline=None):
        if pipeline is None:
            pipeline = Pipeline()

        self.steps[step] = pipeline
        return pipeline

    def get_step_keys(self):
        return list(self.steps.keys())

    def dispatch(self, step, **kwargs):
        return self.pipeline_environment.run(self.steps[step], **kwargs)

    def __new__(cls, *args, **kwargs):
        instance = super(PipelineExecutionContext, cls).__new__(cls)
        instance.steps = {}
        instance.pipeline_environment = PipelineEnvironment()
        return instance
