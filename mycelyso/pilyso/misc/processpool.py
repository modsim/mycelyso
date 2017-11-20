# -*- coding: utf-8 -*-
"""
The processpool submodule contains a custom process-pool, with some additional features not present in standard
Python processpool management, e.g. timeouts.
"""

from time import sleep
import datetime
import traceback
import heapq

from multiprocessing import Process, Pipe, cpu_count

from multiprocessing.pool import Pool

exception_debugging = False


class Future(object):
    command = None
    args = None
    kwargs = None

    value = None
    error = None

    process = None
    pool = None

    priority = 0

    status = None

    timeout = 0
    started_at = None

    def __init__(self):
        pass

    def __lt__(self, other):
        # dummy
        return self.priority < other.priority

    def wait(self, time=None):
        pass

    def fail(self):
        self.status, (self.value, self.error) = True, (None, None)

        if self.process:
            self.process.terminate()
            self.pool.report_broken_process(self.process)
        else:
            self.pool.future_became_ready(self)

    def ready(self):
        if self.status:
            return self.status

        if self.process is None:
            # not scheduled yet
            return False

        if self.timeout > 0:
            now = datetime.datetime.now()
            if (now - self.started_at).total_seconds() > self.timeout:
                # we reached a hard timeout

                try:
                    self.process.terminate()
                    # kill(self.process.pid, SIGINT)
                    sleep(0.25)
                except Exception as e:
                    print(repr(e))

                # self.process.terminate()

                self.pool.report_broken_process(self.process)
                self.status, (self.value, self.error) = \
                    True, (None, RuntimeError('Process took longer than specified timeout and was terminated.'))
                return

        if not self.process.is_alive():
            exitcode = self.process.exitcode
            self.pool.report_broken_process(self.process)
            self.status, (self.value, self.error) = \
                True, (None, RuntimeError('Process trying to work on this future died. Exitcode: %d' % (exitcode,)))
            return

        self.status, (self.value, self.error) = self.process.ready()

        if self.status:
            self.pool.future_became_ready(self)

        return self.status

    def get(self):
        not_ready = True
        while not_ready:
            not_ready = not self.ready()

        if self.error:
            raise self.error
        else:
            return self.value

    def dispatch(self):
        self.started_at = datetime.datetime.now()
        self.process.dispatch()


class WrappedException(RuntimeError):
    exception = None
    message = None

    def __init__(self, exception, message=''):
        self.exception = exception
        self.message = message


class FutureProcess(Process):
    STARTUP = 0
    RUN = 1
    STOP = 2

    def __init__(self):
        super(FutureProcess, self).__init__()
        self.future = None
        self.pipe_parent_end, self.pipe_child_end = Pipe()

    def run(self):
        while True:
            command_type, command, args, kwargs = self.pipe_child_end.recv()

            if command_type == FutureProcess.STOP:
                break

            if command_type == FutureProcess.STARTUP and command is None:
                continue

            result = None
            exc = None

            try:
                result = command(*args, **kwargs)
            except Exception as e:
                exc = WrappedException(e, traceback.format_exc())

            if command_type == FutureProcess.STARTUP:
                continue

            self.pipe_child_end.send((result, exc,))

    def send_command(self, *args):
        self.pipe_parent_end.send(args)

    def dispatch(self):
        self.send_command(FutureProcess.RUN, self.future.command, self.future.args, self.future.kwargs)

    def ready(self):
        if self.pipe_parent_end.poll():
            return True, self.pipe_parent_end.recv()
        else:
            return False, (None, None,)


class SimpleProcessPool(object):
    def new_process(self):
        p = FutureProcess()
        p.start()
        p.send_command(*self.startup_message)
        return p

    def fill_pool(self):
        for _ in range(self.count - (len(self.waiting_processes) + len(self.active_processes))):
            self.waiting_processes.add(self.new_process())

    def __init__(self, processes=0, initializer=None, initargs=None, initkwargs=None, future_timeout=0):

        if initargs is None:
            initargs = ()

        if initkwargs is None:
            initkwargs = {}

        self.startup_message = (FutureProcess.STARTUP, initializer, initargs, initkwargs)

        if processes == 0:
            processes = cpu_count()

        self.future_timeout = future_timeout
        self.count = processes

        self.waiting_processes = set()
        self.active_processes = set()

        self.fill_pool()

        self.active_futures = set()
        self.waiting_futures = []  # priority queue

        self.closing = False

    def close(self):
        self.closing = True

        self.schedule()

    def report_broken_process(self, p):
        f = p.future
        if p in self.active_processes:
            self.active_processes.remove(p)

        if p in self.waiting_processes:
            # print("found a process where it does not belong", p)
            self.waiting_processes.remove(p)

        if f in self.active_futures:
            self.active_futures.remove(f)

        if f in self.waiting_futures:
            # remove one entry, rebuild
            self.waiting_futures.remove(f)
            heapq.heapify(self.waiting_futures)

        # if f in self.waiting_futures:
        #    #print("found a future where it does not belong", f)
        #    self.waiting_futures.remove(f)

        self.fill_pool()

        self.schedule()

    def apply(self, command, *args, **kwargs):
        return self.advanced_apply(command=command, args=args, kwargs=kwargs)

    def advanced_apply(self, command=None, priority=0, args=None, kwargs=None):
        if args is None:
            args = tuple()
        if kwargs is None:
            kwargs = {}

        f = Future()
        f.command = command
        f.args = args
        f.kwargs = kwargs

        f.priority = priority

        f.timeout = self.future_timeout

        f.pool = self

        # self.waiting_futures.add(f)
        heapq.heappush(self.waiting_futures, f)

        self.schedule()

        return f

    # ugly signature
    def apply_async(self, fun, args=None, kwargs=None):
        if args is None:
            args = {}
        if kwargs is None:
            kwargs = {}
        return self.apply(fun, *args, **kwargs)

    def future_became_ready(self, f):
        if f in self.active_futures:
            self.active_futures.remove(f)

        if f.process in self.active_processes:
            self.active_processes.remove(f.process)

        if f.process:
            self.waiting_processes.add(f.process)

        self.schedule()

    def schedule(self):
        for f in self.active_futures.copy():
            f.ready()

        while len(self.waiting_processes) > 0:
            if len(self.waiting_futures) == 0:
                break

            # f = self.waiting_futures.pop()
            f = heapq.heappop(self.waiting_futures)

            p = self.waiting_processes.pop()

            f.process = p
            p.future = f

            self.active_processes.add(p)
            self.active_futures.add(f)

            f.dispatch()

        if self.closing:
            while len(self.waiting_processes) > 0:
                p = self.waiting_processes.pop()
                p.send_command(FutureProcess.STOP, None, [], {})
                self.active_processes.add(p)


class DuckTypedApplyResult(object):
    def __init__(self, callable_):
        self.value = None
        self.called = False
        self.callable = callable_

    # noinspection PyMethodMayBeStatic
    def ready(self):
        return True

    def wait(self, timeout):
        pass

    def fail(self):
        self.value = None
        self.called = True

    def get(self):
        if not self.called:
            self.value = None
            try:
                self.value = self.callable()
            except Exception as e:
                if exception_debugging:
                    raise

                raise WrappedException(e, traceback.format_exc())

        return self.value


# noinspection PyAbstractClass,PyUnusedLocal
class NormalPool(Pool):
    def advanced_apply(self, command, args, **kwargs):
        return self.apply(func=command, args=args)


# noinspection PyUnusedLocal
class InProcessFakePool(object):
    @staticmethod
    def advanced_apply(command, args, **kwargs):
        def _bind_it(inner_args):
            def _perform():
                return command(*inner_args)

            return _perform

        return DuckTypedApplyResult(_bind_it(args))

    def close(self):
        pass
