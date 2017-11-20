# -*- coding: utf-8 -*-
"""
The multiprocessing_patch module monkey-patches the send/receive functions of Python's multiprocessing module,
so they work properly if blocks larger than 2**31 have to be transferred.
"""

# thank you python, in times of exascale computing someone thought 2**31 bytes oughta be enough ...

import multiprocessing.connection
import struct
import io

SEND_MAX = 2**31 - 8*1024 - 1


# noinspection PyProtectedMember
def _new_send(self, buf, write=multiprocessing.connection.Connection._write):
    remaining = len(buf)
    while True:
        try:
            n = write(self._handle, buf)
        except InterruptedError:
            continue
        remaining -= n
        if remaining == 0:
            break
        buf = buf[n:]


# noinspection PyProtectedMember
def _new_recv(self, size, read=multiprocessing.connection.Connection._read):
    buf = io.BytesIO()
    handle = self._handle
    remaining = size
    while remaining > 0:
        try:
            if remaining > SEND_MAX:
                chunk = read(handle, SEND_MAX)
            else:
                chunk = read(handle, remaining)
        except InterruptedError:
            continue
        n = len(chunk)
        if n == 0:
            if remaining == size:
                raise EOFError
            else:
                raise OSError("got end of file during message")
        buf.write(chunk)
        remaining -= n
    return buf


def _new_send_bytes(self, buf):
    n = len(buf)
    # For wire compatibility with 3.2 and lower
    header = struct.pack("!Q", n)
    if n > 16380:
        # The payload is large so Nagle's algorithm won't be triggered
        # and we'd better avoid the cost of concatenation.
        chunks = [header, buf]
    elif n > 0:
        # Issue # 20540: concatenate before sending, to avoid delays due
        # to Nagle's algorithm on a TCP socket.
        chunks = [header + buf]
    else:
        # This code path is necessary to avoid "broken pipe" errors
        # when sending a 0-length buffer if the other end closed the pipe.
        chunks = [header]
    for chunk in chunks:
        self._send(chunk)


def _new_recv_bytes(self, maxsize=None):
    buf = self._recv(8)
    size, = struct.unpack("!Q", buf.getvalue())
    if maxsize is not None and size > maxsize:
        return None
    return self._recv(size)


multiprocessing.connection.Connection._send_bytes = _new_send_bytes
multiprocessing.connection.Connection._recv_bytes = _new_recv_bytes
multiprocessing.connection.Connection._send = _new_send
multiprocessing.connection.Connection._recv = _new_recv
