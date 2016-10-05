#!/usr/bin/env python3

'''
Thread-safe wrapper for interaction with an external I/O shell script. Inspired
by https://github.com/casmacat/moses-mt-server/blob/master/python_server/server.py

Also inspired by eyalarubas.com/python-subproc-nonblock.html
'''

import threading
import logging
from subprocess import Popen, PIPE
from queue import Queue, Empty

from mtrain import commander

class ExternalProcessor(object):
    '''
    Thread-safe wrapper for interaction with an external I/O shell script
    '''

    def __init__(self, command, stream_stderr=False):
        self.command = command
        self._stream_stderr = stream_stderr
        logging.debug("Executing %s", self.command)
        self._process = Popen(
            self.command,
            shell=True,
            stdin=PIPE,
            stdout=PIPE,
            stderr=PIPE if self._stream_stderr else None
        )
        self._lock = threading.Lock()
        if self._stream_stderr:
            self._nbsr = _NonBlockingStreamReader(self._process.stderr)

    def close(self):
        '''
        Closes the underlying shell script (process).
        '''
        self._process.terminate()

    def process(self, line):
        '''
        Processes a line of input through the underlying shell script (process)
        and returns the corresponding output.
        '''
        line = line.strip() + "\n"
        line = line.encode('utf-8')

        with self._lock:
            self._process.stdin.write(line)
            self._process.stdin.flush()
            result = self._process.stdout.readline()
            # attempt reading from STDERR, grace period of 0.1 seconds for external
            if self._stream_stderr:
                # grace period of 0.1 seconds for external process to return STDERR
                errors = self._nbsr.readline(0.1).decode()
                if errors and commander._is_relevant_for_log(errors):
                    logging.info(errors.strip())
        return result.decode().strip()

class _NonBlockingStreamReader:
    '''
    Reads from stream without blocking, even if nothing can be read
    '''
    def __init__(self, stream):
        '''
        @param stream the stream to read from, usually a process' STDOUT or STDERR
        '''

        self._stream = stream
        self._queue = Queue()

        def _populateQueue(stream, queue):
            '''
            Collects lines from '@param stream and puts them in @param queue.
            '''

            while True:
                line = stream.readline()
                if line:
                    queue.put(line)
                else:
                    raise _UnexpectedEndOfStream

        self._thread = threading.Thread(target = _populateQueue,
                args = (self._stream, self._queue))
        self._thread.daemon = True
        self._thread.start() # start collecting lines from the stream

    def readline(self, timeout = None):
        try:
            return self._queue.get(block = timeout is not None,
                    timeout = timeout)
        except Empty:
            return None

class _UnexpectedEndOfStream(Exception): pass
