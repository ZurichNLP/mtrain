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

    def __init__(self, command, stream_stderr=False, trailing_output=False, shell=True):
        '''
        @param command the command that should be executed on the shell
        @param stream_stderr whether STDERR should be streamread in a non-
            blocking way
        @param trailing_output whether the external process outputs trailing
            lines after the actual, single, output line
        @param shell whether or not the command shall be executed as shell script process
        '''
        # calling extenal processor with underlying shell script process
        self.command = command
        self._stream_stderr = stream_stderr
        self._trailing_output = trailing_output
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

        '''
        # calling extenal processor as python process
        # e.g. for bpe encoder. python subprocess needs command as list of elements
        ###BH check reference # cf. https://docs.python.org/3.6/library/subprocess.html
        elif shell == False:
            self.command = command.split()
            self._stream_stderr = stream_stderr
            self._trailing_output = trailing_output
            logging.debug("Executing %s", self.command)
            self._process = Popen(
                self.command,
                shell=False,
                stdin=PIPE,
                stdout=PIPE,
                stderr=PIPE if self._stream_stderr else None
            )
            self._lock = threading.Lock()
            if self._stream_stderr:
                self._nbsr = _NonBlockingStreamReader(self._process.stderr)
        '''

    def close(self):
        '''
        Closes the underlying process.
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
            # work around Moses printing an empty line after alignment info
            if self._trailing_output:
                self._process.stdout.readline() # do nothing with this line
            # attempt reading from STDERR
            if self._stream_stderr:
                errors = self._nbsr.readline()
                if errors:
                    message = errors.decode()
                    if commander._is_relevant_for_log(message):
                        logging.info(message.strip())
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