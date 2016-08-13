#!/usr/bin/env python3

'''
Thread-safe wrapper for interaction with an external I/O shell script. Inspired
by https://github.com/casmacat/moses-mt-server/blob/master/python_server/server.py
'''

import threading
import logging
from subprocess import Popen, PIPE

class ExternalProcessor(object):
    '''
    Thread-safe wrapper for interaction with an external I/O shell script
    '''

    def __init__(self, command):
        self.command = command
        logging.info("Executing %s", self.command)
        self._process = Popen(self.command, shell=True, stdin=PIPE, stdout=PIPE)
        self._lock = threading.Lock()

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
        return result.decode().strip()
