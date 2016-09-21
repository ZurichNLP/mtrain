#!/usr/bin/env python3

from mtrain.constants import *
from mtrain.preprocessing.external import ExternalProcessor

'''
Recases segments using a Moses recaser engine.
'''

class Recaser(object):
    '''
    Creates a recaser which recases sentences on-the-fly, i.e., allowing
    interaction with a Moses recaser engine kept in memory.
    '''

    def __init__(self, path_moses_ini):
        arguments = [
            '-f %s' % path_moses_ini,
            '-dl 0',
            '-minphr-memory',
            '-v 0',
        ]
        self._processor = ExternalProcessor(
            command=" ".join([MOSES] + arguments)
        )

    def close(self):
        del self._processor

    def recase(self, segment):
        '''
        Recases a single segment.
        '''
        return self._processor.process(segment)

    def recase_tokens(self, tokens):
        '''
        Recases a list of tokens.
        '''
        return self.recase(" ".join(tokens)).split(" ")
