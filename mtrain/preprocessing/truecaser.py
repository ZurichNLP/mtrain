#!/usr/bin/env python3

from mtrain.constants import *
from mtrain.preprocessing.external import ExternalProcessor

'''
Truecases segments using the default Moses truecaser.
'''

class Truecaser(object):
    '''
    Creates a truecaser which truecases sentences on-the-fly, i.e., allowing
    interaction with a Moses truecaser process kept in memory.
    '''

    def __init__(self, path_model):
        self._processor = ExternalProcessor(
            command=MOSES_TRUECASER + " --model %s" % path_model
        )

    def close(self):
        del self._processor

    def truecase(self, segment):
        '''
        Truecases a single segment.
        '''
        return self._processor.process(segment)

    def truecase_tokens(self, tokens):
        '''
        Truecases a list of tokens.
        '''
        return self.truecase(" ".join(tokens)).split(" ")
