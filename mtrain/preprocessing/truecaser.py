#!/usr/bin/env python3

from mtrain.constants import *
from mtrain.preprocessing.external import ExternalProcessor

'''
(De)truecases segments using the default Moses (de)truecaser.
'''

class Truecaser(object):
    '''
    Creates a truecaser which truecases sentences on-the-fly, i.e., allowing
    interaction with a Moses truecaser process kept in memory.
    '''

    def __init__(self, path_model):
        '''
        @param path_model path to truecasing model trained in `mtrain`
        '''
        arguments = [
            '-model %s' % path_model,
            '-b' #disable Perl buffering
        ]

        self._processor = ExternalProcessor(
            command=" ".join([MOSES_TRUECASER] + arguments)
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

class Detruecaser(object):
    '''
    Creates a detruecaser which detruecases sentences on-the-fly, i.e., allowing
    interaction with a Moses truecaser process kept in memory.
    '''

    def __init__(self):
        '''
        Detruecaser only needs script, no model
        '''
        arguments = [
            '-b' #disable Perl buffering
        ]

        ###BH add dedication to MOSES_DETRUECASER and postprocess-test.sh
        self._processor = ExternalProcessor(
            command=" ".join([MOSES_DETRUECASER] + arguments)
        )

    def close(self):
        del self._processor

    def detruecase(self, segment):
        '''
        Detruecases a single segment.
        '''
        return self._processor.process(segment)
