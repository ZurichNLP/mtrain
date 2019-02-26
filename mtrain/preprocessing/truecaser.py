#!/usr/bin/env python3

from mtrain import constants as C
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
        """
        @param path_model path to truecasing model trained in `mtrain`
        """
        arguments = [
            '-model %s' % path_model,
            '-b' #disable Perl buffering
        ]

        self._processor = ExternalProcessor(
            command=" ".join([C.MOSES_TRUECASER] + arguments)
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
