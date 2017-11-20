#!/usr/bin/env python3

from mtrain.constants import *
from mtrain.preprocessing.external import ExternalProcessor

'''
###BH text? Copied from tokenizer.py and adapted.
Normalize punctuation using the default Moses normalizer.
'''

class Normalizer(object):
    '''
    Creates a normalizer for processing sentence by sentence, allowing
    interaction with a Moses normalizer process kept in memory.
    '''

    def __init__(self, lang_code):
        '''
        @param lang_code language identifier
        ###BH text todo
        '''
        arguments = [
            '-l %s' % lang_code,
            '-b', #disable Perl buffering
            '-q', #don't report version
        ]   # no aggressive mode '-a' for normalizer

        self._processor = ExternalProcessor(
            command=" ".join([MOSES_NORMALIZER] + arguments)
        )

    def close(self):
        del self._processor

    def normalize_punctuation(self, segment):
        '''
        Normalizes punctuation of a single segment.
        '''
        normalized_segment = self._processor.process(segment)
        return normalized_segment
