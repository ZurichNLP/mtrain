#!/usr/bin/env python3

from mtrain.constants import *
from mtrain.preprocessing.external import ExternalProcessor
from mtrain.commander import run, run_parallel

'''
Tokenizes files using the default Moses tokenizer.
'''

class Tokenizer(object):
    '''
    Creates a tokenizer which tokenizes sentences on-the-fly, i.e., allowing
    interaction with a Moses tokenizer process kept in memory.
    '''

    def __init__(self, lang_code):
        self._processor = ExternalProcessor(
            command=MOSES_TOKENIZER + "-b -X -a -l %s" % lang_code
        )

    def __del__(self):
        del self._processor

    def tokenize(self, segment):
        '''
        Tokenizes a single segment.
        '''
        return self._processor.process(segment).split(" ")
