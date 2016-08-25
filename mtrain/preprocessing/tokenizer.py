#!/usr/bin/env python3

from mtrain.constants import *
from mtrain.preprocessing.external import ExternalProcessor

'''
Tokenizes files using the default Moses tokenizer.
'''

class Tokenizer(object):
    '''
    Creates a tokenizer which tokenizes sentences on-the-fly, i.e., allowing
    interaction with a Moses tokenizer process kept in memory.
    '''

    def __init__(self, lang_code):
        arguments = [
            '-l %s' % lang_code,
            '-b', #disable Perl buffering
            '-q', #don't report version
            '-X', #skip XML
            '-a', #aggressive mode
        ]
        self._processor = ExternalProcessor(
            command=" ".join([MOSES_TOKENIZER] + arguments)
        )

    def close(self):
        del self._processor

    def tokenize(self, segment):
        '''
        Tokenizes a single segment.
        '''
        return self._processor.process(segment).split(" ")

class Detokenizer(object):
    '''
    Creates a detokenizer which detokenizes lists of tokens on-the-fly, i.e.,
    allowing interaction with a Moses detokeinzer process kept in memory.
    '''

    def __init__(self, lang_code, uppercase_first_letter=False):
        '''
        @param uppercase_first_letter whether or not to uppercase the first
            letter in the detokenized output.
        '''
        arguments = [
            '-l %s' % lang_code,
            '-b', #disable Perl buffering
            '-q', #don't report version
        ]
        if uppercase_first_letter:
            arguments.append('-u')
        self._processor = ExternalProcessor(
            command=" ".join([MOSES_DETOKENIZER] + arguments)
        )

    def close(self):
        del self._processor

    def detokenize(self, tokens):
        '''
        Detokenizes a list of tokens into a segment
        '''
        return self._processor.process(" ".join(tokens))
