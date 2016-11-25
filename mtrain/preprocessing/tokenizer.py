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

    def __init__(self, lang_code, protect=False, protected_patterns_path=None, escape=True):
        '''
        @param lang_code language identifier
        @param protect whether the tokenizer should respect patterns that should not be tokenized
        @param protected_patterns_path path to file with protected patterns
        @param whether characters critical to the decoder should be escaped
        '''
        arguments = [
            '-l %s' % lang_code,
            '-b', #disable Perl buffering
            '-q', #don't report version
            '-a', #aggressive mode
        ]

        if protect:
            arguments.append(
                '-protected %s' % protected_patterns_path, # protect e.g. inline XML, URLs and email
            )

        if not escape:
            arguments.append(
                '-no-escape' # do not escape reserved characters in Moses
            )
        
        self._processor = ExternalProcessor(
            command=" ".join([MOSES_TOKENIZER] + arguments)
        )

    def close(self):
        del self._processor

    def tokenize(self, segment):
        '''
        Tokenizes a single segment.
        '''
        return self._processor.process(segment)

class Detokenizer(object):
    '''
    Creates a detokenizer which detokenizes lists of tokens on-the-fly, i.e.,
    allowing interaction with a Moses detokenizer process kept in memory.
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
