#!/usr/bin/env python3

'''
Class for replacing stretches of text with a mask token or
reversing this process.
'''

from mtrain.constants import *
from mtrain.preprocessing import cleaner
import re

class _Replacement(object):
    '''
    Track replacements in strings.
    Based on: http://stackoverflow.com/a/9135166/1987598
    '''
    def __init__(self, replacement, with_id=False):
        self.replacement = replacement
        self.occurrences = []
        self.with_id = with_id

    def __call__(self, match):
        matched = match.group(0)
        if self.with_id:
            replaced = match.expand(
                "__%s_%d__" % (self.replacement, len(self.occurrences))
            )
        else:
            replaced = match.expand(
                "__%s__" % self.replacement
            )
        self.occurrences.append((replaced, matched))
        return replaced

class Masker(object):
    
    def __init__(self, strategy, escape=True):
        '''
        @param strategy valid masking strategy, either 'alignment' or 'identity'
        @param escape whether characters reserved in Moses should be escaped
        '''
        self._strategy = strategy
        self._escape = escape

    def mask_segment(self, segment):
        '''
        Introduces mask tokens into segment and escapes characters.
        @param segment the input text
        '''
        
        if self._strategy == MASKING_ALIGNMENT:    
            replacement = _Replacement('xml', with_id=False)
        elif self._strategy == MASKING_IDENTITY:
            replacement = _Replacement('xml', with_id=True)
        
        segment = re.sub(r'<\/?[a-zA-Z_][a-zA-Z_.\-0-9]*[^<>]*\/?>', replacement, segment)
        mapping = replacement.occurrences
        
        if escape_moses:
            segment = cleaner.escape_special_chars(segment)
    
        return segment, mapping

    def unmask_segment(self, segment, mapping, word_alignment=None, phrase_alignment=None):
        '''
        Removes mask tokens from string and replaces them with their actual content.
        @param segment text to be unmasked
        @param mapping list of tuples [(mask_token, original_content), ...]
        @param word_alignment word alignment reported by Moses
        @param phrase_alignment phrasal alignment reported by Moses
        '''
        if self._strategy == MASKING_ALIGNMENT:
            raise NotImplementedError
        
        elif self._strategy == MASKING_IDENTITY:
            for mask_token, original in mapping:
                segment = segment.replace(mask_token, original)
        
        return segment
