#!/usr/bin/env python3

'''
Replace stretches of text with a mask token.
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

def mask_segment(segment, strategy, escape_moses=True):
    '''
    Introduces mask tokens into segment and escapes characters.
    @param segment the input text
    @param strategy valid masking strategy, either 'alignment' or 'identity'
    @param moses whether characters reserved in Moses should be escaped
    '''
    
    if strategy == MASKING_ALIGNMENT:    
        replacement = _Replacement('xml', with_id=False)
    elif strategy == MASKING_IDENTITY:
        replacement = _Replacement('xml', with_id=True)
    
    segment = re.sub(r'<\/?[a-zA-Z_][a-zA-Z_.\-0-9]*[^<>]*\/?>', replacement, segment)
    mapping = replacement.occurrences
        
    if escape_moses:
        segment = cleaner.escape_special_chars(segment)

    return segment, mapping

def unmask_segment(segment, strategy, mapping, word_alignment=None, phrase_alignment=None):
    '''
    Removes mask tokens from string and replaces them with their actual content.
    '''
    if strategy == MASKING_ALIGNMENT:
        raise NotImplementedError
    
    elif strategy == MASKING_IDENTITY:
        for mask_token, original in mapping:
            segment = segment.replace(mask_token, original)
    
    return segment
