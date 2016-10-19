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
        mapping = []
        
        for mask_token, regex in PROTECTED_PATTERNS.items():        
            if self._strategy == MASKING_ALIGNMENT:    
                replacement = _Replacement(mask_token, with_id=False)
            elif self._strategy == MASKING_IDENTITY:
                replacement = _Replacement(mask_token, with_id=True)
        
            segment = re.sub(regex, replacement, segment)
            mapping.extend(replacement.occurrences)
        
        if self._escape:
            segment = cleaner.escape_special_chars(segment)
    
        return segment, mapping
    
    def mask_tokens(self, tokens):
        masked_segment, mapping = self.mask_segment(" ".join(tokens))
        return masked_segment.split(" "), mapping
    
    def unmask_segment(self, segment, mapping, alignment=None, segmentation=None):
        '''
        Removes mask tokens from string and replaces them with their actual content.
        @param segment text to be unmasked
        @param mapping list of tuples [(mask_token, original_content), ...]
        @param alignment word alignment reported by Moses,
            a list of tuples [(source word, target word), ...]
        @param segmentation phrase segmentation reported by Moses,
            a list of tuples of tuples [((first source token, last source token), ...), ...]
        '''
        if self._strategy == MASKING_SEGMENTATION:
            # assume that info on phrase segmentation is available
            

        elif self._strategy == MASKING_ALIGNMENT:
            raise NotImplementedError
        
        elif self._strategy == MASKING_IDENTITY:
            for mask_token, original in mapping:
                segment = segment.replace(mask_token, original)
        
        return segment

def write_masking_patterns(protected_patterns_path):
    '''
    Writes protected patterns to a physical file in the engine directory.
    @param protected_patterns_path path to file the patterns should be written to
    '''
    with open(protected_patterns_path, 'w') as patterns_file:
        for mask_token, regex in PROTECTED_PATTERNS.items():
            patterns_file.write("# %s\n%s\n" % (mask_token, regex))
