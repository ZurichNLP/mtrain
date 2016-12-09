#!/usr/bin/env python3

'''
Class for replacing stretches of text with a mask token or
reversing this process.
'''

from mtrain.constants import *
from mtrain.preprocessing import cleaner

import sys
import re

class _Replacement(object):
    '''
    Tracks replacements in strings.
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
    
    def __init__(self, strategy, escape=True, force_all=False, remove_all=False):
        '''
        @param strategy valid masking strategy
        @param escape whether characters reserved in Moses should be escaped
        @param force_all in unmasking, use all the entries in the mapping, even
            if their mask tokens do not appear in the target segment
        @param remove_all in unmasking, remove all mask tokens from the target
            segment, even if they do not correspond to a mapping entry
        '''
        self._strategy = strategy
        self._escape = escape
        self._force_all = force_all
        self._remove_all = remove_all

        # if no patterns are defined that could be masked
        if not PROTECTED_PATTERNS:
            sys.exit('Masking is not possible because no patterns are defined in PROTECTED_PATTERNS in mtrain/constants.py.')

    def mask_segment(self, segment):
        '''
        Introduces mask tokens into segment and escapes characters.
        @param segment the input text
        '''
        mapping = []
        
        for mask_token, regex in PROTECTED_PATTERNS.items():        
            if self._strategy == MASKING_IDENTITY:
                replacement = _Replacement(mask_token, with_id=True)
            # currently, for all other strategies
            else:
                replacement = _Replacement(mask_token, with_id=False)
            segment = re.sub(regex, replacement, segment)
            mapping.extend(replacement.occurrences)
        
        if self._escape:
            segment = cleaner.escape_special_chars(segment)
    
        return segment, mapping
    
    def mask_tokens(self, tokens):
        '''
        Introduces mask tokens into a list of tokens.
        @param tokens a list of input tokens
        @return a modified list of tokens that can contain mask tokens
            and a list of mappings for recovering the original content
        '''
        masked_segment, mapping = self.mask_segment(" ".join(tokens))
        return masked_segment.split(" "), mapping
    
    def _mask_in_string(self, string):
        '''
        Determines whether there is a mask token in a string.
        '''
        for token in string.split(" "):
            if self._is_mask_token(token):
                return True

    def _mask_in_list(self, list):
        '''
        Determines whether at least one of the item in @param list
            is a mask token.
        '''
        for item in list:
            if self._is_mask_token(item):
                return True

    def _is_mask_token(self, token):
        '''
        Determines whether an input @param token is a mask or not.
        '''
        # make sure token is a string
        token = str(token)        
        
        for mask in PROTECTED_PATTERNS.keys():
            if self._strategy == MASKING_IDENTITY:
                 if re.match("__%s_\d+__" % mask, token):
                    return True
            else:
                if token == "__%s__" % mask:
                    return True
        return False

    def _first_original(self, mask, mapping):
        '''
        Returns the original that belongs to the first occurrence of
            @param mask from @param mapping.
        @return the original and index of this tuple in the mapping
        '''
        for index, tuple in enumerate(mapping):
            token, original = tuple
            if token == mask:
                return original, index

    def _remove_mask_tokens(self, segment):
        '''
        Removes all mask tokens from a string.
        '''
        normal_tokens = []

        for token in segment.split(" "):    
            if not self._is_mask_token(token):
                normal_tokens.append(token)

        return " ".join(normal_tokens)

    def _unmask_segment_identity(self, target_segment, mapping):
        '''
        Replaces mask tokens in @param target_segment using unique mask
            tokens in @param mapping.
        '''
        unused_originals = []

        for mask_token, original in mapping:
            unmasked_segment = target_segment.replace(mask_token, original, 1)
            
            # find out if mask token was replaced or not
            if unmasked_segment == target_segment:
                unused_originals.append(original)
            target_segment = unmasked_segment

        if self._remove_all and self._mask_in_string(target_segment):
            target_segment = self._remove_mask_tokens(target_segment)

        if self._force_all and unused_originals:
            for original in unused_originals:
                target_segment = " ".join( [target_segment, original] )

        return target_segment

    def _unmask_segment_alignment(self, source_segment, target_segment, mapping, alignment):
        '''
        Replaces mask tokens in @param target_segment with their actual content. Decisions
            are based on the source segment, the mapping and word alignments.
        '''
        mapping_masks = [tuple[0] for tuple in mapping]

        # no duplicate masks in the mapping?
        if len(mapping_masks) == len(set(mapping_masks)):
            # then, same strategy as identity masking
            target_segment = self._unmask_segment_identity(target_segment, mapping)
        else:
            source_tokens = source_segment.split(" ")
            target_tokens = target_segment.split(" ")
            # go through source tokens
            for source_index, source_token in enumerate(source_tokens):
                if self._is_mask_token(source_token):
                    # follow alignment to target token(s)
                    for candidate_index in alignment[source_index]:
                        # determine aligned token in target phrase
                        target_token = target_tokens[candidate_index]
                        if self._is_mask_token(target_token):
                            # then, finally reinsert original content
                            target_tokens[candidate_index], remove_index = self._first_original(source_token, mapping)
                            del mapping[remove_index]

                            # immediately break out, do not check other aligned candidates in the target phrase
                            break

            # if there are still mask tokens in the target segment
            if self._remove_all and self._mask_in_list(target_tokens):
                  for index, token in reversed(list(enumerate(target_tokens))):
                    if self._is_mask_token(token):
                        del target_tokens[index]
            # and if masks are left in the mapping that could not be assigned
            if self._force_all and mapping:
                for _, original in mapping:
                    target_tokens.append(original)
        
            target_segment = " ".join(target_tokens)

        return target_segment

    def unmask_segment(self, source_segment, target_segment, mapping, alignment=None):
        '''
        Removes mask tokens from string and replaces them with their actual content.
        @param source_segment masked segment before translation
        @param target_segment text to be unmasked
        @param mapping list of tuples [(mask_token, original_content), ...]
        @param alignment word alignment reported by Moses, a dictionary with
            source indexes as keys 
        '''
        
        if mapping:
            if self._strategy == MASKING_ALIGNMENT:
                target_segment = self._unmask_segment_alignment(source_segment, target_segment, mapping, alignment)

            elif self._strategy == MASKING_IDENTITY:
                target_segment = self._unmask_segment_identity(target_segment, mapping)

        # check if there are still masks in the target segment
        if self._mask_in_string(target_segment):
            logging.debug(
                "Target segment: '%s' still contains mask tokens after unmasking." % target_segment
            )
                    
        return target_segment

    def force_mask_translation(self, segment):
        '''
        Turns mask tokens into forced translation directives which
            ensure that Moses translates mask tokens correctly.
        @param masked_segment an untranslated segment that probably
            has mask tokens
        '''
        if self._strategy == MASKING_IDENTITY:
            pattern = re.compile(r'__\w+_\d+__')
        else:
            pattern = re.compile(r'__\w+__')
        for mask_token in set(re.findall(pattern, segment)):
            replacement = '<mask translation="%s">%s</mask>' % (mask_token, mask_token)
            segment = segment.replace(mask_token, replacement)

        return segment

def write_masking_patterns(protected_patterns_path, markup_only=False):
    '''
    Writes protected patterns to a physical file in the engine directory.
    @param protected_patterns_path path to file the patterns should be written to
    @param markup_only whether only markup should be protected during tokenization
    '''
    with open(protected_patterns_path, 'w') as patterns_file:
        if markup_only:
            patterns_file.write("# %s\n%s\n" % ('xml', PROTECTED_PATTERNS['xml']))
        else:
            for mask_token, regex in PROTECTED_PATTERNS.items():
                patterns_file.write("# %s\n%s\n" % (mask_token, regex))

