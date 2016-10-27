#!/usr/bin/env python3

from mtrain.constants import *

from lxml import etree
import re

'''
Reinsert markup into target segments, based on the markup found in the source segments.
'''

class Reinserter(object):
    '''
    Class for reinserting markup into segments that were stripped of
        markup prior to translation.
    '''

    def __init__(self, reinsertion_strategy):
        self._reinsertion_strategy = reinsertion_strategy

    def _next_source_tag_region(self, source_tokens):
        '''
        Finds a source tag region for reinsertion that does not itself contain tags.
        @return a list of indexes if a region is found, empty if no region was found
        '''
        while _tag_in_list(source_tokens):
            current_indexes = []
            current_opening_tag = None        
            tags_seen_offset = 0
            
            for source_index, source_token in enumerate(source_tokens):
                if _is_closing_tag(source_token):
                    # delete this pair of tags from the source
                    del source_tokens[source_index]
                    del source_tokens[current_opening_tag[0]]
                    # then yield, because the closing tag is bound to correspond to
                    # the most recent opening tag
                    yield current_opening_tag[1], current_indexes, source_token
                    break # break?            
                elif _is_opening_tag(source_token):
                    tags_seen_offset += 1
                    # then reset current indexes and tags
                    current_indexes = []
                    current_opening_tag = source_index, source_token
                else:
                    if current_opening_tag:
                        # then add to indexes
                        current_indexes.append(source_index - tags_seen_offset)
                    # else, do nothing; not yet in a source tag region

    def _find_source_phrase_regions(self, source_tag_region, segmentation):
        '''
        Given a source tag region and segmentation, determines the source phrases
            the source tag region is in.
        @param source_tag_region a list of indexes that identify a source tag region
        @param segmentation phrase segmentation reported by Moses, a dictionary
            where keys are tuples of source spans
        @return a sorted list of tuples, a subset of @param segmentation
        '''
        source_phrase_region = []

        for start, end in segmentation:
            phrase_collected = False
            # one of the tokens of the source tag region in this phrase?
            for index in source_tag_region:
                if int(start) <= index <= int(end):
                    source_phrase_region.append( (start, end) )
                    phrase_collected = True
                    # break early because there is enough evidence already
                    break
            if source_phrase_region and not phrase_collected:
                # break early because source phrase region must be contiguous
                break

        return sorted(source_phrase_region)

    def _find_target_covering_phrases(self, source_phrase_regions, segmentation):
        '''
        Finds the phrases in the target phrase that were used to translate the source
            phrase regions.
        @param source_phrase_regions list of source phrases that encompass the current
            source tag region
        @param segmentation phrase segmentation reported by Moses, a dictionary
            where keys are tuples of source spans
        '''
        target_covering_phrases = []

        for tuple in source_phrase_regions:
            target_covering_phrases.append(segmentation[tuple])

        return target_covering_phrases

    def _str_spr_coincide(self, source_tag_region, source_phrase_regions):
        '''
        Returns true if the boundaries of the source tag region coincide with the first
            starting index of the source_phrase_regions, and with their last ending index
        '''
        # source_phrase_regions must be sorted
        if ( source_tag_region[0] == source_phrase_regions[0][0] and
            source_tag_region[-1] == source_phrase_regions[-1][1]) :
            return True
        else:
            return False

    def _tcp_is_contiguous(self, target_covering_phrases):
        '''
        Determine whether the target covering phrases are contiguous.
        @param target_covering_phrases a list of tuples with index spans
        '''
        last_tuple = ()
        for start, end in target_covering_phrases:
            if not last_tuple:
                last_tuple = (start, end)
            # if start of phrase not immediately adjacent to last phrase
            elif last_tuple[1] != start - 1:
                return False
            else:
                last_tuple = (start, end)

        # if you made it thus far, then
        return True

    def _reinsert_markup_full(self, source_segment, target_segment, segmentation, alignment):
        '''
        Reinserts markup, taking into account both phrase segmentation and word alignment.
        @param source_segment the original segment in the source language
            before XML markup was removed, but after markup-aware tokenization
        @param target_segment a translated segment without markup
        @param segmentation phrase segmentation reported by Moses, a dictionary
            where keys are tuples of source spans
        @param alignment word alignment information reported by Moses, a
            dictionary where source tokens are keys
        '''
        source_tokens = source_segment.split(" ")
        target_tokens = target_segment.split(" ")

        changes = []
        
        str_generator = self._next_source_tag_region(source_tokens)
        
        for opening_tag, source_tag_region, closing_tag in str_generator:
            if not source_tag_region:
                # tag pair with no content tokens between them
                pass
            else:
                source_phrase_regions = self._find_source_phrase_regions(source_tag_region, segmentation)
                target_covering_phrases = self._find_target_covering_phrases(source_phrase_regions, segmentation)

                if self._str_spr_coincide(source_tag_region, source_phrase_regions):
                    if self._tcp_is_contiguous(target_covering_phrases):
                        # Rule 1
                        changes.append(
                            (target_covering_phrases[-1][1] + 1, closing_tag)
                        )
                        changes.append(
                            (target_covering_phrases[0][0], opening_tag)
                        )
                        
                    else:
                        # Rule 3
                        changes.append(
                            (min(target_covering_phrases, key=lambda x: x[0])[0], opening_tag)
                        )
                        changes.append(
                            (max(target_covering_phrases, key=lambda x: x[1])[1] + 1, closing_tag)
                        )
                else:
                        # Rule 2
                        relevant_alignments = []
                        for index in source_tag_region:
                            relevant_alignments.extend(alignment[index])
                        changes.append(
                            (min(relevant_alignments), opening_tag)
                        )
                        changes.append(
                            (max(relevant_alignments) + 1, closing_tag)
                        )

        # then, actually make changes in reverse order
        for insert_at, tag in sorted(changes, key=lambda x: x[0], reverse=True):
            target_tokens.insert(insert_at, tag)

        return " ".join(target_tokens)

    def reinsert_markup(self, source_segment, target_segment, segmentation, alignment):
        '''
        Reinsert markup found in the source segment into the target segment.
        '''
        if self._reinsertion_strategy == REINSERTION_FULL:
            self._reinsert_markup_full(source_segment, target_segment, segmentation, alignment)
        else:
            raise NotImplementedError(
                "Reinsertion strategy '%s' is unknown." % self._reinsertion_strategy
                )

def _is_opening_tag(token):
    '''
    Determines whether @param token is the opening tag of an XML element.
    '''
    return bool( re.match(r"<[a-zA-Z_][^\/<>]*>", token) )

def _is_closing_tag(token):
    '''
    Determines whether @param token is the closing tag of an XML element.
    '''
    return bool( re.match(r"<\/[a-zA-Z_][^\/<>]*>", token) )

def _is_selfclosing_tag(token):
    '''
    Determines whether @param token is a self-closing XML element.
    '''
    return bool( re.match(r"<[a-zA-Z_][^\/<>]*\/>", token) )

def _is_xml_comment(token):
    '''
    Determines whether @param token is an XML comment.
    '''
    return bool( re.match(r"<!\-\-[^<>]*\-\->", token) )

def _tag_in_list(tokens):
    '''
    Returns True if any token in the list is a tag.
    '''
    for token in tokens:
        if _is_opening_tag(token) or _is_closing_tag(token) or _is_selfclosing_tag(token):
            return True
    return False

def _element_names_identical(opening_tag, closing_tag):
    '''
    Attempts to parse the concatenation of two input strings.
    @return true if the element names are identical.
    '''
    # concatenate strings, if XML parser does not fail then the element names were identical
    try:
        etree.fromstring(opening_tag + closing_tag)
        return True
    except:
        # not well-formed XML = element names are not identical
        return False
