#!/usr/bin/env python3

'''
Class for handling XML markup during training and translation
'''

from mtrain.preprocessing.masking import Masker
from mtrain.preprocessing.reinsertion import Reinserter
from mtrain.constants import *

import re
from lxml import etree
from xml.sax import saxutils

class XmlProcessor(object):
    '''
    Process XML markup properly before training, and before
        and after translation.
    '''
    def __init__(self, xml_strategy):
        self._xml_strategy = xml_strategy
        if self._xml_strategy in (XML_STRIP, XML_STRIP_REINSERT):
            self._reinserter = Reinserter(XML_STRATEGIES_DEFAULTS[self._xml_strategy])
        elif self._xml_strategy == XML_MASK:
            self._masker = Masker(XML_STRATEGIES_DEFAULTS[self._xml_strategy])

    def _strip_markup(self, segment, keep_escaped_markup=True):
        '''
        Removes all XML markup from a segment and normalizes
            whitespace between tokens before returning.
        @param segment the string from which XML markup
            should be removed
        @param keep_escaped_markup whether markup that is escaped in the
            original segment should be removed as well, and
            only its text content should be kept
        '''
        # unescaped markup
        if '<' in segment:
            tree = etree.fromstring('<root>' + segment + '</root>')
            segment  = etree.tostring(tree, encoding='unicode', method='text')
        # markup that was escaped in the original segment, now surfaced
        if '<' in segment and not keep_escaped_markup:        
            segment = re.sub('<[^>]*>', '', segment)
        else:
            segment = saxutils.escape(segment)

        return re.sub(' +', ' ', segment).strip()

    def _mask_markup(self, segment):
        '''
        Replaces XML markup with mask tokens.
        @param segment the segment to be masked
        @return the masked segment and the mapping
            between mask tokens and original content
        '''
        return self._masker.mask_segment(segment)

    def _unmask_markup(self, segment, mapping):
        '''
        When a mask token is found, reinsert the original
            XML markup content.
        @param segment a segment with mask tokens
        @param mapping a dictionary containing the mask tokens
            and the original content
        '''
        return self._masker.unmask_segment(segment, mapping)

    def _reinsert_markup(self, source_segment, target_segment):
        '''
        Reinserts XML markup in a segment where markup was
            stripped before translation.
        @param source_segment the original segment in the source language
            before XML markup was removed, but after markup-aware tokenization
        @param target_segment a TranslatedSegment object, containing a translation
            without markup, segmentation and alignment information
        '''
        return self._reinserter.reinsert_markup(
            source_segment,
            target_segment.translation,
            target_segment.segmentation,
            target_segment.alignment
        )

    # Exposed methods
    def preprocess_markup(self, segment):
        '''
        Strips or masks XML markup before translation, depending
            on the markup strategy.
        '''
        if self._xml_strategy in (XML_STRIP, XML_STRIP_REINSERT):
            return self._strip_markup(segment), None
        elif self._xml_strategy == XML_MASK:
            return self._mask_markup(segment)
        elif self._xml_strategy == XML_PASS_THROUGH:
            return segment # then return segment unchanged

    def postprocess_markup(self, source_segment, target_segment, mapping=None):
        '''
        Unmasks or restores XML markup after translation, depending
            on the markup strategy.
        '''
        if self._xml_strategy == XML_STRIP_REINSERT:
            return self._reinsert_markup(source_segment, target_segment)
        elif self._xml_strategy == XML_STRIP:
            # in this case, do nothing
            return target_segment 
        elif self._xml_strategy == XML_MASK:
            return self._unmask_markup(target_segment, mapping)
        elif self._xml_strategy == XML_PASS_THROUGH:
            return segment # then return segment unchanged
