#!/usr/bin/env python3

import logging
from mtrain.constants import *
from mtrain.preprocessing.external import ExternalProcessor

'''
Translates preprocessed segments.
'''

class Engine(object):
    '''
    Starts a translation engine process and keep it running.
    '''
    def __init__(self, path_moses_ini, report_alignment=False, report_segmentation=False):
        '''
        @param path_moses_ini path to Moses configuration file
        @param report_alignment whether Moses should report word alignments
        @param report_segmentation whether Moses should report how the translation
            is made up of phrases
        '''
        self._path_moses_ini = path_moses_ini
        self._report_alignment = report_alignment
        self._report_segmentation = report_segmentation

        arguments = [
            '-f %s' % self._path_moses_ini,
            '-minphr-memory', # compact phrase table
            '-minlexr-memory', # compact reordering table
            '-v 0', # as quiet as possible
            '-xml-input constraint' # allow forced translations and zones
        ]
        if self._report_alignment:
            arguments.append('-print-alignment-info')
        if self._report_segmentation:
            arguments.append('-report-segmentation')
        
        self._processor = ExternalProcessor(
            command=" ".join([MOSES] + arguments),
            stream_stderr=True,
            trailing_output=True
        )

    def close(self):
        del self._processor

    def _untangle_translation(self, translation):
        '''
        Separates the actual translation from reported segmentation
            and word alignments. Changes slightly the segmentation info
            by adding information about the source tokens.
        @param translation the exact string returned by a Moses engine
        '''
        if self._report_alignment:
            alignment = []
            parts = translation.split('|||')
            translation = parts[0].strip() # update translation to remove alignment info
            alignment = parts[1].strip().split(" ")
        if self._report_segmentation:
            tokens = []
            segmentation = []
            current_phrase_indexes = []
            current_index = 0
            for string in translation.split(" "):
                if '|' in string:
                    current_segmentation = string.replace('|', '')
                    if len(current_phrase_indexes) == 1:
                        current_phrase_indexes.append(current_phrase_indexes[0]) # duplicate single index
                    current_phrase_indexes = "-".join(current_phrase_indexes)
                    segmentation.append(
                        "|".join([current_phrase_indexes, current_segmentation])
                    )
                    current_phrase_indexes = []
                else:
                    current_phrase_indexes.append(str(current_index))
                    tokens.append(string)
                    current_index += 1
            translation = " ".join(tokens) # update translation to only contain actual tokens

        return (
            translation,
            alignment if self._report_alignment else None,
            segmentation if self._report_segmentation else None
        ) 

    def translate_segment(self, segment):
        '''
        Translates a single input segment, @param segment.
        @return a TranslatedSegment object with a translation and,
        optionally, alignments and/or segmentation info
        '''
        translation = self._processor.process(segment)
        translation, alignment, segmentation = self._untangle_translation(translation)

        return TranslatedSegment(
            translated_segment=translation,
            alignment=alignment,
            segmentation=segmentation
        )
        
class TranslatedSegment(object):
    '''
    Models a single translated segment together with its word alignments and
    phrase segmentation.
    '''
    def __init__(self, translated_segment, alignment=None, segmentation=None):
        self.translation = translated_segment
        self.alignment = alignment
        self.segmentation = segmentation
