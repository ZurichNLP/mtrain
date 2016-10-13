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
            stream_stderr=True
        )

    def close(self):
        del self._processor

    def translate_segment(self, segment):
        '''
        Translates a single input segment, @param segment.
        @return a TranslatedSegment object with a translation and,
        optionally, alignments and/or segmentation info
        '''
        translation = self._processor.process(segment)

        if self._report_alignment:
            alignment = []
            parts = translation.split('|||')
            translation = parts[0].strip() # update translation to remove alignment info
            alignment = parts[1].strip().split(" ")
        if self._report_segmentation:
            tokens = []
            segmentation = []
            for string in translation.split(" "):
                if '|' in string:
                    segmentation.append(
                        string.replace('|', '')
                    )
                else:
                    tokens.append(string)
            translation = " ".join(tokens) # update translation to only contain actual tokens
        
        return TranslatedSegment(
            translated_segment=translation,
            alignment=alignment if self._report_alignment else None,
            segmentation=segmentation if self._report_segmentation else None
        )
        
class TranslatedSegment(object):
    '''
    Models a single translated segment together with its word alignments and
    phrase segmentation.
    '''
    def __init__(self, translated_segment, alignment=None, segmentation=None):
        self._segment = translated_segment
        self._alignment = alignment
        self._segmentation = segmentation
