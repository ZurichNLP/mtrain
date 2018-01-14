#!/usr/bin/env python3

import logging

from collections import defaultdict
from mtrain import commander
from mtrain.constants import *
from mtrain.preprocessing.external import ExternalProcessor

'''
Specific engines and methods for either backend.

Abstract class only useful when e.g. Nematus processed segments instead
of whole text. As for now, the backends' translation strategy have too little
in common to create abstract class.
'''

class EngineMoses(object):
    '''
    Starts a translation engine process for moses backend and keep it running.
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
        trailing_output = False

        if self._report_alignment:
            arguments.append('-print-alignment-info')
            trailing_output = True
        if self._report_segmentation:
            arguments.append('-report-segmentation')
        
        self._processor = ExternalProcessor(
            command=" ".join([MOSES] + arguments),
            stream_stderr=True,
            trailing_output=trailing_output
        )

    def close(self):
        del self._processor

    def _extract_alignment(self, alignment_string):
        '''
        Transforms a word alignment string into an easily
            accessible dictionary {source: [target, ...], ...}
        @param alignment_string the exact string returned by Moses
            that contains alignment information
        '''
        alignments = defaultdict(list)

        for alignment in alignment_string.strip().split(" "):
            source, target = [int(string) for string in alignment.split("-")]
            alignments[source].append(target)

        return alignments

    def _separate_tokens_from_segmentation(self, translation):
        '''
        Transform phrase segmentation strings into easily
            accessible dictionary:
            {(source start, source end): (target start, target end), ...}
        @param translation a translation string returned by Moses that does not 
            contain word alignments anymore, but phrase segmentation is still
            interspersed
        '''
        tokens = []
        segmentation = {}
        current_phrase_indexes = []
        current_index = 0
            
        for string in translation.split(" "):
            if '|' in string:
                current_segmentation = string.replace('|', '').split("-")
                if len(current_phrase_indexes) == 1:
                    current_phrase_indexes.append(current_phrase_indexes[0]) # duplicate single index
                
                current_key = tuple(int(index) for index in current_segmentation)
                segmentation[current_key] = tuple(int(index) for index in current_phrase_indexes)

                current_phrase_indexes = []
            else:
                if len(current_phrase_indexes) >= 2:
                    current_phrase_indexes.pop()
                current_phrase_indexes.append(str(current_index))
                tokens.append(string)
                current_index += 1

        return tokens, segmentation

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
    
            alignment = self._extract_alignment(parts[1])
                
        if self._report_segmentation:
            tokens, segmentation = self._separate_tokens_from_segmentation(translation)
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

class EngineNematus(object):
    '''
    Starts a translation engine process for nematus backend.

    Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
    '''
    def __init__(self, path_nematus_model):
        '''
        @param path_nematus_model full path to model trained in `mtrain` using backend nematus
        '''
        self._model = path_nematus_model

    def translate_text(self, device_trans, preallocate_trans, temp_pre, temp_trans):
        '''
        @param device_trans defines the processor (cpu, gpuN or cudaN) for translation
        @param preallocate_trans defines the percentage of memory to be preallocated for translation
        @param temp_pre path to temporary file holding preprocessed segments as one text
        @param temp_trans path to temporary file for translated text

        Script reference https://github.com/EdinburghNLP/nematus/blob/master/nematus/translate.py:
            Rico Sennrich, Orhan Firat, Kyunghyun Cho, Alexandra Birch, Barry Haddow, Julian Hitschler, Marcin Junczys-Dowmunt, Samuel Läubli,
            Antonio Valerio Miceli Barone, Jozef Mokry, and Maria Nadejde (2017): Nematus: a Toolkit for Neural Machine Translation.
            In Proceedings of the Software Demonstrations of the 15th Conference of the European Chapter of the Association for Computational
            Linguistics (EACL 2017). Valencia, Spain, pp. 65-68.

        Code directly applied from example shellscript https://github.com/rsennrich/wmt16-scripts/blob/master/sample/translate.sh:
            Rico Sennrich, Barry Haddow, and Alexandra Birch (2016): Edinburgh Neural Machine Translation Systems for WMT 16.
            In Proceedings of the First Conference on Machine Translation (WMT16). Berlin, Germany.

        Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
        '''
        theano_trans_flags = 'THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device={device},on_unused_input=warn,gpuarray.preallocate={preallocate} python2 {script} '.format(
            device=device_trans,
            preallocate=preallocate_trans,
            script=NEMATUS_TRANSLATE
        )
        nematus_trans_options = '-m {model} -i {input} -o {output} -k 12 -n -p 1'.format(
            model=self._model,
            input=temp_pre,
            output=temp_trans
        )
        commander.run(
            '{nematus_command}'.format(
                nematus_command=theano_trans_flags + nematus_trans_options
            ),
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

    def __repr__(self):
        generic_string = super(TranslatedSegment, self).__repr__()
        return "%s\ntranslation:\t%s\nalignment:\t%s\nsegmentation:\t%s" % (generic_string, self.translation, str(self.alignment), str(self.segmentation))
