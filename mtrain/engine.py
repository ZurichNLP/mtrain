#!/usr/bin/env python3

"""
Translation engine processes.
"""

from collections import defaultdict
from mtrain import commander
from mtrain import constants as C
from mtrain.preprocessing.external import ExternalProcessor


class EngineMoses(object):
    """
    Starts a translation engine process for moses backend and keep it running.
    """
    def __init__(self, path_moses_ini, report_alignment=False, report_segmentation=False):
        """
        @param path_moses_ini path to Moses configuration file
        @param report_alignment whether Moses should report word alignments
        @param report_segmentation whether Moses should report how the translation
            is made up of phrases
        """
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
            command=" ".join([C.MOSES] + arguments),
            stream_stderr=True,
            trailing_output=trailing_output
        )

    def close(self):
        del self._processor

    def _extract_alignment(self, alignment_string):
        """
        Transforms a word alignment string into an easily
            accessible dictionary {source: [target, ...], ...}
        @param alignment_string the exact string returned by Moses
            that contains alignment information
        """
        alignments = defaultdict(list)

        for alignment in alignment_string.strip().split(" "):
            source, target = [int(string) for string in alignment.split("-")]
            alignments[source].append(target)

        return alignments

    def _separate_tokens_from_segmentation(self, translation):
        """
        Transform phrase segmentation strings into easily
            accessible dictionary:
            {(source start, source end): (target start, target end), ...}
        @param translation a translation string returned by Moses that does not
            contain word alignments anymore, but phrase segmentation is still
            interspersed
        """
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
        """
        Separates the actual translation from reported segmentation
            and word alignments. Changes slightly the segmentation info
            by adding information about the source tokens.
        @param translation the exact string returned by a Moses engine
        """
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
        """
        Translates a single input segment, @param segment.

        @return a TranslatedSegment object with a translation and,
        optionally, alignments and/or segmentation info
        """
        translation = self._processor.process(segment)
        translation, alignment, segmentation = self._untangle_translation(translation)

        return TranslatedSegment(
            translated_segment=translation,
            alignment=alignment,
            segmentation=segmentation
        )

    def translate_file(self, input_path, output_path):
        """
        Translates an entire file.

        @param input_path path to temp file with preprocessed input segments
        @param output_path path to temp file were raw translations should be written
        """

        raise NotImplementedError


class EngineNematus(object):
    """
    Starts a translation engine process for a Nematus backend.
    """
    def __init__(self, model_path, device, preallocate):
        """
        @param model_path full path to model trained in `mtrain` using backend nematus
        @param device GPU or CPU device for translation
        @param preallocate preallocate memory on a GPU device
        """
        self._model_path = model_path
        self._device = device
        self._preallocate = preallocate

    def translate_file(self, input_path, output_path):
        """
        Translates an entire file.

        @param input_path path to temp file with preprocessed input segments
        @param output_path path to temp file were raw translations should be written
        """
        theano_trans_flags = 'THEANO_FLAGS=mode=FAST_RUN,floatX=float32,device={device},on_unused_input=warn,gpuarray.preallocate={preallocate} python2 {script} '.format(
            device=self._device,
            preallocate=self._preallocate,
            script=C.NEMATUS_TRANSLATE
        )
        nematus_trans_options = '-m {model} -i {input} -o {output} -k 12 -n -p 1'.format(
            model=self._model_path,
            input=input_path,
            output=output_path
        )
        commander.run(
            '{nematus_command}'.format(
                nematus_command=theano_trans_flags + nematus_trans_options
            ),
        )

class TranslatedSegment(object):
    """
    Models a single translated segment together with its word alignments and
    phrase segmentation.
    """
    def __init__(self, translated_segment, alignment=None, segmentation=None):
        self.translation = translated_segment
        self.alignment = alignment
        self.segmentation = segmentation

    def __repr__(self):
        generic_string = super(TranslatedSegment, self).__repr__()
        return "%s\ntranslation:\t%s\nalignment:\t%s\nsegmentation:\t%s" % (generic_string, self.translation, str(self.alignment), str(self.segmentation))
