#!/usr/bin/env python3

from mtrain.constants import *
from mtrain.preprocessing import lowercaser
from mtrain.preprocessing.tokenizer import Tokenizer, Detokenizer
from mtrain.preprocessing.masking import Masker
from mtrain.preprocessing.truecaser import Truecaser
from mtrain.preprocessing.external import ExternalProcessor
from mtrain.preprocessing.recaser import Recaser
from mtrain import inspector

class TranslationEngine(object):
    '''
    An engine trained using `mtrain`
    '''
    def __init__(self, basepath, src_lang, trg_lang, uppercase_first_letter=False):
        '''
        @param basepath the path to the engine, i.e., `mtrain`'s output
            directory (-o).
        '''
        assert(inspector.is_mtrain_engine(basepath))
        self._basepath = basepath.rstrip(os.sep)
        self._src_lang = src_lang
        self._trg_lang = trg_lang
        self._casing_strategy = inspector.get_casing_strategy(self._basepath)
        self._masking_strategy = inspector.get_masking_strategy(self._basepath)
        self._load_tokenizer()
        self._detokenizer = Detokenizer(trg_lang, uppercase_first_letter)
        self._load_engine()
        if self._casing_strategy == TRUECASING:
            self._load_truecaser()
        elif self._casing_strategy == RECASING:
            self._load_recaser()
        if self._masking_strategy is not None:
            self._load_masker(self._masking_strategy)

    def _load_engine(self, word_alignment=False, phrase_segmentation=False):
        '''
        Start a Moses process and keep it running.
        @param word_alignment whether Moses should report word alignments
        @param phrase_segmentation whether Moses should report how the translation is
            made up of phrases
        '''
        path_moses_ini = os.sep.join([
            self._basepath,
            PATH_COMPONENT['engine'],
            'moses.ini'
        ])
        arguments = [
            '-f %s' % path_moses_ini,
            '-minphr-memory',
            '-minlexr-memory',
            '-v 0',
        ]
        if word_alignment:
            arguments.append('-print-alignment-info')
        if phrase_segmentation:
            arguments.append('-report-segmentation')

        self._engine = ExternalProcessor(
            command=" ".join([MOSES] + arguments)
        )

    def _load_tokenizer(self):
        if self._masking_strategy is not None:
            patterns_path = os.sep.join([
                self._basepath,
                PATH_COMPONENT['engine'],
                MASKING,
                self._masking_strategy,
                PROTECTED_PATTERNS_FILE_NAME
            ])
            
            self._tokenizer = Tokenizer(self._src_lang, protect=True, protected_patterns_path=patterns_path, escape=False)
        else:
            self._tokenizer = Tokenizer(self._src_lang)

    def _load_truecaser(self):
        path_model = os.sep.join([
            self._basepath,
            PATH_COMPONENT['engine'],
            TRUECASING,
            'model.%s' % self._src_lang
        ])
        self._truecaser = Truecaser(path_model)

    def _load_recaser(self):
        path_moses_ini = os.sep.join([
            self._basepath,
            PATH_COMPONENT['engine'],
            RECASING,
            'moses.ini'
        ])
        self._recaser = Recaser(path_moses_ini)

    def _load_masker(self, strategy):
        self._masker = Masker(strategy)

    def _preprocess_segment(self, segment):
        tokens = self._tokenizer.tokenize(segment)
        if self._masking_strategy is not None:
            tokens, mapping = self._masker.mask_tokens(tokens)
        else:
            mapping = None
        if self._casing_strategy == TRUECASING:
            tokens = self._truecaser.truecase_tokens(tokens)
        else:
            tokens = lowercaser.lowercase_tokens(tokens)
        return " ".join(tokens), mapping

    def _postprocess_segment(self, segment, lowercase=False, detokenize=True, mapping=None):
        if self._masking_strategy is not None:
            segment = self._masker.unmask_segment(segment, mapping)
        if lowercase:
            segment = lowercaser.lowercase_string(segment)
        else:
            if self._casing_strategy == RECASING:
                segment = self._recaser.recase(segment)
        if detokenize:
            output_tokens = segment.split(" ")
            return self._detokenizer.detokenize(output_tokens)
        else:
            return segment

    def close(self):
        del self._engine
        if self._casing_strategy == TRUECASING:
            self._truecaser.close()
        elif self._casing_strategy == RECASING:
            self._recaser.close()
        if self._masking_strategy is not None:
            del self._masker

    def translate(self, segment, preprocess=True, lowercase=False, detokenize=True):
        '''
        Translates a single segment.
        @param preprocess whether to apply preprocessing steps to segment
        @param lowercase whether to lowercase (True) or restore the original
            casing (False) of the output segment.
        @param detokenize whether to detokenize the translated segment
        '''
        if preprocess:
            segment, mapping = self._preprocess_segment(segment)
        translation = self._engine.process(segment)
        return self._postprocess_segment(translation, lowercase, detokenize, mapping)
