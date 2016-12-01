#!/usr/bin/env python3

from mtrain.constants import *
from mtrain.engine import Engine
from mtrain.preprocessing import lowercaser
from mtrain.preprocessing.tokenizer import Tokenizer, Detokenizer
from mtrain.preprocessing.masking import Masker
from mtrain.preprocessing.xmlprocessor import XmlProcessor
from mtrain.preprocessing.truecaser import Truecaser
from mtrain.preprocessing.external import ExternalProcessor
from mtrain.preprocessing.recaser import Recaser
from mtrain import inspector

class TranslationEngine(object):
    '''
    An engine trained using `mtrain`
    '''
    def __init__(self, basepath, src_lang, trg_lang, uppercase_first_letter=False, xml_strategy=None):
        '''
        @param basepath the path to the engine, i.e., `mtrain`'s output
            directory (-o).
        '''
        assert(inspector.is_mtrain_engine(basepath))
        self._basepath = basepath.rstrip(os.sep)
        self._src_lang = src_lang
        self._trg_lang = trg_lang
        # set strategies
        self._casing_strategy = inspector.get_casing_strategy(self._basepath)
        self._masking_strategy = inspector.get_masking_strategy(self._basepath)
        self._xml_strategy = xml_strategy
        # load components
        self._load_tokenizer()
        self._detokenizer = Detokenizer(trg_lang, uppercase_first_letter)
        self._load_engine()
        if self._casing_strategy == TRUECASING:
            self._load_truecaser()
        elif self._casing_strategy == RECASING:
            self._load_recaser()
        if self._masking_strategy:
            self._load_masker()
        if self._xml_strategy:
            self._load_xml_processor()

    def _load_engine(self, report_alignment=False, report_segmentation=False):
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
        self._engine = Engine(
            path_moses_ini=path_moses_ini,
            report_alignment=report_alignment,
            report_segmentation=report_segmentation
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

    def _load_masker(self):
        self._masker = Masker(self._masking_strategy)

    def _load_xml_processor(self):
        self._xml_processor = XmlProcessor(self._xml_strategy)

    def _preprocess_segment(self, segment):
        # general preprocessing
        tokens = self._tokenizer.tokenize(segment)
        if self._casing_strategy == TRUECASING:
            tokens = self._truecaser.truecase_tokens(tokens)
        else:
             tokens = lowercaser.lowercase_tokens(tokens)
        source_segment = " ".join(tokens)
        segment = source_segment
        # related to masking and markup
        if self._masking_strategy is not None:
            segment, mask_mapping = self._masker.mask_segment(segment)
        else:
            mask_mapping = None
        if self._xml_strategy is not None:
            segment, xml_mapping = self._xml_processor.preprocess_markup(segment)
        else:
            xml_mapping = None
        return source_segment, segment, mask_mapping, xml_mapping

    def _postprocess_segment(self, source_segment, target_segment, lowercase=False,
                             detokenize=True, mask_mapping=None, xml_mapping=None):
        if self._masking_strategy is not None:
            target_segment = self._masker.unmask_segment(target_segment, mask_mapping)
        if self._xml_strategy is not None:
            target_segment = self._xml_processor.postprocess_markup(source_segment, target_segment, xml_mapping)
        if lowercase:
            target_segment = lowercaser.lowercase_string(target_segment)
        else:
            if self._casing_strategy == RECASING:
                target_segment = self._recaser.recase(target_segment)
        if detokenize:
            output_tokens = target_segment.split(" ")
            return self._detokenizer.detokenize(output_tokens)
        else:
            return target_segment

    def close(self):
        del self._engine
        if self._casing_strategy == TRUECASING:
            self._truecaser.close()
        elif self._casing_strategy == RECASING:
            self._recaser.close()
        if self._masking_strategy is not None:
            del self._masker
        if self._xml_strategy is not None:
            del self._xml_processor

    def translate(self, segment, preprocess=True, lowercase=False, detokenize=True):
        '''
        Translates a single segment.
        @param preprocess whether to apply preprocessing steps to segment
        @param lowercase whether to lowercase (True) or restore the original
            casing (False) of the output segment.
        @param detokenize whether to detokenize the translated segment
        '''
        if preprocess:
            source_segment, segment, mask_mapping, xml_mapping = self._preprocess_segment(segment)
        else:
            source_segment = segment
            mask_mapping = None
            xml_mapping = None
        # an mtrain.engine.TranslatedSegment object is returned
        translated_segment = self._engine.translate_segment(segment)
        translation = translated_segment.translation
        
        return self._postprocess_segment(source_segment, translation, lowercase, detokenize,
                                         mask_mapping, xml_mapping)