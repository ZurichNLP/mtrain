#!/usr/bin/env python3

import abc
import fileinput
import logging
import sys
import os

from abc import ABCMeta
from mtrain import inspector
from mtrain.constants import *
from mtrain.engine import EngineMoses, EngineNematus
from mtrain.preprocessing import lowercaser, cleaner
from mtrain.preprocessing.truecaser import Truecaser, Detruecaser
from mtrain.preprocessing.recaser import Recaser
from mtrain.preprocessing.normalizer import Normalizer
from mtrain.preprocessing.tokenizer import Tokenizer, Detokenizer
from mtrain.preprocessing.masking import Masker
from mtrain.preprocessing.xmlprocessor import XmlProcessor
from mtrain.preprocessing.bpe import BytePairEncoderSegment, BytePairDecoderSegment
from mtrain.preprocessing.external import ExternalProcessor

class TranslationEngineBase(object):
    '''
    Abstract class for using translation engine trained with `mtrain`.
    '''
    __metaclass__ = ABCMeta

    def __init__(self, basepath, src_lang, trg_lang):
        '''
        @param basepath the path to the engine, i.e., `mtrain`'s output
            directory (-o).
        @param src_lang the source language
        @param trg_lang the target language
        '''
        self._src_lang = src_lang
        self._trg_lang = trg_lang

    @abc.abstractmethod
    def _load_engine():
        pass

    @abc.abstractmethod
    def _load_tokenizer():
        pass

    def _load_truecaser(self):
        path_model = os.sep.join([
            self._basepath,
            PATH_COMPONENT['engine'],
            TRUECASING,
            'model.%s' % self._src_lang
        ])
        self._truecaser = Truecaser(path_model)

    @abc.abstractmethod
    def _preprocess_segment(self, segment):
        '''
        Preprocesses a single @param segment.
        '''
        pass

    @abc.abstractmethod
    def _postprocess_segment():
        pass

    @abc.abstractmethod
    def translate(self):
        pass

class TranslationEngineMoses(TranslationEngineBase):
    '''
    Moses translation engine trained using `mtrain`
    '''
    def __init__(self, basepath, src_lang, trg_lang, uppercase_first_letter=False, xml_strategy=None,
        quiet=False):
        '''
        In addition to Metaclass @params:
        @param uppercase_first_letter uppercase first letter of translation
        @param xml_strategy how XML is dealt with during translation
        @param quiet if quiet, do not INFO log events
        '''
        super(TranslationEngineMoses, self).__init__(basepath, src_lang, trg_lang)

        assert(inspector.is_mtrain_engine(basepath))
        self._basepath = basepath.rstrip(os.sep)

        self._quiet = quiet
        # set strategies
        self._casing_strategy = inspector.get_casing_strategy(self._basepath, quiet=self._quiet)
        self._masking_strategy = inspector.get_masking_strategy(self._basepath, quiet=self._quiet)
        # if no XML strategy, guess from directory
        if xml_strategy:
            self._xml_strategy = xml_strategy
        else:
            self._xml_strategy = inspector.get_xml_strategy(self._basepath, quiet=self._quiet)
        # load components
        self._load_tokenizer()
        self._detokenizer = Detokenizer(trg_lang, uppercase_first_letter)
        if self._casing_strategy == TRUECASING:
            self._load_truecaser()
        elif self._casing_strategy == RECASING:
            self._load_recaser()
        if self._masking_strategy:
            self._load_masker()
        if self._xml_strategy:
            self._load_xml_processor()
        # load engine
        if self._masking_strategy or self._xml_strategy:
            self._load_engine(report_alignment=True, report_segmentation=True)
        else:
            self._load_engine()

    def _load_engine(self, report_alignment=False, report_segmentation=False):
        '''
        Start a Moses process and keep it running.
        @param report_alignment whether Moses should report word alignments
        @param report_segmentation whether Moses should report how the translation is
            made up of phrases
        '''
        path_moses_ini = os.sep.join([
            self._basepath,
            PATH_COMPONENT['engine'],
            'moses.ini'
        ])
        self._engine = EngineMoses(
            path_moses_ini=path_moses_ini,
            report_alignment=report_alignment,
            report_segmentation=report_segmentation,
        )

    def _load_tokenizer(self):
        '''
        Loads a tokenizer depending on the masking and XML strategies
        guessed from the engine directory.
        '''
        tokenizer_protects = False
        if self._masking_strategy:
            tokenizer_protects = True
            overall_strategy = MASKING
            detailed_strategy = self._masking_strategy
        elif self._xml_strategy:
            tokenizer_protects = True
            overall_strategy = self._xml_strategy
            detailed_strategy = XML_STRATEGIES_DEFAULTS[self._xml_strategy]            

        if tokenizer_protects:
            patterns_path = os.sep.join([
                self._basepath,
                PATH_COMPONENT['engine'],
                overall_strategy,
                detailed_strategy,
                PROTECTED_PATTERNS_FILE_NAME
            ])
            self._tokenizer = Tokenizer(self._src_lang, protect=True, protected_patterns_path=patterns_path, escape=False)
        else:
            self._tokenizer = Tokenizer(self._src_lang)

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
        '''
        No addition to abstract method @params.
        '''
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
        
        if FORCE_MASK_TRANSLATION:
            if self._masking_strategy:
                segment = self._masker.force_mask_translation(segment)
            elif self._xml_strategy == XML_MASK:
                segment = self._xml_processor.force_mask_translation(segment)

        return source_segment, segment, mask_mapping, xml_mapping

    def _postprocess_segment(self, source_segment, target_segment, masked_source_segment=None,
        lowercase=False, detokenize=True, mask_mapping=None,
        xml_mapping=None, strip_markup=False):
        '''
        todo @params
        '''
        if self._masking_strategy is not None:
            target_segment.translation = self._masker.unmask_segment(masked_source_segment, target_segment.translation, mask_mapping)
        if lowercase:
            target_segment.translation = lowercaser.lowercase_string(target_segment.translation)
        else:
            if self._casing_strategy == RECASING:
                target_segment.translation = self._recaser.recase(target_segment.translation)
        if self._xml_strategy is not None:
            target_segment.translation = self._xml_processor.postprocess_markup(source_segment, target_segment, xml_mapping, masked_source_segment)
        if strip_markup:
            target_segment.translation = self._xml_processor._strip_markup(target_segment.translation)
        if detokenize:
            output_tokens = target_segment.translation.split(" ")
            return self._detokenizer.detokenize(output_tokens)
        else:
            return target_segment.translation

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
        Translates a single @param segment.

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
        
        return self._postprocess_segment(
            source_segment=source_segment,
            masked_source_segment=segment,
            target_segment=translated_segment,
            lowercase=lowercase,
            detokenize=detokenize,
            mask_mapping=mask_mapping,
            xml_mapping=xml_mapping
        )

class TranslationEngineNematus(TranslationEngineBase):
    '''
    Nematus translation engine trained using `mtrain`

    ###BH todo add reference to:
        wmt instructions https://github.com/rsennrich/wmt16-scripts/blob/master/sample/README.md
        wmt preprocess.sh, including:
            moses normalize-punctuation.perl
            wmt normalise-romanian.py
            wmt remove-diacritics.py
            moses tokenizer.perl
            moses truecase.perl
            subword_nmt apply_bpe.py
        wmt translate.sh, including:
            nematus translate.py
        wmt postprocess-test.sh, including:
            moses detruecase.perl
            moses detokenizer.perl
    '''
    def __init__(self, basepath, src_lang, trg_lang, adjust_dictionary=False):
        '''
        In addition to Metaclass @params:
        @param adjust_dictionary whether or not dictionary paths in model config
            shall be adjusted.
        '''
        super(TranslationEngineNematus, self).__init__(basepath, src_lang, trg_lang)

        self._basepath = basepath.rstrip(os.sep)

        # load components
        self._load_normalizer()
        self._load_tokenizer()
        self._load_truecaser()
        self._load_encoder()
        self._load_engine()
        self._load_decoder()
        self._load_detruecaser()
        self._load_detokenizer()

        if adjust_dictionary:
            self._adjust_dictionary()

    def _adjust_dictionary(self):
        '''
        Ensure dictionary paths in model config (model.npz.json located in model basepath) match the .json files
        in the basepath's corpus folder. Necessary when models were trained in a path different than
        the current basepath. If paths are not matching, nematus returns an empty string for any translation
        without error message OR may use the wrong .json files for translation.
        '''
        # get path and file name of model config
        model_config = self._path_nematus_model + '.json'
        # get identifiers for finding entries of source and target dictionary
        old_src_json = ".".join([self._src_lang, 'json'])
        old_trg_json = ".".join([self._trg_lang, 'json'])
        # get correct entry in config file for source and target dictionary
        corpus_path = os.sep.join([
            self._basepath,
            PATH_COMPONENT['corpus']
        ])
        new_src_json = '    "' + corpus_path + '/' + ".".join([BASENAME_TRAINING_CORPUS, SUFFIX_TRUECASED, BPE, self._src_lang, 'json",\n'])
        new_trg_json = '    "' + corpus_path + '/' + ".".join([BASENAME_TRAINING_CORPUS, SUFFIX_TRUECASED, BPE, self._trg_lang, 'json"\n'])
        # replace lines with source and target dictionary with correct entries
        with open(model_config) as f:
            old_config = f.readlines()
        f.close()
        with open(model_config, 'w') as f:
            for line in old_config:
                if old_src_json in line:
                    f.write(new_src_json)
                elif old_trg_json in line:
                    f.write(new_trg_json)
                else:
                    f.write(line)
        f.close()

    def _load_normalizer(self):
        '''
        Create normalizer: Additional preprocessing step for backend nematus.
        '''
        self._normalizer = Normalizer(self._src_lang)

    def _load_tokenizer(self):
        '''
        Create tokenizer: So far neither masking_strategy nor xml_strategy for backend nematus.
        '''
        self._tokenizer = Tokenizer(self._src_lang)

    def _load_encoder(self):
        '''
        Create byte-pair encoder: Uses the bpe model learnt in `mtrain`

        ###BH todo add reference to:
            wmt instructions https://github.com/rsennrich/wmt16-scripts/blob/master/sample/README.md
            wmt preprocess.sh, including:
                subword_nmt apply_bpe.py
        '''
        bpe_model_path = os.sep.join([
            self._basepath,
            PATH_COMPONENT['engine'],
            BPE
        ])
        model = bpe_model_path + '/' + self._src_lang + '-' + self._trg_lang + '.bpe'
        self._encoder = BytePairEncoderSegment(model)

    def _load_engine(self):
        '''
        Start a process for Nematus translation engine.

        ###BH todo add reference to:
            wmt instructions https://github.com/rsennrich/wmt16-scripts/blob/master/sample/README.md
            wmt translate.sh, including:
                nematus translate.py
        '''
        self._path_nematus_model = os.sep.join([
            self._basepath,
            PATH_COMPONENT['engine'],
            'tm',
            'model',
            'model.npz'
        ])

        self._engine = EngineNematus(
            self._path_nematus_model
        )

    def _load_decoder(self):
        '''
        Create byte-pair decoder.
        '''
        self._decoder = BytePairDecoderSegment()

    def _load_detruecaser(self):
        '''
        Create detruecaser.
        '''
        self._detruecaser = Detruecaser()

    def _load_detokenizer(self):
        '''
        Create detokenizer.
        '''
        self._detokenizer = Detokenizer(self._trg_lang, uppercase_first_letter=False)

    def _preprocess_segment(self, segment):
        '''
        No addition to abstract method @params.

        ###BH todo add reference to:
            wmt instructions https://github.com/rsennrich/wmt16-scripts/blob/master/sample/README.md
            wmt preprocess.sh, including:
                moses normalize-punctuation.perl
                wmt normalise-romanian.py
                wmt remove-diacritics.py
                moses tokenizer.perl
                moses truecase.perl
                subword_nmt apply_bpe.py
        '''
        # normalize input segment
        segment = self._normalizer.normalize_punctuation(segment)
        # when normalized, Romanian segments need further cleaning from cedillas and diacritics
        # normalize_romanian() must be called before remove_ro_diacritics()
        if self._src_lang == 'ro':
            segment = cleaner.normalize_romanian(segment)
            segment = cleaner.remove_ro_diacritics(segment)
        # tokenize normalized segment
        tokens = self._tokenizer.tokenize(segment)
        # truecase tokens (using truecasing model trained in `mtrain`)
        tokens = self._truecaser.truecase_tokens(tokens)
        # join truecased tokens to a segment
        segment = " ".join(tokens)
        # encode truecased segment (applying byte-pair processing model trained in `mtrain`)
        segment = self._encoder.bpencode_segment(segment)
        # return encoded segment
        return segment

    def _postprocess_segment(self, segment):
        '''
        Postprocesses a single @param segment.

        ###BH todo add reference to:
            wmt instructions https://github.com/rsennrich/wmt16-scripts/blob/master/sample/README.md
            wmt postprocess-test.sh, including:
                moses detruecase.perl
                moses detokenizer.perl
        '''
        # decode translated segment
        segment = self._decoder.bpdecode_segment(segment)
        # detruecase decoded segment
        segment = self._detruecaser.detruecase(segment)
        # split detruecased segment into tokens
        tokens = segment.split(" ")
        # detokenize detruecased tokens
        segment = self._detokenizer.detokenize(tokens)
        # return detokenized segment
        return segment

    def preprocess(self, segment):
        '''
        Method to make preprocessing of a single @param segment accessible from `mtrans`.
        '''
        return self._preprocess_segment(segment)

    def translate(self, device_trans=None, preallocate_trans=None, temp_pre=None, temp_trans=None):
        '''
        Translates an entire text of preprocessed segments.

        @param device_trans defines the processor (cpu, gpuN or cudaN) for translation
        @param preallocate_trans defines the percentage of memory to be preallocated for translation
        @param temp_pre path to temporary file holding preprocessed segments as one text
        @param temp_trans path to temporary file for translated text

        ###BH todo add reference to:
            wmt instructions https://github.com/rsennrich/wmt16-scripts/blob/master/sample/README.md
            wmt translate.sh, including:
                nematus translate.py
        '''
        # translate preprocessed text provided by @param temp_pre,
        # translated text will be available from @param temp_trans, thus no @return
        self._engine.translate_text(device_trans, preallocate_trans, temp_pre, temp_trans)

    def postprocess(self, segment):
        '''
        Method to make postprocessing of a single @param segment accessible from `mtrans`.
        '''
        return self._postprocess_segment(segment)
