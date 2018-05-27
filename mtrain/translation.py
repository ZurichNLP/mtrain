#!/usr/bin/env python3

import os
import tempfile
import logging
import abc

from abc import ABCMeta

from mtrain import inspector
from mtrain import constants as C
from mtrain import utils
from mtrain.engine import EngineMoses, EngineNematus
from mtrain.preprocessing import lowercaser
from mtrain.preprocessing.truecaser import Truecaser, Detruecaser
from mtrain.preprocessing.recaser import Recaser
from mtrain.preprocessing.normalizer import Normalizer
from mtrain.preprocessing.tokenizer import Tokenizer, Detokenizer
from mtrain.preprocessing.masking import Masker
from mtrain.preprocessing.xmlprocessor import XmlProcessor
from mtrain.preprocessing.bpe import BytePairEncoderSegment, bpe_decode_segment


class TranslationEngineBase(object):
    """
    Abstract class for using translation engine trained with `mtrain`.
    """
    __metaclass__ = ABCMeta

    def __init__(self,
                 basepath,
                 training_config):
        '''
        @param basepath the path to the engine, i.e., `mtrain`'s output
            directory (-o).
        '''
        assert inspector.is_mtrain_engine(basepath)
        self._basepath = basepath.rstrip(os.sep)

        # determine config of trained model, if not given
        if training_config is None:
            self._training_config = utils.load_config_from_basepath(basepath)
        else:
            self._training_config = training_config

        # set strategies for convenience
        self._casing_strategy = self._training_config.caser
        self._masking_strategy = self._training_config.masking
        self._xml_strategy = self._training_config.xml_input

        # set languages for convenience
        self._src_lang = self._training_config.src_lang
        self._trg_lang = self._training_config.trg_lang

        self._components = []

        self._load_engine()
        self._load_components()


    ####################################
    # loading components
    ####################################

    def _load_components(self):
        """
        Loads preprocessing and postprocessing components.
        """
        self._load_tokenizer()
        self._load_detokenizer()

        if self._casing_strategy == C.TRUECASING:
            self._load_truecaser()
            self._load_detruecaser()
        elif self._casing_strategy == C.RECASING:
            self._load_recaser()
        if self._masking_strategy:
            self._load_masker()
        if self._xml_strategy:
            self._load_xml_processor()

        # add all components to self._components list

    def _load_tokenizer(self):
        """
        Loads a tokenizer depending on the masking and XML strategies
        guessed from the engine directory.
        """
        tokenizer_protects = False
        if self._masking_strategy:
            tokenizer_protects = True
            overall_strategy = C.MASKING
            detailed_strategy = self._masking_strategy
        elif self._xml_strategy:
            tokenizer_protects = True
            overall_strategy = self._xml_strategy
            detailed_strategy = C.XML_STRATEGIES_DEFAULTS[self._xml_strategy]

        if tokenizer_protects:
            patterns_path = os.sep.join([
                self._basepath,
                C.PATH_COMPONENT['engine'],
                overall_strategy,
                detailed_strategy,
                C.PROTECTED_PATTERNS_FILE_NAME
            ])
            self._tokenizer = Tokenizer(self._src_lang, protect=True, protected_patterns_path=patterns_path, escape=False)
        else:
            self._tokenizer = Tokenizer(self._src_lang)

        self._components.append(self._tokenizer)

    def _load_detokenizer(self):
        self._detokenizer = Detokenizer(self._trg_lang, uppercase_first_letter=False)
        self._components.append(self._detokenizer)

    def _load_detruecaser(self):
        pass

    def _load_recaser(self):
        path_moses_ini = os.sep.join([
            self._basepath,
            C.PATH_COMPONENT['engine'],
            C.RECASING,
            'moses.ini'
        ])
        self._recaser = Recaser(path_moses_ini)
        self._components.append(self._recaser)

    def _load_masker(self):
        self._masker = Masker(self._masking_strategy)
        self._components.append(self._masker)

    def _load_xml_processor(self):
        self._xml_processor = XmlProcessor(self._xml_strategy)
        self._components.append(self._xml_processor)

    def _load_truecaser(self):
        path_model = os.sep.join([
            self._basepath,
            C.PATH_COMPONENT['engine'],
            C.TRUECASING,
            'model.%s' % self._src_lang
        ])
        self._truecaser = Truecaser(path_model)

        self._components.append(self._truecaser)

    def _load_detruecaser(self):
        """
        Create detruecaser.
        """
        self._detruecaser = Detruecaser()
        self._components.append(self._detruecaser)

    def close(self):
        """
        Deletes references to obsolete objects.
        """
        for component in self._components:
            del component

    ####################################
    # to be defined in subclasses
    ####################################

    @abc.abstractmethod
    def _load_engine(self):
        pass

    @abc.abstractmethod
    def _preprocess_segment(self, segment):
        pass

    @abc.abstractmethod
    def _postprocess_segment(self, segment):
        pass

    @abc.abstractmethod
    def translate_segment(self, segment):
        pass

    @abc.abstractmethod
    def translate_file(self, input_handle, output_handle):
        """
        Translates a whole file given input and output handles.
        """
        pass


class TranslationEngineMoses(TranslationEngineBase):
    """
    Moses translation engine trained using `mtrain`.
    """

    def _load_engine(self):
        """
        Starts a Moses process and keep it running.
        """
        path_moses_ini = os.sep.join([
            self._basepath,
            C.PATH_COMPONENT['engine'],
            'moses.ini'
        ])

        report = self._masking_strategy or self._xml_strategy

        self._engine = EngineMoses(
            path_moses_ini=path_moses_ini,
            report_alignment=report,
            report_segmentation=report,
        )

        self._components.append(self._engine)

    def _preprocess_segment(self, segment):
        # general preprocessing
        tokens = self._tokenizer.tokenize(segment)
        if self._casing_strategy == C.TRUECASING:
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

        if C.FORCE_MASK_TRANSLATION:
            if self._masking_strategy:
                segment = self._masker.force_mask_translation(segment)
            elif self._xml_strategy == C.XML_MASK:
                segment = self._xml_processor.force_mask_translation(segment)

        return source_segment, segment, mask_mapping, xml_mapping

    def _postprocess_segment(self,
                             source_segment,
                             target_segment,
                             masked_source_segment=None,
                             lowercase=False,
                             detokenize=True,
                             mask_mapping=None,
                             xml_mapping=None,
                             strip_markup=False):
        """
        TODO: @params
        """
        if self._masking_strategy is not None:
            target_segment.translation = self._masker.unmask_segment(masked_source_segment, target_segment.translation, mask_mapping)
        if lowercase:
            target_segment.translation = lowercaser.lowercase_string(target_segment.translation)
        else:
            if self._casing_strategy == C.RECASING:
                target_segment.translation = self._recaser.recase(target_segment.translation)
        if self._xml_strategy is not None:
            target_segment.translation = self._xml_processor.postprocess_markup(source_segment, target_segment, xml_mapping, masked_source_segment)
        if strip_markup:
            target_segment.translation = self._xml_processor._strip_markup(target_segment.translation)
        if detokenize:
            output_tokens = target_segment.translation.split(" ")
            return self._detokenizer.detokenize(output_tokens)
        # implicit else
        return target_segment.translation

    def translate_segment(self, segment, preprocess=True, lowercase=False, detokenize=True):
        """
        Translates a single @param segment.

        @param preprocess whether to apply preprocessing steps to segment
        @param lowercase whether to lowercase (True) or restore the original
            casing (False) of the output segment.
        @param detokenize whether to detokenize the translated segment
        """
        if preprocess:
            source_segment, segment, mask_mapping, xml_mapping = self._preprocess_segment(segment)
        else:
            source_segment = segment
            mask_mapping = None
            xml_mapping = None
        # an mtrain.engine.TranslatedSegment object is returned
        translated_segment = self._engine.translate_segment(segment)

        return self._postprocess_segment(
            source_segment=source_segment,
            masked_source_segment=segment,
            target_segment=translated_segment,
            lowercase=lowercase,
            detokenize=detokenize,
            mask_mapping=mask_mapping,
            xml_mapping=xml_mapping
        )

    def translate_file(self, input_handle, output_handle):
        """
        Translates a whole file given input and output handles.

        TODO: use translate_file on the Engine level.
        """
        for line in input_handle:
            segment = line.strip()
            translated_segment = self.translate_segment(segment)
            output_handle.write(translated_segment + "\n")


class TranslationEngineNematus(TranslationEngineBase):
    """
    Nematus translation engine trained using `mtrain`.
    """

    def __init__(self, basepath, training_config, device, preallocate, keep_temp_files=False):
        """
        """
        self._device = device
        self._preallocate = preallocate
        self._keep_temp_files = keep_temp_files

        super(TranslationEngineNematus, self).__init__(basepath, training_config)

    def _load_components(self):
        """
        Loads additional components.
        """
        super(TranslationEngineNematus, self)._load_components()
        self._load_normalizer()
        self._load_bpe_encoder()

    def _load_normalizer(self):
        """
        Creates normalizer.
        """
        self._normalizer = Normalizer(self._src_lang)
        self._components.append(self._normalizer)

    def _load_bpe_encoder(self):
        """
        Creates byte-pair encoder. Uses a trained BPE model.
        """
        bpe_model_path = os.sep.join([
            self._basepath,
            C.PATH_COMPONENT['engine'],
            C.BPE
        ])
        model = os.sep.join([bpe_model_path, "%s-%s.bpe" % (self._src_lang, self._trg_lang)])
        vocab_source_path = os.sep.join([bpe_model_path, "vocab.%s" % self._src_lang])

        self._bpe_encoder = BytePairEncoderSegment(model, vocab_source_path)

        self._components.append(self._bpe_encoder)

    def _load_engine(self):
        """
        Starts a process that holds a Nematus translation engine.
        """
        self._path_nematus_model = os.sep.join([
            self._basepath,
            C.PATH_COMPONENT['engine'],
            'tm',
            'model',
            'model.npz'
        ])

        self._engine = EngineNematus(model_path=self._path_nematus_model,
                                     device=self._device,
                                     preallocate=self._preallocate)
        self._components.append(self._engine)

    def _preprocess_segment(self, segment):
        """
        Preprocesses segments. Specifically: normalization, tokenization,
        truecasing and applying BPE.

        TODO: only truecase if that's the requested strategy
        """
        segment = self._normalizer.normalize_punctuation(segment)
        segment = self._tokenizer.tokenize(segment, split=False)
        segment = self._truecaser.truecase_segment(segment)
        segment = self._bpe_encoder.encode_segment(segment)

        return segment

    def _postprocess_segment(self, segment):
        """
        Postprocesses a single @param segment.

        TODO: only detruecase if input was truecased
        """
        segment = bpe_decode_segment(segment)
        segment = self._detruecaser.detruecase_segment(segment)
        tokens = segment.split(" ")
        segment = self._detokenizer.detokenize(tokens)

        return segment

    def _postprocess_file(self, input_handle, output_handle):
        """
        """
        for line in input_handle:
            segment = line.strip()
            postprocessed_segment = self._postprocess_segment(segment)
            output_handle.write(postprocessed_segment + "\n")

    def _preprocess_file(self, input_handle, output_handle):
        """
        """
        for line in input_handle:
            logging.debug("Line: '%s'", line)
            segment = line.strip()
            preprocessed_segment = self._preprocess_segment(segment)
            output_handle.write(preprocessed_segment)

    def translate_segment(self, segment):
        """
        Currently not possible because of a limitation of the Nematus
        script translate.py.
        """
        raise NotImplementedError

    def translate_file(self, input_handle, output_handle):
        """
        Translates a whole file given input and output handles.
        """
        tempdir = tempfile.mkdtemp()
        preprocessed_handle = tempfile.NamedTemporaryFile(prefix="preprocessed.", dir=tempdir, mode="w", encoding="utf-8", delete=False)
        translated_handle = tempfile.NamedTemporaryFile(prefix="translated.", dir=tempdir, mode="w", delete=False)

        preprocessed_path = preprocessed_handle.name
        translated_path = translated_handle.name

        logging.debug("tempdir=%s, preprocessed_path=%s, translated_path=%s", tempdir, preprocessed_path, translated_path)

        # takes the overall input handle because first step
        self._preprocess_file(input_handle=input_handle, output_handle=preprocessed_handle)

        preprocessed_handle.close()
        translated_handle.close()

        self._engine.translate_file(input_path=preprocessed_path,
                                    output_path=translated_path)

        translated_handle = open(translated_path, "r")

        # takes the overall output handle because last step
        self._postprocess_file(input_handle=translated_handle, output_handle=output_handle)

        # only close remaining handles if opened by this method
        translated_handle.close()

        if not self._keep_temp_files:
            os.remove(preprocessed_path)
            os.remove(translated_path)
            os.rmdir(tempdir)
