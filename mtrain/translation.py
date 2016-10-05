#!/usr/bin/env python3

from mtrain.constants import *
from mtrain.preprocessing import lowercaser
from mtrain.preprocessing.tokenizer import Tokenizer, Detokenizer
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
        self._tokenizer = Tokenizer(src_lang)
        self._detokenizer = Detokenizer(trg_lang, uppercase_first_letter)
        self._load_engine()
        if self._casing_strategy == TRUECASING:
            self._load_truecaser()
        elif self._casing_strategy == RECASING:
            self._load_recaser()

    def _load_engine(self):
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
        self._engine = ExternalProcessor(
            command=" ".join([MOSES] + arguments),
            stream_stderr=True
        )

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

    def _preprocess_segment(self, segment):
        tokens = self._tokenizer.tokenize(segment)
        if self._casing_strategy == TRUECASING:
            tokens = self._truecaser.truecase_tokens(tokens)
        else:
            tokens = lowercaser.lowercase_tokens(tokens)
        return " ".join(tokens)

    def _postprocess_segment(self, segment, lowercase=False, detokenize=True):
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

    def translate(self, segment, preprocess=True, lowercase=False, detokenize=True):
        '''
        Translates a single segment.
        @param preprocess whether to apply preprocessing steps to segment
        @param lowercase whether to lowercase (True) or restore the original
            casing (False) of the output segment.
        @param detokenize whether to detokenize the translated segment
        '''
        if preprocess:
            segment = self._preprocess_segment(segment)
        translation = self._engine.process(segment)
        return self._postprocess_segment(translation, lowercase, detokenize)
