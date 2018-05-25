#!/usr/bin/env python3

"""
Wraps sub-word nmt to provide byte-pair encoding (BPE) as a preprocessing step.
"""
import os

from mtrain import constants as C
from mtrain import commander
from mtrain.preprocessing.external import ExternalProcessor


class BytePairEncoderFile(object):
    """
    Learns a BPE model and applies it to files.
    """
    def __init__(self, corpus_train, corpus_tune, model_path, num_operations, src_lang, trg_lang):
        """
        @params corpus_train location of  training corpus (no language ending)
        @params corpus_tune location of truecased tuning corpus (no language ending)
        @params model_path path of byte-pair encoding model to be stored (no filename)
        @params num_operations number of merging operations
        @params src_lang language identifier of source language
        @params trg_lang language identifier of target language
        """
        self._corpus_train = corpus_train
        self._corpus_tune = corpus_tune
        self._model_path = model_path
        self._num_operations = num_operations
        self._src_lang = src_lang
        self._trg_lang = trg_lang

        self._vocab_path = self._model_path + os.sep + "vocab"

    def learn_bpe_model(self):
        """
        Learns BPE model on preprocessed training corpus.
        Stores the bpe model in the basepath's subfolder 'engine/bpe/'.
        """
        blueprint = '{script} --input {corpus}.{src} {corpus}.{trg} --write-vocabulary {vocab}.{src} {vocab}.{trg} --symbols {bpe_ops} > {bpe_model}/{src}-{trg}.bpe'

        commander.run(blueprint.format(
                corpus=self._corpus_train,
                script=C.SUBWORD_NMT_JOINT,
                vocab=self._vocab_path,
                bpe_ops=self._num_operations,
                bpe_model=self._model_path,
                src=self._src_lang,
                trg=self._trg_lang
            ),
            "Learning BPE model: %s operations" % self._num_operations
        )

    def apply_bpe_model(self):
        """
        Applies BPE to training and tuning corpora.
        """
        def command(current_corpus, current_lang):

            blueprint = '{script} -c {bpe_model}/{src}-{trg}.bpe --vocabulary vocab.{lang} --vocabulary-threshold {threshold} < {corpus}.{lang} > {corpus}.bpe.{lang}'

            return blueprint.format(script=C.SUBWORD_NMT_APPLY,
                                    bpe_model=self._model_path,
                                    src=self._src_lang,
                                    trg=self._trg_lang,
                                    threshold=C.BPE_VOCAB_THRESHOLD,
                                    corpus=current_corpus,
                                    lang=current_lang)
        commands = [
            command(self._corpus_train, self._src_lang),
            command(self._corpus_train, self._trg_lang),
            command(self._corpus_tune, self._src_lang),
            command(self._corpus_tune, self._trg_lang),
        ]
        commander.run_parallel(commands, "Applying BPE model.")

    def build_bpe_dictionary(self):
        """
        Builds BPE vocabulary files (JSON) for a training corpus.
        Note that the JSON files such as train.truecased.bpe.SRC.json and train.truecased.bpe.TRG.json are automatically stored at the location
        of the input files, which is the basepath's subfolder 'corpus'.
        """
        commander.run(
            '{script} {corpus}.bpe.{src} {corpus}.bpe.{trg}'.format(
                script=C.NEMATUS_BUILD_DICT,
                corpus=self._corpus_train,
                src=self._src_lang,
                trg=self._trg_lang,
            ),
            "Building network dictionary from BPE model"
        )


class BytePairEncoderSegment(object):
    """
    Learns a BPE model and applies it to individual segments.
    """
    def __init__(self, bpe_model):
        """
        @param bpe_model full path to byte-pair processing model trained in `mtrain`
        """
        arguments = [
            '-c %s' % bpe_model
        ]
        # the subword script apply_bpe.py needs to be run in a Python 3 environment,
        # a constant is used to avoid version problems
        self._processor = ExternalProcessor(
            command=" ".join([C.PYTHON3] + [C.SUBWORD_NMT_APPLY] + arguments),
            stream_stderr=False,
            trailing_output=False,
            shell=False
        )

    def close(self):
        del self._processor

    def bpencode_segment(self, segment):
        '''
        Encodes a single @param segment by applying a trained BPE model.
        '''
        encoded_segment = self._processor.process(segment)
        return encoded_segment


class BytePairDecoderSegment(object):
    """
    Removes BPE from segments.
    """
    def bpdecode_segment(self, segment):
        """
        Decodes a single @param segment.
        """
        return segment.replace("@@ ", "")
