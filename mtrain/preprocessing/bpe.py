#!/usr/bin/env python3

from mtrain import constants as C
from mtrain import commander
from mtrain.preprocessing.external import ExternalProcessor

"""
Wraps sub-word nmt to provide byte-pair encoding (BPE) as a preprocessing step.
"""


class BytePairEncoderFile(object):
    """
    Learns a BPE model and applies it to files.
    """
    def __init__(self, corpus_train_tc, corpus_tune_tc, bpe_model_path, bpe_operations, src_lang, trg_lang):
        """
        @params corpus_train_tc location of truecased training corpus (no language ending)
        @params corpus_tune_tc location of truecased tuning corpus (no language ending)
        @params bpe_model_path path of byte-pair encoding model to be stored (no filename)
        @params bpe_operations number of n-grams to be created
        @params src_lang language identifier of source language
        @params trg_lang language identifier of target language
        """
        self._corpus_train_tc = corpus_train_tc
        self._corpus_tune_tc = corpus_tune_tc
        self._bpe_model_path = bpe_model_path
        self._bpe_operations = bpe_operations
        self._src_lang = src_lang
        self._trg_lang = trg_lang

    def learn_bpe_model(self):
        """
        Learn bpe model using truecased training corpus.
        Stores the bpe model in the basepath's subfolder 'engine/bpe/', model name SRC-TRG.bpe.
        """
        commander.run(
            'cat {corpus}.{src} {corpus}.{trg} | {script} --s {bpe_ops} > {bpe_model}/{src}-{trg}.bpe'.format(
                corpus=self._corpus_train_tc,
                script=C.SUBWORD_NMT_LEARN,
                bpe_ops=self._bpe_operations,
                bpe_model=self._bpe_model_path,
                src=self._src_lang,
                trg=self._trg_lang
            ),
            "Learning BPE model: %s operations" % self._bpe_operations
        )

    def apply_bpe_model(self):
        """
        Applies BPE to training and tuning corpora.
        """
        def command(current_corpus, current_lang):

            blueprint = '{script} -c {bpe_model}/{src}-{trg}.bpe < {corpus}.{lang} > {corpus}.bpe.{lang}'

            return blueprint.format(script=C.SUBWORD_NMT_APPLY,
                                    bpe_model=self._bpe_model_path,
                                    src=self._src_lang,
                                    trg=self._trg_lang,
                                    corpus=current_corpus,
                                    lang=current_lang)
        commands = [
            command(self._corpus_train_tc, self._src_lang),
            command(self._corpus_train_tc, self._trg_lang),
            command(self._corpus_tune_tc, self._src_lang),
            command(self._corpus_tune_tc, self._trg_lang),
        ]
        commander.run_parallel(commands, "Applying BPE model")

    def build_bpe_dictionary(self):
        """
        Builds BPE vocabulary files (JSON) for a training corpus.
        Note that the JSON files such as train.truecased.bpe.SRC.json and train.truecased.bpe.TRG.json are automatically stored at the location
        of the input files, which is the basepath's subfolder 'corpus'.
        """
        commander.run(
            '{script} {corpus}.bpe.{src} {corpus}.bpe.{trg}'.format(
                script=C.NEMATUS_BUILD_DICT,
                corpus=self._corpus_train_tc,
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
