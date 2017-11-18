#!/usr/bin/env python3

from mtrain.constants import *
from mtrain import commander
from mtrain.preprocessing.external import ExternalProcessor

'''
###BH check for completeness, esp. dedication
Provides further processing steps in order to use neural network training enabled by backend nematus.
Processing steps are implemented according to guidance in https://github.com/rsennrich/wmt16-scripts/tree/master/sample

'''

class Encoder(object):
    '''
    Further preprocessing for nematus backend by byte-pair encoding the given parallel corpora.
    The encoding is limited to input that is already processed using 'truecased' casing strategy.
    Furthermore, neither generic masking nor XML masking are applicable.
    '''

    def __init__(self, corpus_train_tc, corpus_eval_tc, bpe_model_path, bpe_operations, src_lang, trg_lang, evaluation):
        '''
        @params corpus_train_tc location of truecased training corpus (no language ending)
        @params corpus_eval_tc location of truecased evaluation corpus (no language ending)
        @params bpe_model_path path of byte-pair encoding model to be stored (no filename)
        @params bpe_operations number of n-grams to be created ###BH ref?
        @params src_lang language identifier of source language
        @params trg_lang language identifier of target language
        @params evaluation whether a evaluation corpus exists that may be included in encoding
        '''

        self._corpus_train_tc=corpus_train_tc
        self._corpus_eval_tc=corpus_eval_tc
        self._bpe_model_path=bpe_model_path
        self._bpe_operations=bpe_operations
        self._src_lang=src_lang
        self._trg_lang=trg_lang
        self._evaluation=evaluation


    def learn_bpe_model(self):
        '''
        Learn bpe model using truecased training corpus.
        Stores the bpe model in the basepath's subfolder 'engine/bpe/', model name SRC-TRG.bpe.

        Script reference https://github.com/rsennrich/subword-nmt/blob/master/learn_bpe.py:
            Rico Sennrich, Barry Haddow and Alexandra Birch (2016). Neural Machine Translation of Rare Words with Subword Units.
            Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
        '''

        commander.run(
            'cat {corpus}.{src} {corpus}.{trg} | {script} --s {bpe_ops} > {bpe_model}/{src}-{trg}.bpe'.format(
                corpus=self._corpus_train_tc,   # path of the truecased training corpus, no language ending
                script=SUBWORD_NMT_LEARN,
                bpe_ops=self._bpe_operations,
                bpe_model=self._bpe_model_path, # only bpe_model path, not file
                src=self._src_lang,
                trg=self._trg_lang
            ),
            "Learning BPE model: %s operations" % self._bpe_operations
        )

    def apply_bpe_model(self):
        '''
        Apply bpe model on truecased training corpus (and if present, truecased evaluation corpus).
        Creates the files in the basepath's subfolder 'corpus', file names such as train.truecased.bpe.SRC, train.truecased.bpe.TRG
        (and eval.truecased.bpe.SRC, eval.truecased.bpe.TRG if applicable).

        Script reference https://github.com/rsennrich/subword-nmt/blob/master/apply_bpe.py:
            Rico Sennrich, Barry Haddow and Alexandra Birch (2015). Neural Machine Translation of Rare Words with Subword Units.
            Proceedings of the 54th Annual Meeting of the Association for Computational Linguistics (ACL 2016). Berlin, Germany.
        '''

        def command(current_corpus, current_lang):
            return '{script} -c {bpe_model}/{src}-{trg}.bpe < {corpus}.{lang} > {corpus}.bpe.{lang}'.format(
            script=SUBWORD_NMT_APPLY,
            bpe_model=self._bpe_model_path,     # only bpe_model path, not file
            src=self._src_lang,                 # for getting correct bpe_model file name
            trg=self._trg_lang,                 # for getting correct bpe_model file name
            corpus=current_corpus,              # either truecased training or evaluation corpus, depending on commander
            lang=current_lang                   # either source or target language, depending on commander
            )

        commands = [
            command(self._corpus_train_tc, self._src_lang),
            command(self._corpus_train_tc, self._trg_lang),
        ]
        if self._evaluation:
            commands.append(command(self._corpus_eval_tc, self._src_lang))
            commands.append(command(self._corpus_eval_tc, self._trg_lang))
        commander.run_parallel(commands, "Applying BPE model")

    def build_bpe_dictionary(self):
        '''
        Build bpe dictionary (JSON files) for truecased training corpus.
        Note that the JSON files such as train.truecased.bpe.SRC.json and train.truecased.bpe.TRG.json
        are automatically stored at the location of the input files, which is the basepath's subfolder 'corpus'.

        Scipt reference https://github.com/EdinburghNLP/nematus/blob/master/data/build_dictionary.py
        '''

        commander.run(
            '{script} {corpus}.bpe.{src} {corpus}.bpe.{trg}'.format(
                script=NEMATUS_BUILD_DICT,
                corpus=self._corpus_train_tc,   # path of the truecased training corpus, no language ending
                src=self._src_lang,
                trg=self._trg_lang,
            ),
            "Building network dictionary from BPE model"
        )
