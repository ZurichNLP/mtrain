#!/usr/bin/env python3

from mtrain.constants import *
from mtrain import commander
from mtrain.preprocessing.external import ExternalProcessor

'''
###BH todo text
'''

class Encoder(object):
    '''
    ###BH text working with truecased only, as in wmt
    '''

    def __init__(self, corpus_train_tc, corpus_eval_tc, bpe_model_path, bpe_operations, src_lang, trg_lang, tuning):
        '''
        ###BH todo all @params
        '''

        self._corpus_train_tc=corpus_train_tc
        self._corpus_eval_tc=corpus_eval_tc
        self._bpe_model_path=bpe_model_path
        self._bpe_operations=bpe_operations
        self._src_lang=src_lang
        self._trg_lang=trg_lang
        self._tuning=tuning


    def learn_bpe_model(self):
        '''
        ###BH todo text
        '''
        commander.run(
            'cat {corpus}.{src} {corpus}.{trg} | {script} --s {bpe_ops} > {bpe_model}/{src}-{trg}.bpe'.format(
                corpus=self._corpus_train_tc,
                script=SUBWORD_NMT_LEARN,
                bpe_ops=self._bpe_operations,
                bpe_model=self._bpe_model_path,
                src=self._src_lang,
                trg=self._trg_lang
            ),
            "Learning BPE model: %s operations" % self._bpe_operations
        )

    def apply_bpe_model(self):
        '''
        ###BH todo text
        language model built in learn_bpe_model() used to process:
            'train.truecased.SRC'
            'train.truecased.TRG'
            'eval.truecased.SRC' (if existing)
            'eval.truecased.TRG' (if existing)
        '''

        def command(current_corpus, current_lang):
            return '{script} -c {bpe_model}/{src}-{trg}.bpe < {corpus}.{lang} > {corpus}.bpe.{lang}'.format(
            script=SUBWORD_NMT_APPLY,
            bpe_model=self._bpe_model_path,
            src=self._src_lang,
            trg=self._trg_lang,
            corpus=current_corpus,
            lang=current_lang
            )

        commands = [
            command(self._corpus_train_tc, self._src_lang),
            command(self._corpus_train_tc, self._trg_lang),
        ]
        if self._tuning:
            commands.append(command(self._corpus_eval_tc, self._src_lang))
            commands.append(command(self._corpus_eval_tc, self._trg_lang))
        commander.run_parallel(commands, "Applying BPE model")

    def build_bpe_dictionary(self):
        '''
        ###BH todo text
        '''
        # processed using commander (no external processing)

        ###BH implement using constant NEMATUS_BUILD_DICT


    def close(self):
    ###BH ckeck if used, else delete
        del self._processor
