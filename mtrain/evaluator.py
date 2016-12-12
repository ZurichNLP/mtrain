#!/usr/bin/env python3
'''
Automatic evaluation of a trained engine, based on the translations of an evaluation set
'''

import os
import logging

from mtrain.constants import *
from mtrain import assertions
from mtrain import commander
from mtrain.translation import TranslationEngine

class Evaluator(object):
    '''
    Translates a set of test sentences and evaluates the translations with automatic
        metrics of translation quality.
    '''
    def __init__(self, basepath, eval_corpus_path, src_lang, trg_lang, casing_strategy,
        masking_strategy, xml_strategy, num_threads, eval_tool="MultEval"):
        '''
        Creates an evaluation directory, translates test sentences and scores them.
        @param basepath a directory containing a trained engine
        @param eval_corpus_path path to the evaluation corpus
        @param src_lang the language code of the source language, e.g., `en`
        @param trg_lang the language code of the target language, e.g., `fr`
        @param casing_strategy how case should be handled in preprocessing
        @param masking_strategy whether and how mask tokens should be
            introduced into segments
        @param xml_strategy whether fragments of markup in the data should be
            passed through, removed or masked
        @param num_threads max number of threads to be used by all processed
            invoked during evaluation
        @param eval_tool an external tool used for evaluation
        '''
        # file paths and languages
        self._basepath = basepath
        self._eval_corpus_path = eval_corpus_path
        self._src_lang = src_lang
        self._trg_lang = trg_lang
        # strategies
        self._masking_strategy = masking_strategy
        self._casing_strategy = casing_strategy
        self._xml_strategy = xml_strategy
        # evaluation options
        self._num_threads = num_threads
        self._eval_tool = eval_tool

        # create target directories
        self._base_dir_evaluation = os.path.sep.join([self._basepath, PATH_COMPONENT['evaluation']])
        if not assertions.dir_exists(self._base_dir_evaluation):
            os.mkdir(self._base_dir_evaluation)
        if self._eval_tool == 'MultEval':
            self._base_dir_multeval = self._base_dir_evaluation + os.sep + EVALUATION_TOOLS['MultEval']
            if not assertions.dir_exists(self._base_dir_multeval):
                os.mkdir(self._base_dir_multeval)
        else:
            # currently, tools other than MultEval are not supported
            raise NotImplementedError

    def _translate_eval_corpus(self):
        '''
        Translates sentences from an evaluation corpus.
        '''
        logging.info("Translating evaluation corpus")
        self._engine = TranslationEngine(self._basepath, self._src_lang, self._trg_lang)

        # remove all engine processes
        self._engine.close()

    def _get_path_eval(self, lang):
        '''
        Returns the path to a side of the reference corpus.
        @param lang language of the source or target side
        '''
        path_corpus = os.path.sep.join([self._basepath, PATH_COMPONENT['corpus']])
        return path_corpus + os.sep + '.'.join(BASENAME_EVALUATION_CORPUS, lang)

    def _get_path_hypothesis(self):
        '''
        Returns the path to the file translated by the engine.
        '''
        return self._base_dir_multeval + os.sep + "hypothesis" + "." + ".".join(self._eval_options) + "." + self._trg_lang 

    def _get_path_eval_output(self):
        '''
        Returns the path to where the evaluation results should be stored.
        '''
        return self._base_dir_multeval + os.sep + "hypothesis" + "." + ".".join(self._eval_options) + "." + EVALUATION_TOOLS['MultEval']

    def _score(self):
        '''
        Scores existing translations with an external tool.
        '''
        # determine paths to relevant files
        output_path = self._get_path_eval_output()
        hypothesis_path = self._get_path_hypothesis()
        corpus_evaluation_trg = self._get_path_eval(self._trg_lang)

        if self._eval_tool == EVALUATION_TOOLS['MultEval']:
            self._multeval(output_path, hypothesis_path, corpus_evaluation_trg)
        else:
            raise NotImplementedError

    def _multeval(self, output_path, hypothesis_path, corpus_evaluation_trg):
        '''
        Scores existing translations with MultEval.
        @param output_path path to file where results should be stored
        @param hypothesis_path path to machine translated segments
        @param corpus_evaluation_trg path to target side of reference corpus
        '''
        # determine whether METEOR scores can be computed
        meteor_argument = '--meteor.language "%s"' % self._trg_lang
        if self._trg_lang not in METEOR_LANG_CODES.keys():
            logging.warning("Target language not supported by METEOR library. Evaluation will not include METEOR scores.")
            meteor_argument = '--metrics bleu,ter,length'

        multeval_command = '{script} eval --verbosity 0 --bleu.verbosity 0 --threads "{num_threads}" {meteor_argument} --refs "{corpus_evaluation_trg}" --hyps-baseline "{hypothesis}" > "{output_file}"'.format(
            script=MULTEVAL,
            num_threads=self._num_threads,
            meteor_argument=meteor_argument,
            corpus_evaluation_trg=corpus_evaluation_trg,
            hypothesis=hypothesis_path,
            output_file=output_path
        )
        commander.run(multeval_command, "Evaluating engine with MultEval")

    # only exposed method
    def evaluate(self, lowercase, detokenize, strip_markup):
        '''
        Translates test segment and scores them with an external tool, saves
            results to a file.
        @param lowercase whether to lowercase all segments before evaluation
        @param detokenize whether to detokenize all segments before evaluation
        @param strip_markup whether all markup should be removed before evaluation
        '''
        # reset evaluation options for each invocation of the method
        self._lowercase_eval = lowercase
        self._detokenize_eval = detokenize
        self._strip_markup_eval = strip_markup

        # conveniently list evaluation options
        self._eval_options = []
        if self._lowercase_eval:
            self._eval_options.append('lowercased')
        else:
            self._eval_options.append('cased')
        if self._detokenize_eval:
            self._eval_options.append('detokenized')
        else:
            self._eval_options.append('tokenized')
        if self._strip_markup_eval:
            self._eval_options.append('no_markup')
        else:
            self._eval_options.append('with_markup')

        self._translate_eval_corpus()
        self._score()
        
