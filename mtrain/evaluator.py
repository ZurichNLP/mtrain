#!/usr/bin/env python3

"""
Automatic evaluation of a trained engine, based on the translations of an evaluation set.
"""

import os
import logging

from mtrain import constants as C
from mtrain import commander
from mtrain import inspector
from mtrain import utils

from mtrain.translation import TranslationEngineMoses, TranslationEngineNematus

class Evaluator(object):
    """
    Class for automatic evaluation of trained systems.
    """
    def __init__(self, basepath):
        """
        Creates an evaluation directory and loads components.

        @param basepath a directory containing a trained engine
        """
        assert inspector.is_mtrain_engine(basepath)
        self._training_args = utils.load_config_from_basepath(basepath)

        # file paths and tool
        self._basepath = basepath

        # for convenience
        self._src_lang = self._training_args.src_lang
        self._trg_lang = self._training_args.trg_lang

        self._load_engine()

    def _create_eval_directories(self):
        """
        Creates directories to store evaluation results.
        """
        self._base_dir_evaluation = os.path.sep.join([self._basepath, C.PATH_COMPONENT['evaluation']])
        utils.make_dir_if_not_exist(self._base_dir_evaluation)

        self._base_dir_tool = self._base_dir_evaluation + os.sep + self._eval_tool
        utils.make_dir_if_not_exist(self._base_dir_tool)

    def _load_engine(self):
        """
        Loads a translation engine, depending on the training config.

        Note: Nematus translation engines use the training device, but with
        a fixed amount of memory preallocation. Also, beam size is fixed to
        12 for now.
        """

        if self._training_args.backend == C.BACKEND_MOSES:
            self._engine = TranslationEngineMoses(basepath=self._basepath,
                                                  training_config=self._training_args)
        elif self._training_args.backend == C.BACKEND_NEMATUS:
            self._engine = TranslationEngineNematus(basepath=self._basepath,
                                                  training_config=self._training_args,
                                                  device=self._training_args.device_train,
                                                  preallocate=0.2,
                                                  beam_size=12,
                                                  keep_temp_files=False)
        else:
            raise NotImplementedError

    def evaluate(self, eval_tool, eval_corpus_path=None):
        """
        Translates a test set and evaluates the results.

        @param eval_tool an external tool used for evaluation
        @param eval_corpus_path path to the evaluation corpus
        """
        if eval_corpus_path is None:
            self._eval_corpus_path = self._get_path('corpus') + os.sep + C.BASENAME_EVALUATION_CORPUS
        else:
            self._eval_corpus_path = eval_corpus_path

        self._eval_tool = eval_tool
        self._create_eval_directories()

        # file that needs to be translated, source lang
        input_path = self._get_path_eval(self._src_lang)
        # reference in the target language
        reference_path = self._get_path_eval(self._trg_lang)
        # where translation output is stored
        hypothesis_path = self._get_path_hypothesis()
        # where output of evaluation is stored
        output_path = self._get_path_eval_output()

        input_handle = open(input_path, "r")
        hypothesis_handle = open(hypothesis_path, "w")

        self._engine.translate_file(input_handle=input_handle, output_handle=hypothesis_handle)

        input_handle.close()
        hypothesis_handle.close()

        if self._eval_tool == C.MULTEVAL_TOOL:
            self._multeval(output_path, hypothesis_path, reference_path)
        elif self._eval_tool == C.MULTIBLEU_DETOK_TOOL:
            self._multibleu(output_path, hypothesis_path, reference_path)
        else:
            raise NotImplementedError

    def _get_path(self, component):
        '''
        Returns the absolute path to the base directory of the given
        @param component.
        '''
        assert component in C.PATH_COMPONENT, "Unknown component %s" % component
        return os.path.sep.join([self._basepath, C.PATH_COMPONENT[component]])

    def _get_path_eval(self, lang):
        '''
        Returns the path to a side of the reference corpus.
        @param lang language of the source or target side
        '''
        return self._eval_corpus_path + "." + lang

    def _get_path_hypothesis(self):
        """
        Returns the path to the file translated by the engine.
        """
        return self._base_dir_tool + os.sep + "hypothesis." + self._trg_lang

    def _get_path_eval_output(self):
        """
        Returns the path to where the evaluation results should be stored.
        """
        return self._base_dir_tool + os.sep + "hypothesis." + self._eval_tool

    def _multeval(self, output_path, hypothesis_path, reference_path):
        """
        Scores existing translations with MultEval.

        @param output_path path to file where results should be stored
        @param hypothesis_path path to machine translated segments
        @param reference_path path to target side of reference corpus
        """
        # determine whether METEOR scores can be computed
        meteor_argument = '--meteor.language "%s"' % self._trg_lang
        if self._trg_lang not in C.METEOR_LANG_CODES.keys():
            logging.warning("Target language not supported by METEOR library. Evaluation will not include METEOR scores.")
            meteor_argument = '--metrics bleu,ter,length'

        multeval_command = '{script} eval --verbosity 0 --bleu.verbosity 0 --threads "{num_threads}" {meteor_argument} --refs "{reference_path}" --hyps-baseline "{hypothesis}" > "{output_path}"'.format(
            script=C.MULTEVAL,
            num_threads=self._training_args.threads,
            meteor_argument=meteor_argument,
            reference_path=reference_path,
            hypothesis=hypothesis_path,
            output_path=output_path
        )
        commander.run(
            multeval_command,
            "Evaluating with MultEval."
        )

    def _multibleu(self, output_path, hypothesis_path, reference_path):
        """
        Computes BLEU scores with internal tokenization, output identical
        to multi-bleu-detok.perl.
        """
        multibleu_command = '{script} {reference_path} < {hypothesis_path} > {output_path}'.format(
            script=C.MULTIBLEU_DETOK_TOOL,
            reference_path=reference_path,
            hypothesis=hypothesis_path,
            output_path=output_path
        )
        commander.run(
            multibleu_command,
            "Evaluating with multi-bleu-detok.perl."
        )
