#!/usr/bin/env python3

import logging
import random
import shutil
import sys
import os
import time ###BH just testing

from unittest import TestCase
from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup

from mtrain.preprocessing.truecaser import Truecaser, Detruecaser
from mtrain.preprocessing.tokenizer import Tokenizer, Detokenizer
from mtrain.constants import *
from mtrain import commander

class TestTruecaser(TestCaseWithCleanup):
    # english test cases, must be normalized and tokenized for training the truecaser
    test_cases = {
        'This is a sample sentence that was said by Obama, allegedly.': 'this is a sample sentence that was said by Obama, allegedly.',
        'As this sentence uses lowercased words that appeared as uppercased before, Obama said these are actually lowercased.': 'as this sentence uses lowercased words that appeared as uppercased before, Obama said these are actually lowercased.',
        'The only words that are always uppercased must therefore be a name or maybe a concept, as this text is in English, Obama said.': 'the only words that are always uppercased must therefore be a name or maybe a concept, as this text is in English, Obama said.',
        'In conclusion, the trucaser must lowercase sentence initial word unless it is a name, concept or the like.': 'in conclusion, the trucaser must lowercase sentence initial word unless it is a name, concept or the like.'
    }

    #def _get_random_basename(self):
    #    return str(self._basedir_test_cases + os.sep + str(random.randint(0, 9999999)))

    def _prepare_truecaser(self):
        '''
        Train truecaser model according to example in training.py.
        '''
        # setup paths and filenames for corpus and tc model
        self._random_basedir_name = 'test_cases/test_tc' ###BH test self._get_random_basename()
        os.mkdir(self._random_basedir_name)
        corpus_path = os.sep.join([self._random_basedir_name, 'corpus'])
        os.mkdir(corpus_path)
        corpus_file = corpus_path + os.sep + 'example.en'
        engine_path = os.sep.join([self._random_basedir_name, 'engine'])
        os.mkdir(engine_path)
        model_path = os.sep.join([engine_path, TRUECASING])
        os.mkdir(model_path)
        model_file = model_path + os.sep + 'model.en'

        # prepare input corpus from test cases
        with open(corpus_file, 'a') as f:
            for example_segment, truecased_segment in self.test_cases.items():
                f.write(example_segment + '\n')

        # train truecaser
        def command():
            return '{script} --model {model} --corpus {corpus}'.format(
                script=MOSES_TRAIN_TRUECASER,
                model=model_file,
                corpus=corpus_file
            )
        commands = [command()]
        commander.run(commands)

        # cleanup: save model full path for passing to test method; close file handle
        self._model_file = model_file
        f.close()

    def test_truecase_tokens(self):
        '''
        Using tokenizer and detokenizer for better readability of test cases and increase code coverage of tokenizer.py.
        '''
        self._prepare_truecaser()
        # load English tokenizer (tokenized segments precondition for fully testing truecaser)
        tokenizer = Tokenizer('en')
        # load truecaser using truecasing model
        truecaser = Truecaser(self._model_file)
        # load English detokenizer 
        detokenizer = Detokenizer('en')

        # using truecase_tokens() instead of truecase() to increase code coverage
        for example_segment, truecased_segment in self.test_cases.items():
            self.assertEqual(detokenizer.detokenize(truecaser.truecase_tokens(tokenizer.tokenize(example_segment))), truecased_segment)

        # cleanup
        tokenizer.close()
        truecaser.close()
        detokenizer.close()

class TestDetruecaser(TestCase):
    # english test cases, reversed examples from TestTruecaser() class
    test_cases = {
        'this is a sample sentence that was said by Obama, allegedly.': 'This is a sample sentence that was said by Obama, allegedly.',
        'as this sentence uses lowercased words that appeared as uppercased before, Obama said these are actually lowercased.': 'As this sentence uses lowercased words that appeared as uppercased before, Obama said these are actually lowercased.',
        'the only words that are always uppercased must therefore be a name or maybe a concept, as this text is in English, Obama said.': 'The only words that are always uppercased must therefore be a name or maybe a concept, as this text is in English, Obama said.',
        'in conclusion, the trucaser must lowercase sentence initial word unless it is a name, concept or the like.': 'In conclusion, the trucaser must lowercase sentence initial word unless it is a name, concept or the like.'
    }

    def test_detruecase(self):
        # detruecasing (uppercase first word of segment)
        detruecaser = Detruecaser()
        for example_segment, detruecased_segment in self.test_cases.items():
            self.assertEqual(detruecaser.detruecase(example_segment), detruecased_segment)
        detruecaser.close()
