#!/usr/bin/env python3

import os
import random

from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup

from mtrain.preprocessing.bpe import BytePairDecoderSegment
from mtrain.training import TrainingNematus
from mtrain.constants import *


class TestBytePairEncoderFile(TestCaseWithCleanup):
    '''
    '''
    def get_random_basename(self):
        return str(self._basedir_test_cases + os.sep + str(random.randint(0, 9999999)))

    def test_learn_bpe_model(self):
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)

        '''
        t = TrainingNematus(random_basedir_name, "en", "fr", TRUECASING, 50, None)
        t.preprocess()
        t.train_truecaser()

        # _create_random_parallel_corpus_files !!!!!


        #t.truecase()
        #e = asdf(1000)
        # e.learn_bpe_model()
        '''


'''
    def test_apply_bpe_model(?):
        pass
    def test_build_bpe_dictionary(?):
        pass

class TestBytePairEncoderSegment(?):
    def test_encode(?):
        pass
'''

class TestBytePairDecoderSegment(TestCaseWithCleanup):
    '''
    Examples correspond to normalized, tokenized, truecased, encoded and translated segments.
    Decoding must replace strings "@@ " with empty string "".
    '''
    test_cases = {
        "this is an ex@@ ample sent@@ ence .": "this is an example sentence .",
        "esta es una oracion de ej@@ emplo .": "esta es una oracion de ejemplo .",
        "это приме@@ рное пр@@ едложение .": "это примерное предложение .",
        "dies ist ein Bei@@ spiel@@ satz .": "dies ist ein Beispielsatz .",
        "acesta este un ex@@ emplu de propo@@ zitie .": "acesta este un exemplu de propozitie ."
    }

    def test_decode(self):
        decoder = BytePairDecoderSegment()
        for example_segment, decoded_segment in self.test_cases.items():
            self.assertEqual(decoder.decode(example_segment), decoded_segment)
