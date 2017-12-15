#!/usr/bin/env python3

import os

from unittest import TestCase
from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup, TestCaseHelper

from mtrain.preprocessing.bpe import BytePairEncoderFile, BytePairEncoderSegment, BytePairDecoderSegment
from mtrain.training import TrainingNematus
from mtrain.constants import *

class TestBytePairEncoderFile(TestCaseWithCleanup, TestCaseHelper):

    def test_learn_bpe_model(cls):

        random_basedir_name = cls.get_random_basename()
        os.mkdir(random_basedir_name)

        t = TrainingNematus(random_basedir_name, "ro", "en", TRUECASING, 50, None)
        cls._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )


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

class TestBytePairDecoderSegment(TestCase):
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
            self.assertEqual(decoder.bpdecode_segment(example_segment), decoded_segment)
