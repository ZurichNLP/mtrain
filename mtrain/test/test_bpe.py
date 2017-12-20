#!/usr/bin/env python3

import os

from unittest import TestCase
from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup, TestCaseHelper

from mtrain.preprocessing.bpe import BytePairEncoderFile, BytePairEncoderSegment, BytePairDecoderSegment
from mtrain.training import TrainingNematus
from mtrain.constants import *
from mtrain import assertions

class TestBytePairEncoderFile(TestCaseWithCleanup, TestCaseHelper):

    def _prepare_bpencoder_file(self):
        '''
        Learn bpe model, method for reuse in tests.
        '''
        self._random_basedir_name = self.get_random_basename()
        os.mkdir(self._random_basedir_name)

        t = TrainingNematus(self._random_basedir_name, "ro", "en", TRUECASING, 50, None)
        self._create_random_parallel_corpus_files(
            path=self._random_basedir_name,
            filename_source="sample-corpus.ro",
            filename_target="sample-corpus.en",
            num_bisegments=200
        )
        t.preprocess(os.sep.join([self._random_basedir_name, "sample-corpus"]), 1, 80, True)
        t.train_truecaser()
        t.truecase()

        corpus_train_tc=os.sep.join([self._random_basedir_name, "corpus"]) + os.sep + BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED
        corpus_tune_tc=os.sep.join([self._random_basedir_name, "corpus"]) + os.sep + BASENAME_TUNING_CORPUS + "." + SUFFIX_TRUECASED
        self._bpe_model_path = os.sep.join([self._random_basedir_name, "engine", BPE])
        if not assertions.dir_exists(self._bpe_model_path):
            os.mkdir(self._bpe_model_path)
        self._encoder = BytePairEncoderFile(corpus_train_tc, corpus_tune_tc, self._bpe_model_path, 89500, "ro", "en")

    def test_learn_bpe_model(self):
        self._prepare_bpencoder_file()
        self._encoder.learn_bpe_model()

        files_created = os.listdir(self._bpe_model_path)
        self.assertTrue(
            "ro-en.bpe" in files_created,
            "BPE model for source and target language must be created"
        )

        ###BH todo content check

    def test_apply_bpe_model(self):
        self._prepare_bpencoder_file()
        self._encoder.learn_bpe_model()
        self._encoder.apply_bpe_model()

        files_created = os.listdir(os.sep.join([self._random_basedir_name, "corpus"]))
        self.assertTrue(
            BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED + ".bpe.ro" in files_created,
            "Truecased training corpus for source language must be byte-pair encoded"
        )
        self.assertTrue(
            BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED + ".bpe.en" in files_created,
            "Truecased training corpus for target language must be byte-pair encoded"
        )
        self.assertTrue(
            BASENAME_TUNING_CORPUS + "." + SUFFIX_TRUECASED + ".bpe.ro" in files_created,
            "Truecased tuning corpus for source language must be byte-pair encoded"
        )
        self.assertTrue(
            BASENAME_TUNING_CORPUS + "." + SUFFIX_TRUECASED + ".bpe.en" in files_created,
            "Truecased tuning corpus for target language must be byte-pair encoded"
        )

        ###BH todo content check

    def test_build_bpe_dictionary(self):
        self._prepare_bpencoder_file()
        self._encoder.learn_bpe_model()
        self._encoder.apply_bpe_model()
        self._encoder.build_bpe_dictionary()

        files_created = os.listdir(os.sep.join([self._random_basedir_name, "corpus"]))
        self.assertTrue(
            BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED + ".bpe.ro.json" in files_created,
            "Truecased training corpus for source language must be byte-pair encoded"
        )
        self.assertTrue(
            BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED + ".bpe.en.json" in files_created,
            "Truecased training corpus for target language must be byte-pair encoded"
        )

        ###BH todo content check

class TestBytePairEncoderSegment(TestCaseWithCleanup, TestCaseHelper):

    test_cases = {
        'norm, tok, tc !!!': 'encoded!!!',
        '': '',
        '': '',
        '': '',
        '': ''
    }

    def test_bpencode_segment(self):
        '''
        This test is executed with a `mtrain` pre-trained byte-pair encoding model from an external path.
        The said model provided in the test case path and loaded with the ExternalProcessor would
        not be readable by apply_bpe.py. Subsequent encoding would cause the model to lock the
        path, causing tearDownClass() to fail several tests.
        '''

        ''' ###BH remove after testing
        >>> from mtrain.preprocessing.bpe import BytePairEncoderSegment
        >>> e = BytePairEncoderSegment('~/mtrain/mtrain/test/data/ro-en.bpe')
        >> e.bpencode_segment("this is an example sentence")
        '''

        # get current working directory (e.g. ~/mtrain) and external bpe model
        ###BH add reference # cf. https://www.tutorialspoint.com/python/os_getcwd.htm
        cwd = os.getcwd()
        external_bpe_model = os.sep.join([cwd, '/mtrain/test/data/en-ro.bpe'])

        segment_encoder = BytePairEncoderSegment(external_bpe_model)

        segment_encoder.bpencode_segment("aösdkfjaösdkjföasdkjföaksdjf")

        segment_encoder.close()
        #time.sleep(60) ###BH test

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

    def test_bpdecode_segment(self):
        segment_decoder = BytePairDecoderSegment()
        for example_segment, decoded_segment in self.test_cases.items():
            self.assertEqual(segment_decoder.bpdecode_segment(example_segment), decoded_segment)
