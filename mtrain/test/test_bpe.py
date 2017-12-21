#!/usr/bin/env python3

import os
import time ###BH just testing

from unittest import TestCase
from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup, TestCaseHelper

from mtrain.preprocessing.bpe import BytePairEncoderFile, BytePairEncoderSegment, BytePairDecoderSegment
from mtrain.training import TrainingNematus
from mtrain.constants import *
from mtrain import assertions

class TestBytePairEncoderFile(TestCaseWithCleanup, TestCaseHelper):
    '''
    Tests mainly derived from test_training.py.

    CAUTION: When changing the test parallel corpora, also adapt bpe_ops and test cases as they explicitly match these test corpora.
             Also mind that sentence initial words might not be truecased as they occur rarely or not at all as lowercased words.
    '''

    # test parallel corpus ro-en train and tune:
    ###BH check reference:
    # cf. https://github.com/rsennrich/wmt16-scripts/blob/master/sample/data/newsdev2016.ro
    # cf. https://github.com/rsennrich/wmt16-scripts/blob/master/sample/data/newsdev2016.en
    test_parallel_corpus_train_ro_en = {
        "Avem cel mai mare număr de candidați admiși din istoria universității, aproape 920 de studenți în anul I.":
        "We have the largest number of candidates ever admitted in the university's history, nearly 920 students in the first year."
    }
    test_parallel_corpus_tune_ro_en = {
        "Niciodată nu am depășit un asemenea număr.":
        "We have never reached such a number."
    }

    def _prepare_bpencoder_file(self, random_basedir_name):
        '''
        Prepare byte-pair encoder, method for reuse in tests.
        '''
        # setup path and filenames for sample corpus
        sample_train_ro = os.sep.join([random_basedir_name, 'sample-corpus.ro'])
        sample_train_en = os.sep.join([random_basedir_name, 'sample-corpus.en'])
        sample_tune_ro = os.sep.join([random_basedir_name, 'sample-corpus-tune.ro'])
        sample_tune_en = os.sep.join([random_basedir_name, 'sample-corpus-tune.en'])

        # prepare sample parallel corpora train and tune
        with open(sample_train_ro, 'a') as f_src:
            with open(sample_train_en, 'a') as f_trg:
                for ro_segment, en_segment in self.test_parallel_corpus_train_ro_en.items():
                    f_src.write(ro_segment + '\n')
                    f_trg.write(en_segment + '\n')
        f_src.close()
        f_trg.close()
        with open(sample_tune_ro, 'a') as f_src:
            with open(sample_tune_en, 'a') as f_trg:
                for ro_segment, en_segment in self.test_parallel_corpus_tune_ro_en.items():
                    f_src.write(ro_segment + '\n')
                    f_trg.write(en_segment + '\n')
        f_src.close()
        f_trg.close()

        # preprocess and truecase (using and preprocessing external tuning corpus)
        t = TrainingNematus(random_basedir_name, "ro", "en", TRUECASING, os.sep.join([random_basedir_name, 'sample-corpus-tune']), None)
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, True)
        t.train_truecaser()
        t.truecase()

        # set up byte-pair encoder
        corpus_train_tc=os.sep.join([random_basedir_name, "corpus"]) + os.sep + BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED
        corpus_tune_tc=os.sep.join([random_basedir_name, "corpus"]) + os.sep + BASENAME_TUNING_CORPUS + "." + SUFFIX_TRUECASED
        bpe_model_path = os.sep.join([random_basedir_name, "engine", BPE])
        if not assertions.dir_exists(bpe_model_path):
            os.mkdir(bpe_model_path)
        # number of bpe operations = number of n-grams in bpe model
        # use small value, i.e. a number of possible n-grams that can actually be derived from the the rather small sample corpus
        bpe_ops = 36 # 36 matches the current test cases exactly
        encoder = BytePairEncoderFile(corpus_train_tc, corpus_tune_tc, bpe_model_path, bpe_ops, "ro", "en")

        # save for passing to test method
        self._bpe_model_path = bpe_model_path
        self._encoder = encoder
        self._bpe_ops = bpe_ops

    def _check_content(self, file, search_content):
        '''
        Helper method for checking file content, cf. https://stackoverflow.com/questions/4940032/how-to-search-for-a-string-in-text-files ###BH check reference
        '''
        found = False
        line_number = 0
        with open(file, 'r') as f:
            for line in f:
                line_number += 1
                if search_content in line:
                    found = True
                    break
        f.close()
        return found, line_number

    def test_learn_bpe_model(self):
        '''
        Learn bpe model, check model file creation and its basic content.
        '''
        # create basename individually per test, facilitates monitoring tests and tear down
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        self._prepare_bpencoder_file(random_basedir_name)
        self._encoder.learn_bpe_model()

        # check creation of model file 
        files_created = os.listdir(self._bpe_model_path)
        self.assertTrue(
            "ro-en.bpe" in files_created,
            "BPE model for source and target language must be created"
        )

        # check content of model file
        bpe_model = os.sep.join([self._bpe_model_path, "ro-en.bpe"])
        # number of lines in model = number of n-grams + 1 for header that states file version
        self.assertTrue(
            (self._bpe_ops + 1) == self.count_lines(bpe_model),
            "Number of n-grams in BPE model must correspond to number of bpe operations"
        )
        # check header/file version
        found, line_number = self._check_content(bpe_model, "#version: 0.2")
        self.assertTrue(
            found and line_number == 1,
            "BPE model must have expected file header"
        )
        # check selection of n-gram examples that must be covered
        found, line_number = self._check_content(bpe_model, "t h")
        self.assertTrue(
            found,
            "Example n-gram 't h' must be included in model"
        )
        found, line_number = self._check_content(bpe_model, "th e")
        self.assertTrue(
            found,
            "Example n-gram 'th e' must be included in model"
        )
        found, line_number = self._check_content(bpe_model, "s t")
        self.assertTrue(
            found,
            "Example n-gram 's t' must be included in model"
        )
        found, line_number = self._check_content(bpe_model, "u d")
        self.assertTrue(
            found,
            "Example n-gram 'u d' must be included in model"
        )
        found, line_number = self._check_content(bpe_model, "ud e")
        self.assertTrue(
            found,
            "Example n-gram 'ud e' must be included in model"
        )
        found, line_number = self._check_content(bpe_model, "u n")
        self.assertTrue(
            found,
            "Example n-gram 'u n' must be included in model"
        )
        found, line_number = self._check_content(bpe_model, "un i")
        self.assertTrue(
            found,
            "Example n-gram 'un i' must be included in model"
        )
        found, line_number = self._check_content(bpe_model, "9 2")
        self.assertTrue(
            found,
            "Example n-gram '9 2' must be included in model"
        )
        found, line_number = self._check_content(bpe_model, "92 0")
        self.assertTrue(
            found,
            "Example n-gram '92 0' must be included in model"
        )

    def test_apply_bpe_model(self):
        # create basename individually per test, facilitates monitoring tests and tear down
        random_basedir_name = 'test_cases/bpe_testing' #self.get_random_basename()
        os.mkdir(random_basedir_name)
        self._prepare_bpencoder_file(random_basedir_name)
        self._encoder.learn_bpe_model()
        self._encoder.apply_bpe_model()

        # check creation of byte-pair encoded files
        train_src = BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED + ".bpe.ro"
        train_trg = BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED + ".bpe.en"
        tune_src = BASENAME_TUNING_CORPUS + "." + SUFFIX_TRUECASED + ".bpe.ro"
        tune_trg = BASENAME_TUNING_CORPUS + "." + SUFFIX_TRUECASED + ".bpe.en"
        files_created = os.listdir(os.sep.join([random_basedir_name, "corpus"]))
        self.assertTrue(
            train_src in files_created,
            "Truecased training corpus for source language must be byte-pair encoded"
        )
        self.assertTrue(
            train_trg in files_created,
            "Truecased training corpus for target language must be byte-pair encoded"
        )
        self.assertTrue(
            tune_src in files_created,
            "Truecased tuning corpus for source language must be byte-pair encoded"
        )
        self.assertTrue(
            tune_trg in files_created,
            "Truecased tuning corpus for target language must be byte-pair encoded"
        )

        # check content of byte-pair encoded files
        # exploiting uppercased letters as they are not in bpe model and encoded individually
        ###BH check bpe description
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_src]), "A@@ ")
        self.assertTrue(
            found,
            "'A@@ ' "
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_trg]), "W@@ ")
        self.assertTrue(
            found,
            "'W@@ ' "
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", tune_src]), "N@@ ")
        self.assertTrue(
            found,
            "'N@@ ' "
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", tune_trg]), "W@@ ")
        self.assertTrue(
            found,
            "'W@@ ' "
        )
        # checking n-grams of highest order (here 'universit' occurring in training corpus, results in encoding 'universit@@ ') 
        ###BH check bpe description
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_src]), "universit@@ ")
        self.assertTrue(
            found,
            "'universit@@ ' "
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_trg]), "universit@@ ")
        self.assertTrue(
            found,
            "'universit@@ ' "
        )
        # checking n-grams lower than possible highest order (here encoding 'universi@@ ' must NOT be possible as 'universit@@ ' is encoded)
        ###BH check bpe description
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_src]), "universi@@ ")
        self.assertTrue(
            not found,
            "'universi@@ ' "
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_trg]), "universi@@ ")
        self.assertTrue(
            not found,
            "'universi@@ ' "
        )

    def test_build_bpe_dictionary(self):
        # create basename individually per test, facilitates monitoring tests and tear down
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        self._prepare_bpencoder_file(random_basedir_name)
        self._encoder.learn_bpe_model()
        self._encoder.apply_bpe_model()
        self._encoder.build_bpe_dictionary()

        files_created = os.listdir(os.sep.join([random_basedir_name, "corpus"]))
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
