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

    Input for parallel corpora in Romanian and English: ###BH check reference
    cf. https://github.com/rsennrich/wmt16-scripts/blob/master/sample/data/newsdev2016.ro
    cf. https://github.com/rsennrich/wmt16-scripts/blob/master/sample/data/newsdev2016.en

    CAUTION: When changing the test parallel corpora, also adapt bpe_ops and test cases as they explicitly match these test corpora.
             Also mind that sentence initial words might not be truecased as they occur rarely or not at all as lowercased words.
    '''

    # test corpus Romanian-English training set
    test_parallel_corpus_train_ro_en = {
        "Avem cel mai mare număr de candidați admiși din istoria universității, aproape 920 de studenți în anul I.":
        "We have the largest number of candidates ever admitted in the university's history, nearly 920 students in the first year."
    }
    # test corpus Romanian-English tuning set
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

        # save arguments/components for using in test method
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
        Learn BPE model according to script learn_bpe.py. ###BH check reference
        Check model file creation and its basic content.
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
            "BPE model for source and target side must be created"
        )

        # check content of model file
        bpe_model = os.sep.join([self._bpe_model_path, "ro-en.bpe"])
        # number of lines in model = number of n-grams + 1 for header that states file version
        self.assertTrue(
            (self._bpe_ops + 1) == self.count_lines(bpe_model),
            "Number of n-grams in BPE model must correspond to number of bpe operations"
        )
        # check header/file version: content and position of string checked
        found, line_number = self._check_content(bpe_model, "#version: 0.2")
        self.assertTrue(
            found and line_number == 1,
            "BPE model must have expected file header"
        )
        # check selection of n-grams that must be covered in model
        ###BH check bpe description
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
        '''
        Apply BPE model according to script apply_bpe.py. ###BH check reference
        Check creation of byte-pair encoded files and their content.
        '''
        # create basename individually per test, facilitates monitoring tests and tear down
        random_basedir_name = self.get_random_basename()
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
            "Truecased training corpus for source side must be byte-pair encoded to new file"
        )
        self.assertTrue(
            train_trg in files_created,
            "Truecased training corpus for target side must be byte-pair encoded to new file"
        )
        self.assertTrue(
            tune_src in files_created,
            "Truecased tuning corpus for source side must be byte-pair encoded to new file"
        )
        self.assertTrue(
            tune_trg in files_created,
            "Truecased tuning corpus for target side must be byte-pair encoded to new file"
        )

        # check content of byte-pair encoded files

        # exploiting presence of sentence initial uppercased letters in truecased (tc) corpora:
        #   as uppercased letters are not included in n-grams of this particular bpe model they are encoded individually
        ###BH check bpe description
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_src]), "A@@ ")
        self.assertTrue(
            found,
            "Example 'A@@ ' must be encoded in tc training corpus of source side"
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_trg]), "W@@ ")
        self.assertTrue(
            found,
            "Example 'W@@ ' must be encoded in tc training corpus of target side"
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", tune_src]), "N@@ ")
        self.assertTrue(
            found,
            "Example 'N@@ ' must be encoded in tc tune corpus of source side"
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", tune_trg]), "W@@ ")
        self.assertTrue(
            found,
            "Example 'W@@ ' must be encoded in tc tune corpus of target side"
        )
        # checking n-gram example of highest order in model: 'universit'.
        #   occurring in tc training corpus of both source (in 'universitatii') and target side (in 'university'),
        #   resulting in encoding 'universit@@ '
        ###BH check bpe description
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_src]), "universit@@ ")
        self.assertTrue(
            found,
            "Example 'universit@@ ' must be encoded in tc training corpus of source side"
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_trg]), "universit@@ ")
        self.assertTrue(
            found,
            "Example 'universit@@ ' must be encoded in tc training corpus of target side"
        )
        # due to encoding 'universit@@ ', 'universitatii' and 'university' must NOT be encoded as 'universi@@ '
        #   as this would be lower n-gram order than possible according to model
        ###BH check bpe description
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_src]), "universi@@ ")
        self.assertTrue(
            not found,
            "Example 'universi@@ ' must NOT be encoded in tc training corpus of source side (n-gram order too low)"
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_trg]), "universi@@ ")
        self.assertTrue(
            not found,
            "Example 'universi@@ ' must NOT be encoded in tc training corpus of target side (n-gram order too low)"
        )
        # due to encoding 'universit@@ ', 'universitatii' and 'university' must NOT be encoded as 'universita@@ ' or 'university@@ '
        #   as this would be higher n-gram order than possible according to model
        ###BH check bpe description
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_src]), "universita@@ ")
        self.assertTrue(
            not found,
            "Example 'universita@@ ' must NOT be encoded in tc training corpus of source side (n-gram order too high)"
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_trg]), "university@@ ")
        self.assertTrue(
            not found,
            "Example 'university@@ ' must NOT be encoded in tc training corpus of target side (n-gram order too high)"
        )

    def test_build_bpe_dictionary(self):
        '''
        Build BPE dictionary according to script build_dictionary.py. ###BH check reference
        Check creation of dictionary and its basic content.
        '''
        # create basename individually per test, facilitates monitoring tests and tear down
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        self._prepare_bpencoder_file(random_basedir_name)
        self._encoder.learn_bpe_model()
        self._encoder.apply_bpe_model()
        self._encoder.build_bpe_dictionary()

        # check creation of network dictionaries
        train_src = BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED + ".bpe.ro.json"
        train_trg = BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED + ".bpe.en.json"
        files_created = os.listdir(os.sep.join([random_basedir_name, "corpus"]))

        self.assertTrue(
            train_src in files_created,
            "Network dictionary for tc and encoded training corpus of source side must be created"
        )
        self.assertTrue(
            train_trg in files_created,
            "Network dictionary for tc and encoded training corpus of target side must be created"
        )

        # check content of network dictionaries

        # check technical dictionary entries 
        ###BH check bpe description
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_src]), '"eos": 0')
        self.assertTrue(
            found,
            'Example "eos": 0 must be covered in network dictionary of source side'
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_trg]), '"eos": 0')
        self.assertTrue(
            found,
            'Example "eos": 0 must be covered in network dictionary of target side'
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_src]), '"UNK": 1')
        self.assertTrue(
            found,
            'Example "UNK": 1 must be covered in network dictionary of source side'
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_trg]), '"UNK": 1')
        self.assertTrue(
            found,
            'Example "UNK": 1 must be covered in network dictionary of target side'
        )

        # adapting tests from test_apply_bpe_model() and applying to dictionaries:
        #   '"universit@@"' must be covered in dictionaries while '"universi@@"', '"universita@@"'
        #   and '"university@@"' must NOT be covered
        ###BH check bpe description
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_src]), '"universit@@": ')
        self.assertTrue(
            found,
            'Example "universit@@": nn must be covered in network dictionary of source side'
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_trg]), '"universit@@": ')
        self.assertTrue(
            found,
            'Example "universit@@": nn must be covered in network dictionary of target side'
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_src]), '"universi@@": ')
        self.assertTrue(
            not found,
            'Example "universi@@": nn must NOT be covered in network dictionary of source side'
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_trg]), '"universi@@": ')
        self.assertTrue(
            not found,
            'Example "universi@@": nn must NOT be covered in network dictionary of target side'
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_src]), '"universita@@": ')
        self.assertTrue(
            not found,
            'Example "universita@@": nn must NOT be covered in network dictionary of source side'
        )
        found, line_number = self._check_content(os.sep.join([random_basedir_name, "corpus", train_trg]), '"university@@": ')
        self.assertTrue(
            not found,
            'Example "university@@": nn must NOT be covered in network dictionary of target side'
        )

class TestBytePairEncoderSegment(TestCase):
    # English test cases: normalized, tokenized and truecased segments.    
    test_cases_en = {
        "we have the largest number of candidates ever admitted to the university .":
        "we have the largest number of candidates ever admitted to the university .",
        '': '',
        '': '',
        '': ''
    }

    def test_bpencode_segment(self):
        '''
        This test is executed with a `mtrain` pre-trained byte-pair encoding models from an external path.
        The said models provided in the test case path and loaded with the ExternalProcessor would
        not be readable by apply_bpe.py. Subsequent encoding would cause the models to lock the
        path, causing tearDownClass() to fail several tests where tear down is applied.
        '''

        ''' ###BH remove after testing
        >>> from mtrain.preprocessing.bpe import BytePairEncoderSegment
        >>> en = BytePairEncoderSegment('~/mtrain/mtrain/test/data/en-ro.bpe')
        >>> en.bpencode_segment("")

        >>> from mtrain.preprocessing.bpe import BytePairEncoderSegment
        >>> ro = BytePairEncoderSegment('~/mtrain/mtrain/test/data/ro-en.bpe')
        >>> ro.bpencode_segment("")
        '''

        # get current working directory (e.g. ~/mtrain) and external bpe model therein
        cwd = os.getcwd()
        external_bpe_model_en_ro = os.sep.join([cwd, '/mtrain/test/data/en-ro.bpe'])
        external_bpe_model_ro_en = os.sep.join([cwd, '/mtrain/test/data/ro-en.bpe'])

        # encoding segments of English test cases
        segment_encoder = BytePairEncoderSegment(external_bpe_model_en_ro)
        for example_segment, encoded_segment in self.test_cases_en.items():
            self.assertEqual(segment_encoder.bpencode_segment(example_segment), encoded_segment)
        segment_encoder.close()

        #time.sleep(600) ###BH test

class TestBytePairDecoderSegment(TestCase):
    # test cases devised using https://translate.google.com/m/translate, ###BH check reference
    #   correspond to normalized, tokenized, truecased and encoded segments
    test_cases = {
        "this is an ex@@ ample sent@@ ence .": "this is an example sentence .",
        "esta es una oracion de ej@@ emplo .": "esta es una oracion de ejemplo .",
        "это приме@@ рное пр@@ едложение .": "это примерное предложение .",
        "dies ist ein Bei@@ spiel@@ satz .": "dies ist ein Beispielsatz .",
        "acesta este un ex@@ emplu de propo@@ zitie .": "acesta este un exemplu de propozitie ."
    }

    def test_bpdecode_segment(self):
        '''
        Byte-pair decoding must replace strings "@@ " with empty strings "",
        cf. https://github.com/rsennrich/wmt16-scripts/blob/master/sample/postprocess-test.sh. ###BH check reference
        '''
        segment_decoder = BytePairDecoderSegment()
        for example_segment, decoded_segment in self.test_cases.items():
            self.assertEqual(segment_decoder.bpdecode_segment(example_segment), decoded_segment)
