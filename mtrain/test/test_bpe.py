#!/usr/bin/env python3

import logging
import random
import shutil
import sys
import os
#import time ###BH just testing

from unittest import TestCase
from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup, TestCaseHelper

from mtrain.preprocessing.bpe import BytePairEncoderFile, BytePairEncoderSegment, BytePairDecoderSegment
from mtrain.training import TrainingNematus
#from mtrain.translation import TranslationEngineNematus ###BH just testing
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

class TestBytePairEncoderSegment(TestCaseWithCleanup):
    # Test parallel corpus RO-EN ###BH check reference:
    # cf. https://github.com/rsennrich/wmt16-scripts/blob/master/sample/data/newsdev2016.ro
    # cf. https://github.com/rsennrich/wmt16-scripts/blob/master/sample/data/newsdev2016.en
    test_parallel_corpus_ro_en = {
        "Avem cel mai mare număr de candidați admiși din istoria universității, aproape 920 de studenți în anul I.":
        "We have the largest number of candidates ever admitted in the university's history, nearly 920 students in the first year.",
        "Niciodată nu am depășit un asemenea număr.":
        "We have never reached such a number.",
        "Acesta a mai precizat că universitatea a înmatriculat și un număr considerabil de studenți la taxă, ocupând 20 procent din locurile respective, o cifră semnificativ mai mare decât în alți ani.":
        "He also said that the university has registered a considerable number of paying students, occupying 20 percent of its seats, a figure significantly higher than in other years.",
        "Deși s-au publicat și rezultatele la liniile de studiu în limba română, admiterea încă mai continuă la programul de studiu în limba engleză de la Facultatea de Medicină Veterinară, care a scos la concurs 50 de locuri, unde taxa este de 5.000 de euro.":
        "While the results of the Romanian language study programs have been published, admissions still continue for the English program at the Faculty of Veterinary Medicine, who put up 50 seats and for which the fee is 5,000 Euro.",
        "Iar acum se așteaptă răspuns de la minister cu privire la confirmarea dosarelor studenților și a vizelor de studiu necesare.":
        "Officials are now waiting for a response from the ministry regarding the confirmation of students' files and study visas required."
    }

#        "Oscàr .": "O@@ sc@@ à@@ r ."

    # @staticmethod
    # def get_random_sentence():
    #     words = ["103", "físh", "HUM", "84#ça", "banana", "Oscar", "—"]
    #     return " ".join([random.choice(words) for _ in range(0, random.randrange(1,len(words)))])

    # @classmethod
    # def _create_random_parallel_corpus_files(self, path, filename_source="test-corpus.src", filename_target="test_corpus.trg", num_bisegments=200):
    #     for filename in [filename_source, filename_target]:
    #         if path:
    #             filename = os.sep.join([path, filename])
    #         with open(filename, 'w') as f:
    #             for i in range(0, num_bisegments):
    #                 f.write("line %s: %s\n" % (i, self.get_random_sentence()))

    # def _get_random_basename(self):
    #     return str(self._basedir_test_cases + os.sep + str(random.randint(0, 9999999)))

    def _prepare_bpencoder_segment(self):
        # self._random_basedir_name = self._get_random_basename()
        # os.mkdir(self._random_basedir_name)

        # t = TrainingNematus(self._random_basedir_name, "ro", "en", TRUECASING, 50, None)
        # self._create_random_parallel_corpus_files(
        #     path=self._random_basedir_name,
        #     filename_source="sample-corpus.ro",
        #     filename_target="sample-corpus.en",
        #     num_bisegments=200
        # )
        # t.preprocess(os.sep.join([self._random_basedir_name, "sample-corpus"]), 1, 80, True)
        # t.train_truecaser()
        # t.truecase()

        # corpus_train_tc=os.sep.join([self._random_basedir_name, "corpus"]) + os.sep + BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED
        # corpus_tune_tc=os.sep.join([self._random_basedir_name, "corpus"]) + os.sep + BASENAME_TUNING_CORPUS + "." + SUFFIX_TRUECASED
        # self._bpe_model_path = os.sep.join([self._random_basedir_name, "engine", BPE])
        # if not assertions.dir_exists(self._bpe_model_path):
        #     os.mkdir(self._bpe_model_path)
        # self._encoder = BytePairEncoderFile(corpus_train_tc, corpus_tune_tc, self._bpe_model_path, 89500, "ro", "en")

        # setup paths and filenames for corpus
        self._random_basedir_name = 'test_cases/test_bpe_segment'
        os.mkdir(self._random_basedir_name)
        corpus_file_ro = self._random_basedir_name + os.sep + 'sample-corpus.ro'
        corpus_file_en = self._random_basedir_name + os.sep + 'sample-corpus.en'

        # prepare sample parallel corpus
        with open(corpus_file_ro, 'a') as f_src:
            with open(corpus_file_en, 'a') as f_trg:
                for ro_segment, en_segment in self.test_parallel_corpus_ro_en.items():
                    f_src.write(ro_segment + '\n')
                    f_trg.write(en_segment + '\n')
        f_src.close()
        f_trg.close()

        # preprocess and truecase sample parallel corpus
        t = TrainingNematus(self._random_basedir_name, "ro", "en", TRUECASING, 2, None)
        t.preprocess(os.sep.join([self._random_basedir_name, "sample-corpus"]), 1, 80, True)
        t.train_truecaser()
        t.truecase()

        # prepare byte-pair encoder
        corpus_path = os.sep.join([self._random_basedir_name, 'corpus'])
        corpus_train_tc=os.sep.join([corpus_path, BASENAME_TRAINING_CORPUS]) + "." + SUFFIX_TRUECASED
        corpus_tune_tc=os.sep.join([corpus_path, BASENAME_TUNING_CORPUS]) + "." + SUFFIX_TRUECASED
        self._bpe_model_path = os.sep.join([self._random_basedir_name, "engine", BPE])
        if not assertions.dir_exists(self._bpe_model_path):
            os.mkdir(self._bpe_model_path)
        self._encoder = BytePairEncoderFile(corpus_train_tc, corpus_tune_tc, self._bpe_model_path, 100, "ro", "en")

    def test_bpencode_segment(self):
        #self._prepare_bpencoder_segment()
        #self._encoder.learn_bpe_model()
        
        bpe_model = '/mnt/storage/walle/users/horat/facharbeit/mtrain/mtrain/test/data/ro-en.bpe'

        segment_encoder = BytePairEncoderSegment(bpe_model)








        segment_encoder.close()
        #time.sleep(60) ###BH test
        

        '''
        # ExtProc Locks path removal, causing tests to fail: OSError: [Errno 16] Device or resource busy: '.nfs00000000066000160001d833'

        # 1) Manually creating paths did not help because problems below

        segment_encoder = BytePairEncoderSegment(model)

        for example_segment, encoded_segment in self.test_cases.items():
            self.assertEqual(segment_encoder.bpencode_segment(example_segment), encoded_segment)

        segment_encoder.close()
        '''

        '''
        # 2) Instantiating BytePairEncoderSegment shows that model cannot be accessed by bpe_apply.py

        segment_encoder = BytePairEncoderSegment(model)
        '''

        '''
        # 3) Making model readable, but causes error when reading model: /bin/sh: 1: Syntax error: end of file unexpected
        #    Test fails because model called as: -c <_io.TextIOWrapper name='test_cases/4547623/engine/bpe/ro-en.bpe' mode='r' encoding='UTF-8'>

        with open(model, 'r') as f:
            segment_encoder = BytePairEncoderSegment(f)

            for example_segment, encoded_segment in self.test_cases.items():
                self.assertEqual(segment_encoder.bpencode_segment(example_segment), encoded_segment)

            segment_encoder.close()
        '''

        '''
        # 4) Tests successful but encoder does not do anything, sys.stderr.write is ''
        with open(model, 'r') as f:
            segment_encoder = BytePairEncoderSegment(f)
        f.close()

        segment = segment_encoder.bpencode_segment("Oscàr .")
        sys.stderr.write("'" + segment + "'")
        '''

        '''
        # 5) No Syntax error end of file and encoding OK 'O@@ sc@@ à@@ r .' But OSError: [Errno 16] Device or resource busy: '.nfs00000000066000160001d833' crashes all tests again
        open(model, 'r')

        segment_encoder = BytePairEncoderSegment(model)
        segment = segment_encoder.bpencode_segment("Oscàr .")
        sys.stderr.write("'" + segment + "'")

        segment_encoder.close()
        '''

        '''
        # 6) Among other tests, give processes 10 seconds time to finish before rmtree. However, command makes program wait BEFORE encoding for no apparent reason
        open(model, 'r')
        #open_model.close()
        #for line in open_model:
            #sys.stderr.write(line)
            #f.read()
        segment_encoder = BytePairEncoderSegment(model)
        segment = segment_encoder.bpencode_segment("Oscàr .")
        sys.stderr.write("'" + segment + "'")
        segment_encoder.close()
        time.sleep(3)
        '''

        '''
        # 7) Test using Translation Engine does not work as truecasing model not found although trained in _prepare_bpencoder_segment()
        engine = TranslationEngineNematus(self._random_basedir_name, 'ro', 'en')
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

    def test_bpdecode_segment(self):
        segment_decoder = BytePairDecoderSegment()
        for example_segment, decoded_segment in self.test_cases.items():
            self.assertEqual(segment_decoder.bpdecode_segment(example_segment), decoded_segment)
