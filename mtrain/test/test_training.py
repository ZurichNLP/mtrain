#!/usr/bin/env python3

import logging
import random
import shutil
import sys
import os

from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup

from mtrain.training import Training
from mtrain.constants import *
from mtrain import assertions

class TestTraining(TestCaseWithCleanup):
    @staticmethod
    def count_lines(filename):
        with open(filename) as f:
            return sum(1 for line in f)

    @staticmethod
    def get_random_sentence():
        words = ["103", "físh", "HUM", "84#ça", "banana", "Oscar", "—"]
        return " ".join([random.choice(words) for _ in range(0, random.randrange(1,len(words)))])

    @classmethod
    def _create_random_parallel_corpus_files(self, path, filename_source="test-corpus.src", filename_target="test_corpus.trg", num_bisegments=200):
        for filename in [filename_source, filename_target]:
            if path:
                filename = os.sep.join([path, filename])
            with open(filename, 'w') as f:
                for i in range(0, num_bisegments):
                    f.write("line %s: %s\n" % (i, self.get_random_sentence()))

    @classmethod
    def _count_lines(filepath):
        return sum(1 for line in open(filepath))

    def get_random_basename(self):
        return str(self._basedir_test_cases + os.sep + str(random.randint(0, 9999999)))

    # preprocessing
    def test_preprocess_base_corpus_file_creation_train_only(self):
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name, "en", "fr", SELFCASING, None, None, None)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, True, False, False)
        files_created = os.listdir(os.sep.join([random_basedir_name, "corpus"]))
        self.assertTrue(
            BASENAME_TRAINING_CORPUS + ".en" in files_created,
            "Training corpus for source language must be created"
        )
        self.assertTrue(
            BASENAME_TRAINING_CORPUS + ".fr" in files_created,
            "Training corpus for target language must be created"
        )

    def test_preprocess_base_corpus_file_creation_train_tune_eval(self):
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name, "en", "fr", SELFCASING, None, 50, 20)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, True, False, False)
        files_created = os.listdir(os.sep.join([random_basedir_name, "corpus"]))
        self.assertTrue(
            BASENAME_TRAINING_CORPUS + ".en" in files_created,
            "Training corpus for source language must be created"
        )
        self.assertTrue(
            BASENAME_TRAINING_CORPUS + ".fr" in files_created,
            "Training corpus for target language must be created"
        )
        self.assertTrue(
            BASENAME_TUNING_CORPUS + ".en" in files_created,
            "Tuning corpus for source language must be created"
        )
        self.assertTrue(
            BASENAME_TUNING_CORPUS + ".fr" in files_created,
            "Tuning corpus for target language must be created"
        )
        self.assertTrue(
            BASENAME_EVALUATION_CORPUS + ".en" in files_created,
            "Evaluation corpus for source language must be created"
        )
        self.assertTrue(
            BASENAME_EVALUATION_CORPUS + ".fr" in files_created,
            "Evaluation corpus for target language must be created"
        )

    def test_preprocess_base_corpus_correct_number_of_lines_train_only(self):
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name, "en", "fr", SELFCASING, None, None, None)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, True, False, False)
        self.assertTrue(
            200 == self.count_lines(os.sep.join([random_basedir_name, "corpus", BASENAME_TRAINING_CORPUS + ".en"])),
            "Number of segments in source side of training corpus must be correct"
        )
        self.assertTrue(
            200 == self.count_lines(os.sep.join([random_basedir_name, "corpus", BASENAME_TRAINING_CORPUS + ".fr"])),
            "Number of segments in target side of training corpus must be correct"
        )

    def test_preprocess_base_corpus_correct_number_of_lines_train_tune_eval(self):
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name, "en", "fr", SELFCASING, None, 50, 20)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, True, False, False)
        self.assertTrue(
            130 == self.count_lines(os.sep.join([random_basedir_name, "corpus", BASENAME_TRAINING_CORPUS + ".en"])),
            "Number of segments in source side of training corpus must be correct"
        )
        self.assertTrue(
            130 == self.count_lines(os.sep.join([random_basedir_name, "corpus", BASENAME_TRAINING_CORPUS + ".fr"])),
            "Number of segments in target side of training corpus must be correct"
        )
        self.assertTrue(
            50 == self.count_lines(os.sep.join([random_basedir_name, "corpus", BASENAME_TUNING_CORPUS + ".en"])),
            "Number of segments in source side of tuning corpus must be correct"
        )
        self.assertTrue(
            50 == self.count_lines(os.sep.join([random_basedir_name, "corpus", BASENAME_TUNING_CORPUS + ".fr"])),
            "Number of segments in target side of tuning corpus must be correct"
        )
        self.assertTrue(
            20 == self.count_lines(os.sep.join([random_basedir_name, "corpus", BASENAME_EVALUATION_CORPUS + ".en"])),
            "Number of segments in source side of evaluation corpus must be correct"
        )
        self.assertTrue(
            20 == self.count_lines(os.sep.join([random_basedir_name, "corpus", BASENAME_EVALUATION_CORPUS + ".fr"])),
            "Number of segments in target side of evaluation corpus must be correct"
        )

    def test_preprocess_external_tuning_corpus(self):
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        t = Training(
            random_basedir_name, "en", "fr", SELFCASING, None,
            tuning=self._basedir_test_cases + os.sep + "external-sample-corpus",
            evaluation=None
        )
        # create sample base corpus
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        # create sample external tuning corpus
        self._create_random_parallel_corpus_files(
            path=self._basedir_test_cases,
            filename_source="external-sample-corpus.en",
            filename_target="external-sample-corpus.fr",
            num_bisegments=50
        )
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, True, False, False)
        self.assertTrue(
            assertions.file_exists(random_basedir_name + os.sep + "corpus" + os.sep + BASENAME_TUNING_CORPUS + ".en"),
            "Source side of external tuning corpus must be created"
        )
        self.assertTrue(
            assertions.file_exists(random_basedir_name + os.sep + "corpus" + os.sep + BASENAME_TUNING_CORPUS + ".fr"),
            "Target side of external tuning corpus must be created"
        )
        self.assertTrue(
            self.count_lines(random_basedir_name + os.sep + "corpus" + os.sep + BASENAME_TUNING_CORPUS + ".en"),
            "Number of segments in source side of external tuning corpus must be correct"
        )
        self.assertTrue(
            self.count_lines(random_basedir_name + os.sep + "corpus" + os.sep + BASENAME_TUNING_CORPUS + ".fr"),
            "Number of segments in target side of external tuning corpus must be correct"
        )

    def test_preprocess_external_eval_corpus(self):
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        t = Training(
            random_basedir_name, "en", "fr", SELFCASING, None,
            tuning=None,
            evaluation=self._basedir_test_cases + os.sep + "external-sample-corpus"
        )
        # create sample base corpus
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        # create sample external eval corpus
        self._create_random_parallel_corpus_files(
            path=self._basedir_test_cases,
            filename_source="external-sample-corpus.en",
            filename_target="external-sample-corpus.fr",
            num_bisegments=50
        )
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, True, False, False)
        self.assertTrue(
            assertions.file_exists(random_basedir_name + os.sep + "corpus" + os.sep + BASENAME_EVALUATION_CORPUS + ".en"),
            "Source side of external evaluation corpus must be created"
        )
        self.assertTrue(
            assertions.file_exists(random_basedir_name + os.sep + "corpus" + os.sep + BASENAME_EVALUATION_CORPUS + ".fr"),
            "Target side of external evaluation corpus must be created"
        )
        self.assertTrue(
            self.count_lines(random_basedir_name + os.sep + "corpus" + os.sep + BASENAME_EVALUATION_CORPUS + ".en"),
            "Number of segments in source side of external evaluation corpus must be correct"
        )
        self.assertTrue(
            self.count_lines(random_basedir_name + os.sep + "corpus" + os.sep + BASENAME_EVALUATION_CORPUS + ".fr"),
            "Number of segments in target side of external evaluation corpus must be correct"
        )

    def test_preprocess_create_lowercased_eval_trg_file(self):
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name, "en", "fr", SELFCASING, None, 50, 20)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, True, False, False)
        self.assertTrue(
            assertions.file_exists(random_basedir_name + os.sep + "corpus" + os.sep + BASENAME_EVALUATION_CORPUS + "." + SUFFIX_LOWERCASED + ".fr"),
            "A lowercased version of the evaluation corpus' target side must be created"
        )

    def test_preprocess_min_tokens(self):
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        with open(os.sep.join([random_basedir_name, 'sample-corpus.en']), 'w') as f:
            f.write('one' + '\n')
            f.write('one two' + '\n')
            f.write('one two three' + '\n')
        with open(os.sep.join([random_basedir_name, 'sample-corpus.fr']), 'w') as f:
            f.write('one two three' + '\n')
            f.write('one two' + '\n')
            f.write('one' + '\n')
        t = Training(random_basedir_name, "en", "fr", SELFCASING, None, None, None)
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]),
            min_tokens=2, max_tokens=80, tokenize_external=False, mask=False, mask_external=False)
        self.assertIs(
            self.count_lines(os.sep.join([random_basedir_name, 'corpus', 'train.en'])),
            1, # only one line satisfies min_tokens for both en and fr
            "There must be no segment with less than min_tokens"
        )
        self.assertIs(
            self.count_lines(os.sep.join([random_basedir_name, 'corpus', 'train.fr'])),
            1, # only one line satisfies min_tokens for both en and fr
            "There must be no segment with less than min_tokens"
        )

    def test_preprocess_max_tokens(self):
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        with open(os.sep.join([random_basedir_name, 'sample-corpus.en']), 'w') as f:
            f.write('one' + '\n')
            f.write('one two' + '\n')
            f.write('one two three' + '\n')
        with open(os.sep.join([random_basedir_name, 'sample-corpus.fr']), 'w') as f:
            f.write('one two three' + '\n')
            f.write('one two' + '\n')
            f.write('one' + '\n')
        t = Training(random_basedir_name, "en", "fr", SELFCASING, None, None, None)
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]),
            min_tokens=1, max_tokens=2, tokenize_external=False, mask=False, mask_external=False)
        self.assertIs(
            self.count_lines(os.sep.join([random_basedir_name, 'corpus', 'train.en'])),
            1, # only one line satisfies max_tokens for both en and fr
            "There must be no segment with less than min_tokens"
        )
        self.assertIs(
            self.count_lines(os.sep.join([random_basedir_name, 'corpus', 'train.fr'])),
            1, # only one line satisfies max_tokens for both en and fr
            "There must be no segment with less than min_tokens"
        )

    def test_preprocess_remove_empty_lines(self):
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        with open(os.sep.join([random_basedir_name, 'sample-corpus.en']), 'w') as f:
            f.write('one' + '\n')
            f.write('\n') # must be removed
            f.write('one two three' + '\n')
            f.write('one two' + '\n')
            f.write('one' + '\n') # must be removed (because .fr is empty)
            f.write('one two' + '\n')
        with open(os.sep.join([random_basedir_name, 'sample-corpus.fr']), 'w') as f:
            f.write('one' + '\n')
            f.write('\n') # must be removed
            f.write('one two three' + '\n')
            f.write('one two' + '\n')
            f.write('\n') # must be removed
            f.write('one two' + '\n')
        t = Training(random_basedir_name, "en", "fr", SELFCASING, None, None, None)
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]),
            min_tokens=1, max_tokens=80, tokenize_external=False, mask=False, mask_external=False)
        self.assertIs(
            self.count_lines(os.sep.join([random_basedir_name, 'corpus', 'train.en'])),
            4, # only one line satisfies max_tokens for both en and fr
            "Bi-segments where src and/or trg are empty lines must be removed"
        )
        self.assertIs(
            self.count_lines(os.sep.join([random_basedir_name, 'corpus', 'train.fr'])),
            4, # only one line satisfies max_tokens for both en and fr
            "Bi-segments where src and/or trg are empty lines must be removed"
        )
