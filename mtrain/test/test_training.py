#!/usr/bin/env python3

from unittest import TestCase

from mtrain.training import Training
from mtrain.constants import *
from mtrain import assertions

import random
import shutil
import os

class TestTraining(TestCase):
    @staticmethod
    def count_lines(filename):
        with open(filename) as f:
            return sum(1 for line in f)

    @staticmethod
    def get_random_name():
        return str(random.randint(0, 9999999))

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

    # preprocessing
    def test_preprocess_base_corpus_file_creation_train_only(self):
        random_basedir_name = self.get_random_name()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name, "en", "fr", SELFCASING, None, None, 1, 80)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(base_corpus_path=os.sep.join([random_basedir_name, "sample-corpus"]))
        files_created = os.listdir(os.sep.join([random_basedir_name, "corpus"]))
        self.assertTrue(
            BASENAME_TRAINING_CORPUS + ".en" in files_created,
            "Training corpus for source language must be created"
        )
        self.assertTrue(
            BASENAME_TRAINING_CORPUS + ".fr" in files_created,
            "Training corpus for target language must be created"
        )
        shutil.rmtree(random_basedir_name)

    def test_preprocess_base_corpus_file_creation_train_tune_eval(self):
        random_basedir_name = self.get_random_name()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name, "en", "fr", SELFCASING, 50, 20, 1, 80)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(base_corpus_path=os.sep.join([random_basedir_name, "sample-corpus"]))
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
        shutil.rmtree(random_basedir_name)

    def test_preprocess_base_corpus_correct_number_of_lines_train_only(self):
        random_basedir_name = self.get_random_name()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name, "en", "fr", SELFCASING, None, None, 1, 80)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(base_corpus_path=os.sep.join([random_basedir_name, "sample-corpus"]))
        self.assertTrue(
            200 == self.count_lines(os.sep.join([random_basedir_name, "corpus", BASENAME_TRAINING_CORPUS + ".en"])),
            "Number of segments in source side of training corpus must be correct"
        )
        self.assertTrue(
            200 == self.count_lines(os.sep.join([random_basedir_name, "corpus", BASENAME_TRAINING_CORPUS + ".fr"])),
            "Number of segments in target side of training corpus must be correct"
        )
        shutil.rmtree(random_basedir_name)

    def test_preprocess_base_corpus_correct_number_of_lines_train_tune_eval(self):
        random_basedir_name = self.get_random_name()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name, "en", "fr", SELFCASING, 50, 20, 1, 80)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(base_corpus_path=os.sep.join([random_basedir_name, "sample-corpus"]))
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
        shutil.rmtree(random_basedir_name)

    def test_preprocess_external_tuning_corpus(self):
        random_basedir_name = self.get_random_name()
        os.mkdir(random_basedir_name)
        t = Training(
            random_basedir_name, "en", "fr", SELFCASING,
            tuning="external-sample-corpus",
            evaluation=None,
            min_tokens=1, max_tokens=80
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
            path="",
            filename_source="external-sample-corpus.en",
            filename_target="external-sample-corpus.fr",
            num_bisegments=50
        )
        t.preprocess(base_corpus_path=os.sep.join([random_basedir_name, "sample-corpus"]))
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
        shutil.rmtree(random_basedir_name)
        os.remove("external-sample-corpus.en")
        os.remove("external-sample-corpus.fr")

    def test_preprocess_external_eval_corpus(self):
        random_basedir_name = self.get_random_name()
        os.mkdir(random_basedir_name)
        t = Training(
            random_basedir_name, "en", "fr", SELFCASING,
            tuning=None,
            evaluation="external-sample-corpus",
            min_tokens=1, max_tokens=80
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
            path="",
            filename_source="external-sample-corpus.en",
            filename_target="external-sample-corpus.fr",
            num_bisegments=50
        )
        t.preprocess(base_corpus_path=os.sep.join([random_basedir_name, "sample-corpus"]))
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
        shutil.rmtree(random_basedir_name)
        os.remove("external-sample-corpus.en")
        os.remove("external-sample-corpus.fr")

    def test_preprocess_create_lowercased_eval_trg_file(self):
        random_basedir_name = self.get_random_name()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name, "en", "fr", SELFCASING, 50, 20, 1, 80)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(base_corpus_path=os.sep.join([random_basedir_name, "sample-corpus"]))
        self.assertTrue(
            assertions.file_exists(random_basedir_name + os.sep + "corpus" + os.sep + BASENAME_EVALUATION_CORPUS + "." + SUFFIX_LOWERCASED + ".fr"),
            "A lowercased version of the evaluation corpus' target side must be created"
        )
        shutil.rmtree(random_basedir_name)
