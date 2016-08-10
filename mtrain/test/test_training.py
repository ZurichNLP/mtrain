#!/usr/bin/env python3

from unittest import TestCase

from mtrain.training import Training
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
            with open(os.sep.join([path, filename]), 'w') as f:
                for i in range(0, num_bisegments):
                    f.write("line %s: %s\n" % (i, self.get_random_sentence()))

    # preprocessing
    def test_corpus_splitting_file_creation_train_only(self):
        random_basedir_name = self.get_random_name()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(
            corpus_base_path=os.sep.join([random_basedir_name, "sample-corpus"]),
            src_lang="en",
            trg_lang="fr",
            tuning=50,
            evaluation=20
        )
        files_created = os.listdir(os.sep.join([random_basedir_name, "corpus"]))
        self.assertTrue(
            "sample-corpus.train.en" in files_created,
            "Training corpus for source language must be created"
        )
        self.assertTrue(
            "sample-corpus.train.fr" in files_created,
            "Training corpus for target language must be created"
        )
        shutil.rmtree(random_basedir_name)

    def test_corpus_splitting_file_creation_train_tune_eval(self):
        random_basedir_name = self.get_random_name()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(
            corpus_base_path=os.sep.join([random_basedir_name, "sample-corpus"]),
            src_lang="en",
            trg_lang="fr",
            tuning=50,
            evaluation=20
        )
        files_created = os.listdir(os.sep.join([random_basedir_name, "corpus"]))
        self.assertTrue(
            "sample-corpus.train.en" in files_created,
            "Training corpus for source language must be created"
        )
        self.assertTrue(
            "sample-corpus.train.fr" in files_created,
            "Training corpus for target language must be created"
        )
        self.assertTrue(
            "sample-corpus.tune.en" in files_created,
            "Tuning corpus for source language must be created"
        )
        self.assertTrue(
            "sample-corpus.tune.fr" in files_created,
            "Tuning corpus for target language must be created"
        )
        self.assertTrue(
            "sample-corpus.test.en" in files_created,
            "Evaluation corpus for source language must be created"
        )
        self.assertTrue(
            "sample-corpus.test.fr" in files_created,
            "Evaluation corpus for target language must be created"
        )
        shutil.rmtree(random_basedir_name)

    def test_corpus_splitting_correct_number_of_lines_train_only(self):
        random_basedir_name = self.get_random_name()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(
            corpus_base_path=os.sep.join([random_basedir_name, "sample-corpus"]),
            src_lang="en",
            trg_lang="fr"
        )
        self.assertTrue(
            200 == self.count_lines(os.sep.join([random_basedir_name, "corpus", "sample-corpus.train.en"])),
            "Number of segments in source side of training corpus must be correct"
        )
        self.assertTrue(
            200 == self.count_lines(os.sep.join([random_basedir_name, "corpus", "sample-corpus.train.fr"])),
            "Number of segments in target side of training corpus must be correct"
        )
        shutil.rmtree(random_basedir_name)

    def test_corpus_splitting_correct_number_of_lines_train_tune_eval(self):
        random_basedir_name = self.get_random_name()
        os.mkdir(random_basedir_name)
        t = Training(random_basedir_name)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(
            corpus_base_path=os.sep.join([random_basedir_name, "sample-corpus"]),
            src_lang="en",
            trg_lang="fr",
            tuning=50,
            evaluation=20
        )
        self.assertTrue(
            130 == self.count_lines(os.sep.join([random_basedir_name, "corpus", "sample-corpus.train.en"])),
            "Number of segments in source side of training corpus must be correct"
        )
        self.assertTrue(
            130 == self.count_lines(os.sep.join([random_basedir_name, "corpus", "sample-corpus.train.fr"])),
            "Number of segments in target side of training corpus must be correct"
        )
        self.assertTrue(
            50 == self.count_lines(os.sep.join([random_basedir_name, "corpus", "sample-corpus.tune.en"])),
            "Number of segments in source side of tuning corpus must be correct"
        )
        self.assertTrue(
            50 == self.count_lines(os.sep.join([random_basedir_name, "corpus", "sample-corpus.tune.fr"])),
            "Number of segments in target side of tuning corpus must be correct"
        )
        self.assertTrue(
            20 == self.count_lines(os.sep.join([random_basedir_name, "corpus", "sample-corpus.test.en"])),
            "Number of segments in source side of evaluation corpus must be correct"
        )
        self.assertTrue(
            20 == self.count_lines(os.sep.join([random_basedir_name, "corpus", "sample-corpus.test.fr"])),
            "Number of segments in target side of evaluation corpus must be correct"
        )
        shutil.rmtree(random_basedir_name)
