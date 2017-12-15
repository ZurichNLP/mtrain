#!/usr/bin/env python3

from unittest import TestCase

import os
import shutil
import random

class TestCaseWithCleanup(TestCase):

    @classmethod
    def setUpClass(cls):
        cls._basedir_test_cases = "test_cases" # temporary folder for files created through nosetests
        os.makedirs(cls._basedir_test_cases, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._basedir_test_cases)

class TestCaseHelper(TestCase):

    @classmethod
    def get_random_basename(cls):
        return str(cls._basedir_test_cases + os.sep + str(random.randint(0, 9999999)))

    @classmethod
    def _create_random_parallel_corpus_files(cls, path, filename_source="test-corpus.src", filename_target="test_corpus.trg", num_bisegments=200):
        for filename in [filename_source, filename_target]:
            if path:
                filename = os.sep.join([path, filename])
            with open(filename, 'w') as f:
                for i in range(0, num_bisegments):
                    f.write("line %s: %s\n" % (i, cls.get_random_sentence()))

    @classmethod
    def get_random_sentence(cls):
        words = ["103", "físh", "HUM", "84#ça", "banana", "Oscar", "—"]
        return " ".join([random.choice(words) for _ in range(0, random.randrange(1,len(words)))])
