#!/usr/bin/env python3

import os
import shutil
import random

from unittest import TestCase

class TestCaseWithCleanup(TestCase):

    @classmethod
    def setUpClass(cls):
        cls._basedir_test_cases = "test_cases" # temporary folder for files created through nosetests
        os.makedirs(cls._basedir_test_cases, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._basedir_test_cases)

class TestCaseHelper(TestCase):
    '''
    Helper class to avoid code repetition. Methods moved from test_training.py here.

    todo: maybe remove methods from test_evaluator.py where most of these methods are the same
    '''
    @staticmethod
    def count_lines(filename):
        with open(filename) as f:
            return sum(1 for line in f)

    @staticmethod
    def get_random_sentence():
        words = ["103", "físh", "HUM", "84#ça", "banana", "Oscar", "—"]
        return " ".join([random.choice(words) for _ in range(0, random.randrange(1,len(words)))])

    @classmethod
    def _create_random_parallel_corpus_files(cls, path, filename_source="test-corpus.src", filename_target="test_corpus.trg", num_bisegments=200):
        for filename in [filename_source, filename_target]:
            if path:
                filename = os.sep.join([path, filename])
            with open(filename, 'w') as f:
                for i in range(0, num_bisegments):
                    f.write("line %s: %s\n" % (i, cls.get_random_sentence()))

    @classmethod
    def _count_lines(cls, filepath):
        return sum(1 for line in open(filepath))

    def get_random_basename(self):
        return str(self._basedir_test_cases + os.sep + str(random.randint(0, 9999999)))
