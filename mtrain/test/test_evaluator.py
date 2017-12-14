#!/usr/bin/env python3

import logging
from mtrain.evaluator import Evaluator
from mtrain.constants import *
from mtrain.training import TrainingMoses

from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup

class TestEvaluation(TestCaseWithCleanup):
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

    def test_evaluator(self):
        pass
