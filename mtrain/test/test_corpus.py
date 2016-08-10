#!/usr/bin/env python3

from unittest import TestCase

from mtrain.corpus import ParallelCorpus
from mtrain import assertions

import os
import random

#todo test cleaner

class TestParallelCorpus(TestCase):
    @staticmethod
    def _create_test_corpus(max_size=None):
        random_basename = random.randint(0, 9999999)
        filename_source = "%s.en" % random_basename
        filename_target = "%s.fr" % random_basename
        return ParallelCorpus(filename_source, filename_target, max_size)

    @staticmethod
    def _remove_test_corpus(corpus):
        corpus.delete()

    def test_corpus_creation(self):
        corpus = self._create_test_corpus()
        filepath_source, filepath_target = corpus.get_filepaths()
        self.assertTrue(
            assertions.file_exists(filepath_source),
            "File for source side of parallel corpus not created"
        )
        self.assertTrue(
            assertions.file_exists(filepath_target),
            "File for target side of parallel corpus not created"
        )
        corpus.close()
        self._remove_test_corpus(corpus)

    def test_corpus_with_size_limit_insertion(self):
        corpus = self._create_test_corpus(max_size=2)
        filepath_source, filepath_target = corpus.get_filepaths()
        segment1_source = "I have the simplest tastes. I am always satisfied with the best."
        segment1_target = "Ich habé einen gânz ëinfachen Geschmack: Ich bin immer mit dem Besten zufrieden."
        corpus.insert(segment1_source, segment1_target)
        corpus.close()
        with open(filepath_source, 'r') as f:
            self.assertTrue(
                f.readline().strip() == segment1_source,
                "Writing source segments to file must not change their content."
            )
        with open(filepath_target, 'r') as f:
            self.assertTrue(
                f.readline().strip() == segment1_target,
                "Writing target segments to file must not change their content."
            )
        self._remove_test_corpus(corpus)

    def test_corpus_with_size_limit_insertion_when_full(self):
        corpus = self._create_test_corpus(max_size=2)
        return_insert_1 = corpus.insert("source:1", "target:1")
        return_insert_2 = corpus.insert("source:2", "target:2")
        return_insert_3 = corpus.insert("source:3", "target:3")
        self.assertIsNone(
            return_insert_1,
            "Corpus should not return segment upon insertion if it isn't full yet"
        )
        self.assertIsNone(
            return_insert_2,
            "Corpus should not return segment upon insertion if it isn't full yet"
        )
        self.assertIsInstance(
            return_insert_3,
            tuple,
            "Corpus must return segment upon insertion if it is full"
        )
        returned_source_segment, returned_target_segment = return_insert_3
        self.assertTrue(
            returned_source_segment.startswith("source"),
            "Corpus must return valid source segment upon insertion if it is full"
        )
        self.assertTrue(
            returned_target_segment.startswith("target"),
            "Corpus must return valid target segment upon insertion if it is full"
        )
        corpus.close()
        self._remove_test_corpus(corpus)

    def test_corpus_without_size_limit_insertion(self):
        corpus = self._create_test_corpus()
        filepath_source, filepath_target = corpus.get_filepaths()
        segment1_source = "I have the simplest tastes. I am always satisfied with the best."
        segment1_target = "Ich habé einen gânz ëinfachen Geschmack: Ich bin immer mit dem Besten zufrieden."
        corpus.insert(segment1_source, segment1_target)
        corpus.close()
        with open(filepath_source, 'r') as f:
            self.assertTrue(
                f.readline().strip() == segment1_source,
                "Writing source segments to file must not change their content."
            )
        with open(filepath_target, 'r') as f:
            self.assertTrue(
                f.readline().strip() == segment1_target,
                "Writing target segments to file must not change their content."
            )
        self._remove_test_corpus(corpus)

    def test_corpus_without_size_limit_flush_immediately(self):
        corpus = self._create_test_corpus()
        filepath_source, filepath_target = corpus.get_filepaths()
        segment1_source = "I have the simplest tastes. I am always satisfied with the best."
        segment1_target = "Ich habé einen gânz ëinfachen Geschmack: Ich bin immer mit dem Besten zufrieden."
        corpus.insert(segment1_source, segment1_target)
        self.assertTrue(
            os.stat(filepath_source).st_size > 0,
            "Source segments must be written to file immediately"
        )
        self.assertTrue(
            os.stat(filepath_target).st_size > 0,
            "Target segments must be written to file immediately"
        )
        corpus.close()
        self._remove_test_corpus(corpus)
