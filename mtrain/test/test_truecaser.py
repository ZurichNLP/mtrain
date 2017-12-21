#!/usr/bin/env python3

import os

from unittest import TestCase
from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup, TestCaseHelper
from mtrain.preprocessing.normalizer import Normalizer
from mtrain.preprocessing.tokenizer import Tokenizer, Detokenizer
from mtrain.preprocessing.truecaser import Truecaser, Detruecaser
from mtrain.training import TrainingNematus
from mtrain.constants import *
from mtrain import commander

class TestTruecaser(TestCaseWithCleanup, TestCaseHelper):
    # english test cases for training and testing the truecaser
    # cases derived from script truecase.perl ###BH todo add reference
    test_cases = {
        'This is a sample sentence that was said by Obama, allegedly.': 'this is a sample sentence that was said by Obama, allegedly.',
        'As this sentence uses lowercased words that appeared as uppercased before, Obama said these are actually lowercased.': 'as this sentence uses lowercased words that appeared as uppercased before, Obama said these are actually lowercased.',
        'The only words that are always uppercased must therefore be a name or maybe a concept, as this text is in English, Obama said.': 'the only words that are always uppercased must therefore be a name or maybe a concept, as this text is in English, Obama said.',
        'In conclusion, the trucaser must lowercase sentence initial word unless it is a name, concept or the like.': 'in conclusion, the trucaser must lowercase sentence initial word unless it is a name, concept or the like.'
    }

    def _prepare_truecase_tokens(self):
        '''
        Train truecaser model according to https://github.com/rsennrich/wmt16-scripts/blob/master/sample/preprocess.sh. ###BH todo add reference
        Sample corpus and truecasing model training use the same test cases that are used to test the truecasing model (see test_truecase_tokens)).
        '''
        # setup paths and filenames for sample corpus, corpus and tc model
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)

        sample_corpus = os.sep.join([random_basedir_name, 'sample_corpus.en'])

        corpus_path = os.sep.join([random_basedir_name, 'corpus'])
        os.mkdir(corpus_path)
        corpus = os.sep.join([corpus_path, 'train.en'])

        engine_path = os.sep.join([random_basedir_name, 'engine'])
        os.mkdir(engine_path)
        model_path = os.sep.join([engine_path, TRUECASING])
        os.mkdir(model_path)
        model = os.sep.join([model_path, 'model.en'])

        # prepare sample corpus from test cases
        with open(sample_corpus, 'a') as f:
            for example_segment, truecased_segment in self.test_cases.items():
                f.write(example_segment + '\n')
        f.close()

        # preprocess sample corpus as in training
        normalizer = Normalizer('en')
        tokenizer = Tokenizer('en')

        with open(sample_corpus, 'r') as f_sample:
            with open(corpus, 'a') as f_corpus:
                for segment in f_sample:
                    preprocessed_tokens = (tokenizer.tokenize(normalizer.normalize_punctuation(segment.strip())))
                    preprocessed_segment =" ".join(preprocessed_tokens)
                    f_corpus.write(preprocessed_segment + '\n')
            f_corpus.close()
        f_sample.close()

        normalizer.close()
        tokenizer.close()

        # train truecaser
        def command():
            return '{script} --model {model} --corpus {corpus}'.format(
                script=MOSES_TRAIN_TRUECASER,
                model=model,
                corpus=corpus
            )
        commands = [command()]
        commander.run(commands)

        # check creation of truecasing model
        files_created = os.listdir(os.sep.join([random_basedir_name, "engine", TRUECASING]))
        self.assertTrue(
            'model.en' in files_created,
            "Truecasing model for respective language must be created"
        )

        # save model full path for passing to test method
        self._model = model

    def test_truecase_tokens(self):
        '''
        Check functionality of truecaser. Using normalizer, tokenizer and detokenizer for better readability of test cases and include more code coverage.
        Sample corpus and truecasing model training (see _prepare_truecase_tokens()) use the same test cases that are used to test the truecasing model.
        '''
        self._prepare_truecase_tokens()
        # load English normalizer
        normalizer = Normalizer('en')
        # load English tokenizer
        tokenizer = Tokenizer('en')
        # load truecaser using truecasing model
        truecaser = Truecaser(self._model)
        # load English detokenizer 
        detokenizer = Detokenizer('en')

        # using truecase_tokens() instead of truecase() to increase code coverage of truecaser.py
        for example_segment, truecased_segment in self.test_cases.items():
            self.assertEqual(detokenizer.detokenize(truecaser.truecase_tokens(tokenizer.tokenize(normalizer.normalize_punctuation(example_segment)))), truecased_segment)

        # cleanup
        normalizer.close()
        tokenizer.close()
        truecaser.close()
        detokenizer.close()

    def test_truecase_files_all_sets(self):
        '''
        Random samples for all sets for testing creation of truecasing files. Test derived from test_training.py.
        '''
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)

        t = TrainingNematus(random_basedir_name, "en", "fr", TRUECASING, 50, 20)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, True)
        t.train_truecaser()
        t.truecase()

        files_created = os.listdir(os.sep.join([random_basedir_name, "corpus"]))
        self.assertTrue(
            BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED + ".en" in files_created,
            "Truecased training corpus for source language must be created"
        )
        self.assertTrue(
            BASENAME_TRAINING_CORPUS + "." + SUFFIX_TRUECASED + ".fr" in files_created,
            "Truecased training corpus for target language must be created"
        )
        self.assertTrue(
            BASENAME_TUNING_CORPUS + "." + SUFFIX_TRUECASED + ".en" in files_created,
            "Truecased tuning corpus for source language must be created"
        )
        self.assertTrue(
            BASENAME_TUNING_CORPUS + "." + SUFFIX_TRUECASED + ".fr" in files_created,
            "Truecased tuning corpus for target language must be created"
        )
        self.assertTrue(
            BASENAME_EVALUATION_CORPUS + "." + SUFFIX_TRUECASED + ".en" in files_created,
            "Truecased evaluation corpus for source language must be created"
        )
        self.assertTrue(
            BASENAME_EVALUATION_CORPUS + "." + SUFFIX_TRUECASED + ".fr" in files_created,
            "Truecased evaluation corpus for target language must be created"
        )

class TestDetruecaser(TestCase):
    # english test cases, reversed examples from TestTruecaser() class to match detruecase.perl ###BH todo add reference
    test_cases = {
        'this is a sample sentence that was said by Obama, allegedly.': 'This is a sample sentence that was said by Obama, allegedly.',
        'as this sentence uses lowercased words that appeared as uppercased before, Obama said these are actually lowercased.': 'As this sentence uses lowercased words that appeared as uppercased before, Obama said these are actually lowercased.',
        'the only words that are always uppercased must therefore be a name or maybe a concept, as this text is in English, Obama said.': 'The only words that are always uppercased must therefore be a name or maybe a concept, as this text is in English, Obama said.',
        'in conclusion, the trucaser must lowercase sentence initial word unless it is a name, concept or the like.': 'In conclusion, the trucaser must lowercase sentence initial word unless it is a name, concept or the like.'
    }

    def test_detruecase(self):
        # detruecasing (uppercase first word of sentence)
        detruecaser = Detruecaser()
        for example_segment, detruecased_segment in self.test_cases.items():
            self.assertEqual(detruecaser.detruecase(example_segment), detruecased_segment)
        detruecaser.close()
