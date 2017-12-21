#!/usr/bin/env python3

import logging
import random
import shutil
import sys
import os
import time ###BH just testing

from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup, TestCaseHelper

from mtrain.training import TrainingMoses, TrainingNematus
from mtrain.constants import *
from mtrain import assertions

class TestTrainingMoses(TestCaseWithCleanup, TestCaseHelper):

    # preprocessing
    def test_preprocess_base_corpus_file_creation_train_only(self):
        random_basedir_name = self.get_random_basename()
        os.mkdir(random_basedir_name)
        ###BH changed order of args
        t = TrainingMoses(random_basedir_name, "en", "fr", SELFCASING, None, None, None, XML_PASS_THROUGH)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        ###BH changed order of args not necessary here
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, False, False, False)
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
        ###BH changed order of args
        t = TrainingMoses(random_basedir_name, "en", "fr", SELFCASING, 50, 20, None, XML_PASS_THROUGH)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        ###BH changed order of args
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
        ###BH changed order of args
        t = TrainingMoses(random_basedir_name, "en", "fr", SELFCASING, None, None, None, XML_PASS_THROUGH)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        ###BH changed order of args not necessary here
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, False, False, False)
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
        ###BH changed order of args
        t = TrainingMoses(random_basedir_name, "en", "fr", SELFCASING, 50, 20, None, XML_PASS_THROUGH)
        self._create_random_parallel_corpus_files(
            path=random_basedir_name,
            filename_source="sample-corpus.en",
            filename_target="sample-corpus.fr",
            num_bisegments=200
        )
        ###BH changed order of args
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
        ###BH changed order of args, made mask&xml explicit to fix keyword/positional args order
        t = TrainingMoses(
            random_basedir_name, "en", "fr", SELFCASING,
            tuning=self._basedir_test_cases + os.sep + "external-sample-corpus",
            evaluation=None,
            masking_strategy=None, xml_strategy=XML_PASS_THROUGH
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
        ###BH changed order of args
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
        ###BH changed order of args, made mask&xml explicit to fix keyword/positional args order
        t = TrainingMoses(
            random_basedir_name, "en", "fr", SELFCASING,
            tuning=None,
            evaluation=self._basedir_test_cases + os.sep + "external-sample-corpus",
            masking_strategy=None, xml_strategy=XML_PASS_THROUGH
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
        ###BH changed order of args
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
        ###BH changed order of args
        t = TrainingMoses(random_basedir_name, "en", "fr", SELFCASING, None, None, None, XML_PASS_THROUGH)
        ###BH changed order of args
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]),
            min_tokens=2, max_tokens=80, preprocess_external=False, mask=False, process_xml=False)
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
        ###BH changed order of args
        t = TrainingMoses(random_basedir_name, "en", "fr", SELFCASING, None, None, None, XML_PASS_THROUGH)
        ###BH changed order of args
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]),
            min_tokens=1, max_tokens=2, preprocess_external=False, mask=False, process_xml=False)
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
        ###BH changed order of args
        t = TrainingMoses(random_basedir_name, "en", "fr", SELFCASING, None, None, None, XML_PASS_THROUGH)
        ###BH changed order of args
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]),
            min_tokens=1, max_tokens=80, preprocess_external=False, mask=False, process_xml=False)
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

class TestTrainingNematus(TestCaseWithCleanup, TestCaseHelper):
    '''
    Tests for Nematus mainly adopted from Moses tests where similar.
    test_train_nematus_engine() added for Nematus specific training.
    '''

    # test parallel corpus ro-en:
    ###BH check reference:
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

    def test_preprocess_base_corpus_file_creation_train_tune_eval(self):
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

    def test_preprocess_base_corpus_correct_number_of_lines_train_tune_eval(self):
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

        t = TrainingNematus(
            random_basedir_name, "en", "fr", TRUECASING,
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

        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, True)
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

        t = TrainingNematus(
            random_basedir_name, "en", "fr", TRUECASING,
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

        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, True)
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

        t = TrainingNematus(random_basedir_name, "en", "fr", TRUECASING, None, None)

        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]),
            min_tokens=2, max_tokens=80, preprocess_external=False)
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

        t = TrainingNematus(random_basedir_name, "en", "fr", TRUECASING, None, None)

        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]),
            min_tokens=1, max_tokens=2, preprocess_external=False)
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

        t = TrainingNematus(random_basedir_name, "en", "fr", TRUECASING, None, None)

        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]),
            min_tokens=1, max_tokens=80, preprocess_external=False)
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


    def test_train_nematus_engine(self):
        '''
        Testing nematus training from start to end.

        Checks are only included when not covered in respective processing step. Moreover, file contents are not
        checked as this mainly depends on parameters and data. These are not set here to produce a serviceable model
        (which would take hours or even days to finish) but to perform all necessary steps.

        File checks below are commented according to their importance and when they are possible, if termination
        of test process is initiated due to time consumption, skip respective file checks.
        '''
        '''
        # setup path and filenames for corpus
        random_basedir_name = 'test_cases/nematus_training' ###BH self.get_random_basename()
        os.mkdir(random_basedir_name)
        corpus_file_ro = os.sep.join([random_basedir_name, 'sample-corpus.ro'])
        corpus_file_en = os.sep.join([random_basedir_name, 'sample-corpus.en'])

        # prepare sample parallel corpus
        linecounter = 0
        while linecounter < 1001:
            with open(corpus_file_ro, 'a') as f_src:
                with open(corpus_file_en, 'a') as f_trg:
                    for ro_segment, en_segment in self.test_parallel_corpus_ro_en.items():
                        f_src.write(ro_segment + '\n')
                        f_trg.write(en_segment + '\n')
                        linecounter +=1
        f_src.close()
        f_trg.close()

        # preprocess, truecase and byte-pair encode sample parallel corpus
        t = TrainingNematus(random_basedir_name, "ro", "en", TRUECASING, 100, None)
        t.preprocess(os.sep.join([random_basedir_name, "sample-corpus"]), 1, 80, True)
        t.train_truecaser()############### testing of truecasing model checked in test_truecaser.py
        t.truecase()############### testing of truecased corpora checked in test_truecaser.py
        t.bpe_encoding(1000)############### testing of byte-pair encoding model, encoded files and bpe dictionary checked in test_bpe.py

        # start training
        t.train_engine(
            'cpu', # training device
            '0.01',  # training device preallocated memory, keep low for testing e.g. 0.01 = 1%, more is used if available
            'cpu', # validation device
            '0.01',  # validation device preallocated memory, keep low for testing e.g. 0.01 = 1%, more is used if available
            10,     # validation frequency, keep low for testing even if unrealistic
            None,    # external validation script, cannot be tested as this must be provided by user if chosen
            5000,    # maximum number of epochs, default is fine for test
            10      # maximum number of updates, same as validation frequency so test waits for validation to finish and then stops
        )

        # created before actual training starts. check could be ommited as it is not specific to backend.
        files_created = os.listdir(random_basedir_name)
        self.assertTrue(
            "training.log" in files_created,
            "'training.log' must be created"
        )

        files_created = os.listdir(os.sep.join([random_basedir_name, "engine", "tm", "model"]))
        # created before actual training starts. MUST be checked as training serviceable model is impossible without them.
        self.assertTrue(
            "validate.sh" in files_created,
            "'validate.sh' must be created"
        )
        self.assertTrue(
            "postprocess-dev.sh" in files_created,
            "'postprocess-dev.sh' must be created"
        )

        # created early in training, approximately after 2 minutes. check highly recommended as file is created even with dummy parameters and data.
        self.assertTrue(
            "model.npz.json" in files_created,
            "'model.npz.json' must be created"
        )

        # created early in training, approximately after 3 minutes. check optional because if missing, it is mainly due to unfortunate parameters or data.
        self.assertTrue(
            "model.npz.dev.npz" in files_created,
            "'model.npz.dev.npz' must be created"
        )

        # created rather late in training, approximately after 4 minutes. check optional because if missing, it is mainly due to unfortunate parameters or data.
        self.assertTrue(
            "model.npz.dev.gradinfo.npz" in files_created,
            "'model.npz.dev.gradinfo.npz' must be created"
        )
        self.assertTrue(
            "model.npz.dev.progress.json" in files_created,
            "'model.npz.dev.progress.json' must be created"
        )
        self.assertTrue(
            "model.npz.dev.npz.json" in files_created,
            "'model.npz.dev.npz.json' must be created"
        )
        self.assertTrue(
            "model.npz" in files_created,
            "'model.npz' must be created"
        )
        self.assertTrue(
            "model.npz_bleu_scores" in files_created,
            "'model.npz_bleu_scores' must be created"
        )

        # created rather late in training, approximately after 5 minutes. check optional because if missing, it is mainly due to unfortunate parameters or data.
        self.assertTrue(
            "model.npz.gradinfo.npz" in files_created,
            "'model.npz.gradinfo.npz' must be created"
        )
        self.assertTrue(
            "model.npz.progress.json" in files_created,
            "'model.npz.progress.json' must be created"
        )

        #time.sleep(120) ###BH just testing
        '''