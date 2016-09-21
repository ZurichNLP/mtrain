#!/usr/bin/env python3

from unittest import TestCase

import os
import shutil

class TestCaseWithCleanup(TestCase):

    @classmethod
    def setUpClass(cls):
        cls._basedir_test_cases = "test_cases" # temporary folder for files created through nosetests
        os.makedirs(cls._basedir_test_cases, exist_ok=True)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls._basedir_test_cases)
