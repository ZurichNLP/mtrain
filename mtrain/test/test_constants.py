#!/usr/bin/env python3

from unittest import TestCase

import mtrain

class TestConstants(TestCase):
    def test_path_to_moses_is_string(self):
        s = mtrain.MOSES_HOME
        self.assertTrue(isinstance(s, str))
    def test_path_to_fastalign_is_string(self):
        s = mtrain.FASTALIGN_HOME
        self.assertTrue(isinstance(s, str))
