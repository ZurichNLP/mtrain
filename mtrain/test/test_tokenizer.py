#!/usr/bin/env python3

from unittest import TestCase

from mtrain.preprocessing.tokenizer import Tokenizer
from mtrain.constants import *

class TestTokenizer(TestCase):

    def test_returns_list(self):
        t = Tokenizer('en')
        self.assertTrue(
            t.tokenize("alpha beta gamma") == ["alpha", "beta", "gamma"],
            "Tokenizer must split segment into a list of tokens"
        )
        t.close()

    def test_nonascii(self):
        t = Tokenizer('en')
        self.assertTrue(
            t.tokenize("Äbé üâbeñA ∑ €") == ["Äbé", "üâbeñA", "∑" , "€"],
            "Tokenizer must handle non-ascii chars correctly"
        )
        t.close()

    def test_escape_special_chars(self):
        t = Tokenizer('en')
        for char, replacement in MOSES_SPECIAL_CHARS.items():
            self.assertTrue(
                t.tokenize(char) == [replacement],
                "Tokenizer must replace special char `%s` with `%s`" % (char, replacement)
            )
        t.close()
