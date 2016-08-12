#!/usr/bin/env python3

from unittest import TestCase

from mtrain.preprocessing import cleaner
from mtrain.constants import *

class TestCleaner(TestCase):

    def test_escape_special_chars(self):
        test_cases = [
            '<a href="index.html">foo</a>',
            "a test's legacy",
            'She said it was "awesome".',
            "[foo]"
        ]
        for test_case in test_cases:
            for char in MOSES_SPECIAL_CHARS.keys():
                self.assertTrue(
                    char not in cleaner.escape_special_chars(test_case),
                    "Special char `%s` must not be escaped in cleaned semgnets" % char
                )

    def test_deescape_special_chars(self):
        test_cases = [
            "&lt; a href = &quot; index.html &quot; &gt; foo &lt; / a &gt;",
            "a test &apos; s legacy",
            "She said it was &quot; awesome &quot; .",
            "&#91; foo &#93;"
        ]
        for test_case in test_cases:
            for replacement in MOSES_SPECIAL_CHARS.values():
                self.assertTrue(
                    replacement not in cleaner.escape_special_chars(test_case),
                    "`%s` must be de-escaped" % replacement
                )
