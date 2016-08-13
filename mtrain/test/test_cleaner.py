#!/usr/bin/env python3

from unittest import TestCase

from mtrain.preprocessing import cleaner
from mtrain.constants import *

class TestCleaner(TestCase):
    test_cases = {
        "salt&pepper": "salt&amp;pepper",
        '<a href="index.html">foo</a>': "&lt;a href=&quot;index.html&quot;&gt;foo&lt;/a&gt;",
        "a test's legacy": "a test&apos;s legacy",
        'She said it was "awesome".': "She said it was &quot;awesome&quot;.",
        "[foo]":"&#91;foo&#93;"
    }

    def test_escape_special_chars(self):
        for original, escaped in self.test_cases.items():
            self.assertTrue(
                escaped == cleaner.escape_special_chars(original),
                "Escaped version of `%s` must be `%s`" % (original, escaped)
            )

    def test_deescape_special_chars(self):
        for original, escaped in self.test_cases.items():
            self.assertTrue(
                cleaner.deescape_special_chars(escaped) == original,
                "De-escaped version of `%s` must be `%s`" % (escaped, original)
            )
