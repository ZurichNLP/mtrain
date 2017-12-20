#!/usr/bin/env python3

from unittest import TestCase
from mtrain.preprocessing import cleaner

class TestCleaner(TestCase):
    # dummy test case for not yet implemented clean() method
    test_cases_clean = {
        '': ''
    }
    # test cases for characters with special meaning in Moses
    test_cases_special_chars = {
        "salt&pepper": "salt&amp;pepper",
        '<a href="index.html">foo</a>': "&lt;a href=&quot;index.html&quot;&gt;foo&lt;/a&gt;",
        "a test's legacy": "a test&apos;s legacy",
        'She said it was "awesome".': "She said it was &quot;awesome&quot;.",
        "[foo]":"&#91;foo&#93;"
    }
    # test cases for normalizing Romanian in Nematus
    # cases derived from script https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/normalise-romanian.py ###BH todo add reference
    test_cases_romanian = {
        'Şantierul': 'Șantierul',
        'Totuşi': 'Totuși',
        'Ţivat': 'Țivat',
        'puţin': 'puțin'
    }
    # test cases for removing diacritics from normalized Romanian in Nematus
    # cases derived from script https://github.com/rsennrich/wmt16-scripts/blob/master/preprocess/remove-diacritics.py###BH todo add reference
    test_cases_ro_diacritics = {
        'Șantierul': 'Santierul',
        'Țivat': 'Tivat',
        'Ărsenal': 'Arsenal',
        'Âprobarea': 'Aprobarea',
        'Întreruperea': 'Intreruperea',
        'Totuși': 'Totusi',
        'puțin': 'putin',
        'procedură': 'procedura',
        'Constrângerile': 'Constrangerile',
        'înscrise': 'inscrise'
    }

    def test_clean(self):
        # not yet implemented method, skeleton for test implementation if method used
        for example_segment, clean_segment in self.test_cases_clean.items():
            self.assertEqual(cleaner.clean(example_segment), clean_segment)

    def test_escape_special_chars(self):
        for original, escaped in self.test_cases_special_chars.items():
            self.assertTrue(
                escaped == cleaner.escape_special_chars(original),
                "Escaped version of `%s` must be `%s`" % (original, escaped)
            )

    def test_deescape_special_chars(self):
        for original, escaped in self.test_cases_special_chars.items():
            self.assertTrue(
                cleaner.deescape_special_chars(escaped) == original,
                "De-escaped version of `%s` must be `%s`" % (escaped, original)
            )

    def test_normalize_romanian(self):
        for example_segment, normalized_segment in self.test_cases_romanian.items():
            self.assertEqual(cleaner.normalize_romanian(example_segment), normalized_segment)

    def test_remove_ro_diacritics(self):
        for example_segment, diac_free_segment in self.test_cases_ro_diacritics.items():
            self.assertEqual(cleaner.remove_ro_diacritics(example_segment), diac_free_segment)
