#!/usr/bin/env python3

from unittest import TestCase
from collections import Counter
import random

from mtrain import constants
from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup

# override protected patterns before import, since they can be modified in mtrain.constants.py:
constants.PROTECTED_PATTERNS = {}
constants.PROTECTED_PATTERNS['xml'] = r'<\/?[a-zA-Z_][a-zA-Z_.\-0-9]*[^<>]*\/?>'
constants.PROTECTED_PATTERNS['email'] = r'[\w\-\_\.]+\@([\w\-\_]+\.)+[a-zA-Z]{2,}'
constants.PROTECTED_PATTERNS['url'] = r'(?:(?:https?|ftp)://)(?:\S+(?::\S*)?@)?(?:(?:[1-9]\d?|1\d\d|2[01]\d|22[0-3])(?:\.(?:1?\d{1,2}|2[0-4]\d|25[0-5])){2}(?:\.(?:[1-9]\d?|1\d\d|2[0-4]\d|25[0-4]))|(?:(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)(?:\.(?:[a-z\u00a1-\uffff0-9]+-?)*[a-z\u00a1-\uffff0-9]+)*(?:\.(?:[a-z\u00a1-\uffff]{2,})))(?::\d{2,5})?(?:/[^\s]*)?'

from mtrain.preprocessing.masking import *

class TestIdentityMasker(TestCase):

    test_cases_identity_masking = [
        ("Email me at an@ribute.com . . .", 'Email me at __email_0__ . . .', [('__email_0__', 'an@ribute.com')]),
        ("http://www.statmt.org is a terrific site !", '__url_0__ is a terrific site !', [('__url_0__', 'http://www.statmt.org')]),
        ("Here is a terrific site : http://www.statmt.org .", 'Here is a terrific site : __url_0__ .', [('__url_0__', 'http://www.statmt.org')]),
        ("<all> in the sky much </all>", '__xml_0__ in the sky much __xml_1__', [('__xml_0__', '<all>'), ('__xml_1__', '</all>')]),
        ("in the <b> sky </b> much", 'in the __xml_0__ sky __xml_1__ much', [('__xml_0__', '<b>'), ('__xml_1__', '</b>')])
    ]

    def test_identity_masking(self):
        m = Masker('identity')
        for unmasked, masked, mapping in self.test_cases_identity_masking:
            self.assertTrue(
                m.mask_segment(unmasked)[0] == masked,
                "Identity masking must replace matching tokens with unique masks"
            )
    
    def test_identity_masking_escape_characters(self):
        m = Masker('identity')
        self.assertTrue(
            m.mask_segment("the ships & hung < in the > [ sky ] .")[0] == "the ships &amp; hung &lt; in the &gt; &#91; sky &#93; .",
            "Masker must escape characters reserved in Moses by default"
        )

    def test_identity_masking_do_not_escape_option(self):
        m = Masker('identity', escape=False)
        self.assertTrue(
            m.mask_segment("the ships & hung < in the > [ sky ] .")[0] == "the ships & hung < in the > [ sky ] .",
            "Masker must not escape characters reserved in Moses if requested explicitly"
        )

    def test_identity_masking_return_mapping(self):
        m = Masker('identity')
        for unmasked, masked, mapping in self.test_cases_identity_masking:
            self.assertTrue(
                m.mask_segment(unmasked)[1] == mapping,
                "Identity masker must return a list of mappings (mask_token, original_content)."
            )

    def test_identity_unmasking(self):
        m = Masker('identity')
        for unmasked, masked, mapping in self.test_cases_identity_masking:
            self.assertTrue(
                m.unmask_segment(unmasked, masked, mapping) == unmasked,
                "Identity masker must restore masks correctly given a translated segment and a mapping"
            )

    test_cases_identity_force_mask_translation = [
        ("", ""),
        ("text without masks", "text without masks"),
        ("text with single __identity_1__ mask", 'text with single <mask translation="__identity_1__">__identity_1__</mask> mask'),
        ("more than __one_1__ identity __mask_1__", 'more than <mask translation="__one_1__">__one_1__</mask> identity <mask translation="__mask_1__">__mask_1__</mask>')
    ]

    def test_identity_force_mask_translation(self):
        m = Masker('identity')
        for masked, forced in self.test_cases_identity_force_mask_translation:
            self.assertTrue(
               m.force_mask_translation(masked) == forced,
                "Identity masking must replace mask tokens with Moses forced translation directives"
            )

    def test_identity_mask_tokens(self):
        m = Masker('identity')
        result, mapping = m.mask_tokens('Email me at an@ribute.com or <a> http://www.statmt.org </a>'.split(" "))

        self.assertTrue(
            result == ['Email', 'me', 'at', '__email_0__', 'or', '__xml_0__', '__url_0__', '__xml_1__'],
            "Identity masking did not mask a list of tokens correctly"
        )
        self.assertTrue(
            _mappings_equal(mapping, [('__url_0__', 'http://www.statmt.org'), ('__xml_0__', '<a>'), ('__xml_1__', '</a>'), ('__email_0__', 'an@ribute.com')]),
            "Identity masking did not return the correct mapping for a list of input tokens"
        )

class TestAlignmentMasker(TestCase):
    
    test_cases_alignment_masking = [
        ('Email me at an@ribute.com . . .', 'Email me at __email__ . . .', [('__email__', 'an@ribute.com')]),
        ('http://www.statmt.org is a terrific site !', '__url__ is a terrific site !', [('__url__', 'http://www.statmt.org')]),
        ('Here is a terrific site : http://www.statmt.org .', 'Here is a terrific site : __url__ .', [('__url__', 'http://www.statmt.org')]),
        ('<all> in the sky much </all>', '__xml__ in the sky much __xml__', [('__xml__', '<all>'), ('__xml__', '</all>')]),
        ('in the <b> sky </b> much', 'in the __xml__ sky __xml__ much', [('__xml__', '<b>'), ('__xml__', '</b>')]),
        ('Email me at an@ribute.com or <a> http://www.statmt.org </a>', 'Email me at __email__ or __xml__ __url__ __xml__', [('__url__', 'http://www.statmt.org'),('__xml__', '<a>'), ('__xml__', '</a>'), ('__email__', 'an@ribute.com')])
    ]

    def test_alignment_masking(self):
        m = Masker('alignment')
        for unmasked, masked, mapping in self.test_cases_alignment_masking:
            self.assertTrue(
                m.mask_segment(unmasked)[0] == masked,
                "Alignment masking must insert mask tokens in the correct places"
            )

    def test_alignment_mapping(self):
        m = Masker('alignment')
        for unmasked, masked, mapping in self.test_cases_alignment_masking:
            self.assertTrue(
                _mappings_equal(m.mask_segment(unmasked)[1], mapping),
                "Alignment masking must correctly record the mapping between mask tokens and their original content"
            )

    test_cases_alignment_unmasking = [
        ('Email me at __email__ . . .', 'Message moi à __email__ . . .', [('__email__',
            'an@ribute.com')], {0:[0], 1:[1], 2:[2], 3:[3], 4:[4], 5:[5], 6:[6]}, 'Message moi à an@ribute.com . . .'),
        ('__url__ is a terrific site !', '__url__ c&apos; est une site super chouette !',
            [('__url__', 'http://www.statmt.org')], {0:[0], 1:[1,2], 2:[3], 3:[5,6], 4:[4], 5:[7]}, 'http://www.statmt.org c&apos; est une site super chouette !'),
        ('Here is a terrific site : __url__ .', 'Ici est une site super chouette : __url__ .',
            [('__url__', 'http://www.statmt.org')], {0:[0], 1:[1], 2:[2], 3:[4,5], 4:[3], 5:[6], 6:[7], 7:[8]}, 'Ici est une site super chouette : http://www.statmt.org .'),
        ('__xml__ in the sky much __xml__', '__xml__ dans le ciel beaucoup __xml__',
            [('__xml__', '<all>'), ('__xml__', '</all>')], {0:[0], 1:[1], 2:[2], 3:[3], 4:[4], 5:[5]}, '<all> dans le ciel beaucoup </all>'),
        ('in the __xml__ sky __xml__ much', 'dans le __xml__ ciel __xml__ beaucoup',
            [('__xml__', '<b>'), ('__xml__', '</b>')], {0:[0], 1:[1], 2:[2], 3:[3], 4:[4], 5:[5]}, 'dans le <b> ciel </b> beaucoup'),
        ('Email me at __email__ or __xml__ __url__ __xml__', 'Message moi à __email__ ou __xml__ __url__ __xml__',
            [('__url__', 'http://www.statmt.org'),('__xml__', '<a>'), ('__xml__', '</a>'), ('__email__', 'an@ribute.com')], {0:[0], 1:[1], 2:[2], 3:[3], 4:[4], 5:[5], 6:[6], 7:[7]},
            'Message moi à an@ribute.com ou <a> http://www.statmt.org </a>')
    ]

    def test_alignment_unmasking(self):
        m = Masker('alignment')
        for source, target, mapping, alignment, final_result in self.test_cases_alignment_unmasking:
            self.assertTrue(
                m.unmask_segment(source, target, mapping, alignment) == final_result,
                "Alignment masking must restore markup in translated text based on the source segment, target segment, mapping and alignment"
            )
    
    test_cases_alignment_force_mask_translation = [
        ("text with __single__ alignment mask", 'text with <mask translation="__single__">__single__</mask> alignment mask'),
        ("Email me at __email__ or __xml__ __url__ __xml__", 'Email me at <mask translation="__email__">__email__</mask> or <mask translation="__xml__">__xml__</mask> <mask translation="__url__">__url__</mask> <mask translation="__xml__">__xml__</mask>'),
        ("text without masks", "text without masks"),
        ("", "")        
    ]

    def test_alignment_force_mask_translation(self):
        m = Masker('alignment')
        for masked, forced in self.test_cases_alignment_force_mask_translation:
            self.assertTrue(
                m.force_mask_translation(masked) == forced,
                "Alignment masking must replace mask tokens with forced translation directives if requested"
            )

class TestWritingPatterns(TestCaseWithCleanup):
    
    lines_that_should_be_read = [
        '# xml\n', '<\\/?[a-zA-Z_][a-zA-Z_.\\-0-9]*[^<>]*\\/?>\n',
        '# email\n', '[\\w\\-\\_\\.]+\\@([\\w\\-\\_]+\\.)+[a-zA-Z]{2,}\n',
        '# url\n', '(?:(?:https?|ftp)://)(?:\\S+(?::\\S*)?@)?(?:(?:[1-9]\\d?|1\\d\\d|2[01]\\d|22[0-3])(?:\\.(?:1?\\d{1,2}|2[0-4]\\d|25[0-5])){2}(?:\\.(?:[1-9]\\d?|1\\d\\d|2[0-4]\\d|25[0-4]))|(?:(?:[a-z\\u00a1-\\uffff0-9]+-?)*[a-z\\u00a1-\\uffff0-9]+)(?:\\.(?:[a-z\\u00a1-\\uffff0-9]+-?)*[a-z\\u00a1-\\uffff0-9]+)*(?:\\.(?:[a-z\\u00a1-\\uffff]{2,})))(?::\\d{2,5})?(?:/[^\\s]*)?\n'
    ]

    def test_write_masking_patterns_all(self):
        random_filename = self._basedir_test_cases + os.sep + str(random.randint(0, 9999999))
        write_masking_patterns(random_filename, markup_only=False)
        with open(random_filename, 'r') as random_file:
            self.assertTrue(
                Counter(random_file.readlines()) == Counter(self.lines_that_should_be_read),
                "Masking patterns must be written to file faithfully"
            )

    def test_write_masking_patterns_markup_only(self):
        random_filename = self._basedir_test_cases + os.sep + str(random.randint(0, 9999999))
        write_masking_patterns(random_filename, markup_only=True)
        with open(random_filename, 'r') as random_file:
            self.assertTrue(
                random_file.read() == '# xml\n<\\/?[a-zA-Z_][a-zA-Z_.\\-0-9]*[^<>]*\\/?>\n',
                "If requested, only the pattern to protect markup should be written to a file"
            )


def _mappings_equal(mapping_a, mapping_b):
    return Counter(mapping_a) == Counter(mapping_b)
