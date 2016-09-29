#!/usr/bin/env python3

from unittest import TestCase

from mtrain.preprocessing.masking import Masker
from mtrain.constants import *

class TestIdentityMasker(TestCase):
    
    def test_mask_inline_xml(self):
        m = Masker('identity')
        self.assertTrue(
            m.mask_segment("in the <b> sky </b> much")[0] == "in the __xml_0__ sky __xml_1__ much",
            "Identity masker must replace inline XML markup with unique mask tokens"
        )

    def test_mask_wrapping_xml(self):
        m = Masker('identity')
        self.assertTrue(
            m.mask_segment("<all> in the sky much </all>")[0] == "__xml_0__ in the sky much __xml_1__",
            "Identity masker must replace wrapping XML markup with unique mask tokens"
        )

    def test_escape_characters(self):
        m = Masker('identity')
        self.assertTrue(
            m.mask_segment("the ships & hung < in the > [ sky ] .")[0] == "the ships &amp; hung &lt; in the &gt; &#91; sky &#93; .",
            "Masker must escape characters reserved in Moses by default"
        )

    def test_do_not_escape_option(self):
        m = Masker('identity', escape=False)
        self.assertTrue(
            m.mask_segment("the ships & hung < in the > [ sky ] .")[0] == "the ships & hung < in the > [ sky ] .",
            "Masker must not escape characters reserved in Moses if requested explicitly"
        )

    def test_return_mapping(self):
        m = Masker('identity')
        self.assertTrue(
            m.mask_segment("in the <b> sky </b> much")[1] == [('__xml_0__', '<b>'), ('__xml_1__', '</b>')],
            "Identity masker must return a list of mappings (mask_token, original_content) for restoration"
        )

    def test_unmask_xml(self):
        m = Masker('identity')
        self.assertTrue(
            m.unmask_segment("in the __xml_0__ sky __xml_1__ much", [('__xml_0__', '<b>'), ('__xml_1__', '</b>')]) == "in the <b> sky </b> much",
            "Identity masker must restore markup correctly given a translated segment and a mapping"
        )
