#!/usr/bin/env python3

from unittest import TestCase

from mtrain.preprocessing.xmlprocessor import *
from mtrain.constants import *

class TestXmlProcessor(TestCase):
    
    test_cases_xmlprocessor_markup_stripping = [
        ("", ""),
        ("<b/>", ""),
        ("<i> </i>", ""),
        ("das ist <b> ein </b> test", "das ist ein test"),
        ("das <i> </i> ist <b> ein </b> test", "das ist ein test")
    ]

    def test_xmlprocessor_markup_stripping(self):
        x = XmlProcessor('strip')
        for input, output in self.test_cases_xmlprocessor_markup_stripping:
            self.assertTrue(
                x._strip_markup(input) == output,
                "XML processor did not remove all markup from string or removed too much"
            )
