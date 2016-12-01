#!/usr/bin/env python3

from unittest import TestCase

from mtrain.preprocessing import reinsertion
from mtrain.constants import *

class TestReinsertion(TestCase):
    
    test_cases_reinsertion_full = [
        # original, translation, segmentation, alignment, result
        #("", "", {}, {}, ""),
        #("<b/>", "", {}, {}, "<b/>"),
        #("<i> </i>", "", {}, {}, "<i> </i>"),
        # Rule 1
        ("das ist <b> ein </b> test", "this is a test", {(0,1):(0,1), (2,2):(2,2), (3,3):(3,3)}, {0: [0], 1: [1], 2: [2], 3: [3]}),
        # Rule 2
        ("das ist <b> ein </b> test", "this is a test", {(0,1):(0,1), (2,3):(2,3)}, {0: [0], 1: [1], 2: [2], 3: [3]}),
        # Rule 3
        ("das <b> ist ein </b> test", "a test this is", {(0,0):(2,2), (1,1):(3,3), (2,2):(0,0), (3,3):(1,1)}, {0: [2], 1:[3], 2:[0], 3:[1]}),
        # Rule 3.5
        ("das <b> ist ein </b> test", "a test this is", {(0,0):(2,2), (1,1):(3,3), (2,2):(0,0), (3,3):(1,1)}, {0: [2], 1:[3], 2:[0], 3:[1]}),
        # selfclosing markup
        ("das <i/> ist <b> ein </b> test", "this is a test", {(0,1):(0,1), (2,2):(2,2), (3,3):(3,3)}, {0: [0], 1: [1], 2: [2], 3: [3]}),
        # isolated tag pairs
        ("das <i> </i> ist <b> ein </b> test", "this is a test", {(0,1):(0,1), (2,2):(2,2), (3,3):(3,3)}, {0: [0], 1: [1], 2: [2], 3: [3]})
    ]

    def test_reinsertion_full(self):
        r = reinsertion.Reinserter('full')

        for original, translation, segmentation, alignment in self.test_cases_reinsertion_full:
            try:
                r.reinsert_markup(original, translation, segmentation, alignment)
            except:
                self.fail("Full reinsertion failed to reinsert markup into translated segments")

    test_cases_reinsertion_alignment = [
        ("das ist <b> ein </b> test", "this is a test", {(0,1):(0,1), (2,2):(2,2), (3,3):(3,3)}, {0: [0], 1: [1], 2: [2], 3: [3]}),
        ("das <b> ist ein </b> test", "a test this is", {(0,0):(2,2), (1,1):(3,3), (2,2):(0,0), (3,3):(1,1)}, {0: [2], 1:[3], 2:[0], 3:[1]}),
        ("das <i/> ist <b> ein </b> test", "this is a test", {(0,1):(0,1), (2,2):(2,2), (3,3):(3,3)}, {0: [0], 1: [1], 2: [2], 3: [3]}),
        ("das <i> </i> ist <b> ein </b> test", "this is a test", {(0,1):(0,1), (2,2):(2,2), (3,3):(3,3)}, {0: [0], 1: [1], 2: [2], 3: [3]})
    ]

    def test_reinsertion_alignment(self):
        r = reinsertion.Reinserter('alignment')
        for original, translation, segmentation, alignment in self.test_cases_reinsertion_alignment:
            try:
                r.reinsert_markup(original, translation, segmentation, alignment)
            except:
                self.fail("Alignment reinsertion failed to reinsert markup into translated segments")

    test_cases_reinsertion_segmentation = [
        ("das ist <b> ein </b> test", "this is a test", {(0,1):(0,1), (2,2):(2,2), (3,3):(3,3)}, {0: [0], 1: [1], 2: [2], 3: [3]}),
        ("das <b> ist ein </b> test", "a test this is", {(0,0):(2,2), (1,1):(3,3), (2,2):(0,0), (3,3):(1,1)}, {0: [2], 1:[3], 2:[0], 3:[1]}),
        ("das <i/> ist <b> ein </b> test", "this is a test", {(0,1):(0,1), (2,2):(2,2), (3,3):(3,3)}, {0: [0], 1: [1], 2: [2], 3: [3]}),
        ("das <i> </i> ist <b> ein </b> test", "this is a test", {(0,1):(0,1), (2,2):(2,2), (3,3):(3,3)}, {0: [0], 1: [1], 2: [2], 3: [3]})
    ]

    def test_reinsertion_segmentation(self):
        r = reinsertion.Reinserter('segmentation')
        for original, translation, segmentation, alignment in self.test_cases_reinsertion_segmentation:
            try:
                r.reinsert_markup(original, translation, segmentation, alignment)
            except:
                self.fail("Segmentation reinsertion failed to reinsert markup into translated segment")

    test_cases_element_names_identical = [
        ("<b>", "</b>", True),
        ("<b attr='1'>", "</b>", True),
        ("<b>", "</i>", False),
        ("gibberish", "jkjbh", False),
        ("", "", False)
    ]

    def test_element_names_identical(self):
        for opening_tag, closing_tag, boolean in self.test_cases_element_names_identical:
            self.assertTrue(
                reinsertion._element_names_identical(opening_tag, closing_tag) == boolean,
                "Elements are reported to be identical even if they are nor or vice versa"
            )

    test_cases_is_xml_comment = [
        ("<!-- this is a comment -->", True),
        ("<! this is not a complete comment", False),
        ("other string", False),
        ("<!---->", True),
        ("<b> no comment </b>", False)
    ]

    def test_is_xml_comment(self):
        for token, boolean in self.test_cases_is_xml_comment:
            self.assertTrue(
                reinsertion._is_xml_comment(token) == boolean,
                "some strings are erroneously considered to be comments or vice versa"
            )

    def test_unknown_strategy_raises_error(self):
        with self.assertRaises(NotImplementedError):
            r = reinsertion.Reinserter('unknown strategy')
            r.reinsert_markup("foo", "foo", {}, {})
