#!/usr/bin/env python3

from unittest import TestCase
from mtrain.preprocessing.normalizer import Normalizer

class TestNormalizer(TestCase):
    '''
    Test cases and comments derived from script normalize-punctuation.perl.

    Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
    '''
    # English test cases
    test_cases_en = {
        'This  has  too  much  spaces!': 'This has too much spaces!', # remove extra spaces
        '„This has “weird“ signs in it…“': '"This has "weird" signs in it..."', # normalize unicode punctuation
        'This has a space at the wrong place : There it was.': 'This has a space at the wrong place: There it was.', # handle pseudo-spaces
        '"English normalizer does not like commas after quotes", the script says.': '"English normalizer does not like commas after quotes," the script says.' # English "quotation," followed by comma, style
    }
    # French test cases
    test_cases_fr = {
        '«Une citation française»': '"Une citation française"', # French quotes
        '"Je ne comprend pas," il repond.': '"Je ne comprend pas", il repond.' # German/Spanish/French "quotation", followed by comma, style, opposite to English normalizer for some reason
    }

    def test_normalize_punctuation(self):
        '''
        Testing implementation of script normalize-punctuation.perl.
        '''
        # English normalizer
        normalizer_en = Normalizer("en")
        for example_segment, normalized_segment in self.test_cases_en.items():
            self.assertEqual(normalizer_en.normalize_punctuation(example_segment), normalized_segment)
        normalizer_en.close()

        # French normalizer
        normalizer_fr = Normalizer("fr")
        for example_segment, normalized_segment in self.test_cases_fr.items():
            self.assertEqual(normalizer_fr.normalize_punctuation(example_segment), normalized_segment)
        normalizer_fr.close()
