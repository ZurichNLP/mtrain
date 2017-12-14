#!/usr/bin/env python3

from mtrain.test.test_case_with_cleanup import TestCaseWithCleanup

from mtrain.preprocessing import bpe

'''
class TestEncoder(?):
    def test_learn_bpe_model(?):
        pass
    def test_apply_bpe_model(?):
        pass
    def test_build_bpe_dictionary(?):
        pass

class TestTranslationEncoder(?):
    def test_encode(?):
        pass
'''

class TestTranslationDecoder(TestCaseWithCleanup):
    '''
    Examples correspond to normalized, tokenized, truecased, encoded and translated segments.
    Decoding must replace strings "@@ " with empty string "".
    '''
    test_cases = {
        "this is an ex@@ ample sent@@ ence .": "this is an example sentence ."
        "esta es una oracion de ej@@ emplo .": "esta es una oracion de ejemplo ."
        "это приме@@ рное пр@@ едложение .": "это примерное предложение ."
        "dies ist ein Bei@@ spiel@@ satz .": "dies ist ein Beispielsatz ."
        "acesta este un ex@@ emplu de propo@@ zitie .": "acesta este un exemplu de propozitie ."
    }

    def test_decode(self):
        for example, decoded in self.test_cases.items():
            self.assertEqual(bpe.decode(source), decoded)
