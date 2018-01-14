#!/usr/bin/env python3

from mtrain.constants import *
from mtrain.preprocessing.external import ExternalProcessor

'''
Normalize punctuation using the default Moses normalizer script,
applied for segments using backend nematus.
'''

class Normalizer(object):
    '''
    Creates a normalizer for processing segment per segment, allowing
    interaction with a normalizer process kept in memory.

    Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
    '''

    def __init__(self, lang_code):
        '''
        @param lang_code language identifier

        Scipt reference https://github.com/moses-smt/mosesdecoder/blob/master/scripts/tokenizer/normalize-punctuation.perl:
            Philipp Koehn, Hieu Hoang, Alexandra Birch, Chris Callison-Burch, Marcello Federico, Nicola Bertoldi, Brooke Cowan, Wade Shen, Christine Moran,
            Richard Zens, Chris Dyer, Ondrej Bojar, Alexandra Constantin, and Evan Herbst (2007): Moses: Open Source Toolkit for Statistical Machine
            Translation. In Proceedings of the 45th Annual Meeting of the Association for Computational Linguistics (ACL 2007). Prague, Czech Republic.

        Cf. https://gitlab.cl.uzh.ch/mt/mtrain/blob/nematus/README.md for list of references.
        '''
        arguments = [
            '-l %s' % lang_code,
            '-b', #disable Perl buffering
            '-q', #don't report version
        ]   # no aggressive mode '-a' for normalizer

        self._processor = ExternalProcessor(
            command=" ".join([MOSES_NORMALIZER] + arguments)
        )

    def close(self):
        del self._processor

    def normalize_punctuation(self, segment):
        '''
        Normalizes punctuation of a single @param segment.
        '''

        # e.g. '«' or '»' replaced by '&quot;'
        normalized_segment = self._processor.process(segment)
        return normalized_segment
