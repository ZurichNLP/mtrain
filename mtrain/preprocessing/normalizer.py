#!/usr/bin/env python3

from mtrain import constants as C
from mtrain.preprocessing.external import ExternalProcessor

"""
Normalize punctuation using the default Moses normalizer script.
"""


class Normalizer(object):
    """
    Creates a normalizer for processing segment by segment, allowing
    interaction with a normalizer process kept in memory.
    """

    def __init__(self, lang_code):
        """
        @param lang_code language identifier
        """
        arguments = [
            '-l %s' % lang_code,
            '-b',  # disable Perl buffering
            '-q',  # don't report version
        ]   # no aggressive mode '-a' for normalizer

        self._processor = ExternalProcessor(
            command=" ".join([C.MOSES_NORMALIZER] + arguments)
        )

    def close(self):
        del self._processor

    def normalize_punctuation(self, segment):
        """
        Normalizes punctuation characters of a single @param segment.
        """
        normalized_segment = self._processor.process(segment)
        return normalized_segment
