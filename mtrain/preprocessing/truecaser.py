#!/usr/bin/env python3

"""
(De)truecases segments using the default Moses (de)truecaser.
"""

from mtrain import constants as C
from mtrain.preprocessing.external import ExternalProcessor

class Truecaser(object):
    """
    Creates a truecaser which truecases sentences on-the-fly, i.e., allowing
    interaction with a Moses truecaser process kept in memory.
    """

    def __init__(self, path_model):
        """
        @param path_model path to truecasing model trained in `mtrain`
        """
        arguments = [
            '-model %s' % path_model,
            '-b' #disable Perl buffering
        ]

        self._processor = ExternalProcessor(
            command=" ".join([C.MOSES_TRUECASER] + arguments)
        )

    def close(self):
        """
        Deletes object to free up memory.
        """
        del self._processor

    def truecase_segment(self, segment):
        """
        Truecases a single segment.
        """
        return self._processor.process(segment)

    def truecase_tokens(self, tokens, split=True):
        """
        Truecases a list of tokens.
        """
        truecased_string = self.truecase_segment(" ".join(tokens))
        if split:
            return truecased_string.split(" ")
        return truecased_string

class Detruecaser(object):
    """
    Creates a detruecaser which detruecases sentences on-the-fly, i.e., allowing
    interaction with a Moses truecaser process kept in memory.
    """
    def __init__(self):
        """
        Detruecaser that is a script, no model training.
        """
        arguments = [
            '-b' # disable Perl buffering
        ]

        self._processor = ExternalProcessor(
            command=" ".join([C.MOSES_DETRUECASER] + arguments)
        )

    def close(self):
        del self._processor

    def detruecase_segment(self, segment):
        """
        Detruecases a single segment.
        """
        return self._processor.process(segment)

    def detruecase_tokens(self, tokens):
        """
        Detruecases a list of tokens.
        """
        detruecased_segment = self.detruecase_segment(" ".join(tokens))
        return detruecased_segment.split(" ")
