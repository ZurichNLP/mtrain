#!/usr/bin/env python3

"""
(De-)tokenizes segments using the default Moses (de-)tokenizer.
"""

from mtrain import constants as C
from mtrain.preprocessing.external import ExternalProcessor


class Tokenizer(object):
    """
    Creates a tokenizer which tokenizes sentences on-the-fly, i.e., allowing
    interaction with a Moses tokenizer process kept in memory.
    """

    def __init__(self, lang_code, protect=False, protected_patterns_path=None, escape=True):
        """
        @param lang_code language identifier
        @param protect whether the tokenizer should respect patterns that should not be tokenized
        @param protected_patterns_path path to file with protected patterns
        @param escape whether characters that break the Moses decoder should be escaped
        """
        arguments = [
            '-l %s' % lang_code,
            '-b',  # disable Perl buffering
            '-q',  # don't report version
            '-a',  # aggressive mode
        ]

        if protect:
            arguments.append(
                '-protected %s' % protected_patterns_path,  # protect e.g. inline XML, URLs and email
            )

        if not escape:
            arguments.append(
                '-no-escape'  # do not escape reserved characters in Moses
            )

        self._processor = ExternalProcessor(
            command=" ".join([C.MOSES_TOKENIZER] + arguments)
        )

    def close(self):
        del self._processor

    def tokenize(self, segment, split=True):
        """
        Tokenizes a single @param segment.

        @param split determines if a tokenized segmet should be split by a space
        """
        tokenized_segment = self._processor.process(segment)
        if split:
            return tokenized_segment.split(" ")
        return tokenized_segment


class Detokenizer(object):
    """
    Creates a detokenizer which detokenizes lists of tokens on-the-fly, i.e.,
    allowing interaction with a Moses detokenizer process kept in memory.
    """

    def __init__(self, lang_code, uppercase_first_letter=False):
        """
        @param lang_code language identifier
        @param uppercase_first_letter whether or not to uppercase the first
            letter in the detokenized output.
        """
        arguments = [
            '-l %s' % lang_code,
            '-b',  # disable Perl buffering
            '-q',  # don't report version
        ]
        if uppercase_first_letter:
            arguments.append('-u')
        self._processor = ExternalProcessor(
            command=" ".join([C.MOSES_DETOKENIZER] + arguments),
            stream_stderr=True
        )

    def close(self):
        del self._processor

    def detokenize(self, tokens):
        """
        Detokenizes a list of @param tokens into a segment
        """
        return self._processor.process(" ".join(tokens))
